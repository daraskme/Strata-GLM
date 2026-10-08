"""Download a pinned Hugging Face file manifest with size/SHA256 verification.

Uses an existing HF token; never accepts gated-model terms or prints credentials.
Credentials are stripped on redirects leaving huggingface.co.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def hf_token(env_file=None):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and env_file and env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
                token = value.strip().strip("\"'")
    if not token:
        candidates = [Path(os.environ["HF_TOKEN_PATH"])] if os.environ.get("HF_TOKEN_PATH") else []
        candidates += [Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"]
        for path in candidates:
            if path.is_file():
                token = path.read_text().strip()
                break
    return token


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        out = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlparse(newurl).hostname != "huggingface.co":
            out.remove_header("Authorization")
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--workers", type=int, choices=[1, 2, 3, 4, 5], default=3)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for file in manifest["files"]:
        path = (root / file["rfilename"]).resolve()
        if not path.is_relative_to(root):
            p.error("manifest contains an unsafe path")
    required = 0
    for file in manifest["files"]:
        path = root / file["rfilename"]
        partial = path.with_suffix(path.suffix + ".partial")
        present = path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0
        required += max(0, file["size"] - present)
    if shutil.disk_usage(root).free < required + 8 * 2**30:
        p.error("insufficient disk space for files plus 8 GiB reserve")
    token = hf_token(a.env_file)
    headers = {"Authorization": "Bearer " + token} if token else {}
    lock = threading.Lock()
    progress = {}

    def fetch(file):
        name, size, sha = file["rfilename"], file["size"], file["lfs"]["sha256"]
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".partial")
        for attempt in range(5):
            digest = hashlib.sha256()
            source = path if path.exists() else part
            offset = source.stat().st_size if source.exists() else 0
            if offset > size:
                raise RuntimeError("oversized file: " + name)
            if offset:
                with source.open("rb") as stream:
                    while chunk := stream.read(8 * 2**20):
                        digest.update(chunk)
            with lock:
                progress[name] = offset
            if path.exists():
                if offset != size or digest.hexdigest() != sha:
                    raise RuntimeError("existing final file is unverified: " + name)
                return name
            try:
                if offset < size:
                    url = "https://huggingface.co/" + manifest["repo"] + "/resolve/" + manifest["revision"] + "/" + name
                    req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={offset}-"})
                    opener = urllib.request.build_opener(Redirect())
                    with opener.open(req, timeout=90) as response:
                        if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                            raise RuntimeError("server refused a verified range resume")
                        with part.open("ab" if offset else "wb") as stream:
                            while chunk := response.read(8 * 2**20):
                                stream.write(chunk)
                                digest.update(chunk)
                                offset += len(chunk)
                                with lock:
                                    progress[name] = offset
                if offset != size or digest.hexdigest() != sha:
                    raise RuntimeError("download size/SHA256 mismatch: " + name)
                part.replace(path)
                print(json.dumps({"verified": name, "bytes": size}), flush=True)
                return name
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                # Exceptions can embed signed URLs: print only class/status.
                print(json.dumps({"retry": attempt+1, "file": name, "error_type": type(exc).__name__, "http": getattr(exc, "code", None)}), flush=True)
                if getattr(exc, "code", None) in {401, 403} or attempt == 4:
                    raise RuntimeError("download failed; check HF access for " + name) from None
                time.sleep(min(10, attempt+1))

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        tasks = [pool.submit(fetch, f) for f in manifest["files"]]
        while not all(t.done() for t in tasks):
            with lock:
                complete = sum(progress.values())
            print(json.dumps({"downloaded_gib": round(complete/2**30, 2), "total_gib": round(sum(f["size"] for f in manifest["files"])/2**30, 2)}), flush=True)
            time.sleep(10)
        for task in tasks:
            task.result()
    (root / "verified-model.json").write_text(json.dumps({**manifest, "verified": True, "verified_at": time.time()}, indent=2))
    print("All model files verified", flush=True)


if __name__ == "__main__":
    main()
