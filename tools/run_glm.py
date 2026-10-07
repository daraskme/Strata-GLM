"""Single-GPU GLM server with memory guards and optional Mang-AI GPU queue.

Never stops other applications. The owned process group is released on exit.
Default is a read-only plan; --run explicitly starts the large model.
"""
import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


def resources():
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True, timeout=10)
    free, total = map(float, gpu.stdout.splitlines()[0].split(","))
    return {"ram_available_gib": int(mem["MemAvailable"].split()[0])/1024**2,
            "vram_free_gib": free/1024, "vram_total_gib": total/1024}


def resident_ready(url):
    if not url:
        return True
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--context", type=int, choices=[32768, 65536, 131072, 262144], default=131072)
    p.add_argument("--port", type=int, default=1243)
    p.add_argument("--vram-cap-gib", type=float, default=88)
    p.add_argument("--ram-headroom-gib", type=float, default=20)
    p.add_argument("--min-ram-gib", type=float, default=94)
    p.add_argument("--min-vram-gib", type=float, default=80)
    p.add_argument("--cpu-lane", choices=["auto", "off"], default="auto")
    p.add_argument("--mangai-root", type=Path)
    p.add_argument("--resident-health", default="http://127.0.0.1:1240/health")
    p.add_argument("--benchmark", action="store_true", help="disable conversation prefix reuse")
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    if a.ram_headroom_gib < 12 or a.vram_cap_gib <= 0:
        p.error("RAM headroom must be >=12 GiB and VRAM cap must be positive")
    pack = a.pack.resolve()
    exe = ROOT / "build/strata"
    for path in [exe, pack / "tokenizer/tokenizer.json", pack / "tokenizer/chat_template.jinja"]:
        if not path.is_file():
            p.error(f"missing {path}")
    info = resources()
    if a.vram_cap_gib > info["vram_total_gib"] - 4:
        p.error("VRAM cap must leave at least 4 GiB outside the cap")
    plan = {"pack": str(pack), "context": a.context, "port": a.port, "resources": info,
            "resident_ready": resident_ready(a.resident_health), "cpu_lane": a.cpu_lane,
            "vram_cap_gib": a.vram_cap_gib, "ram_headroom_gib": a.ram_headroom_gib}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not a.run:
        return
    if not plan["resident_ready"]:
        p.error("resident A1 is not ready; start it first")
    if info["ram_available_gib"] < a.min_ram_gib or info["vram_free_gib"] < a.min_vram_gib:
        p.error("insufficient free memory; close VMs/other large models yourself and retry")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", a.port))
    runtime = ROOT / "build/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    lease = nullcontext()
    if a.mangai_root:
        sys.path.insert(0, str(a.mangai_root.resolve() / "manga-studio/python"))
        from mangai_gpu import GpuLease
        lease = GpuLease("coder", "GLM-5.3 Flash / custom Strata", root=a.mangai_root.resolve() / "work/gpu",
                         limits=(a.min_ram_gib, a.min_vram_gib), timeout=60,
                         progress=lambda text: print(text, flush=True))
    env = os.environ.copy()
    # Do not inherit experimental overrides from another Strata model/session.
    for key in list(env):
        if key.startswith("STRATA_GLM_"):
            del env[key]
    env.update(STRATA_GLM_SPLIT="0", STRATA_GLM_NO_MTP="1", STRATA_GLM_VRAM_GB=str(a.vram_cap_gib),
               STRATA_GLM_RESERVE_MB="2048", STRATA_GLM_RAM_HEADROOM_GB=str(a.ram_headroom_gib),
               STRATA_GLM_TIMING="1", STRATA_GLM_POOL_STATS="1",
               STRATA_GLM_USAGE=str(pack / "expert_usage.txt"), OMP_NUM_THREADS="8")
    if a.cpu_lane == "off":
        env["STRATA_GLM_CPU_LANE"] = "0"
    if a.benchmark:
        env["STRATA_GLM_NO_REUSE"] = "1"
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    config = {"exe": str(exe), "args": ["--glm-pack", str(pack), "--max-context", str(a.context)],
              "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"), "gpu": [0],
              "model_name": "glm-5.3-flash-iq4-mangai", "host": "127.0.0.1",
              "log": str(runtime / f"{run_id}-engine.log"), "thinking_budget": 256}
    cfg_path = runtime / f"{run_id}.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics_path = runtime / f"{run_id}-resources.jsonl"
    child = None
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with open(runtime / "server.lock", "a") as lock, lease:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            Path("/proc/self/oom_score_adj").write_text("500")
        except OSError:
            pass
        print(f"GPU・RAM負荷開始。context={a.context}。ログ: {runtime / run_id}", flush=True)
        child = subprocess.Popen([sys.executable, "-u", str(ROOT / "serve/server.py"), "--engine", "strata",
                                  "--config", str(cfg_path), "--port", str(a.port)], env=env, start_new_session=True)
        health_failures = 0
        try:
            with open(metrics_path, "a", encoding="utf-8") as metrics:
                while child.poll() is None and not stopped:
                    info = resources()
                    ready = resident_ready(a.resident_health)
                    health_failures = 0 if ready else health_failures + 1
                    metrics.write(json.dumps(dict(time=time.time(), resident_ready=ready, **info)) + "\n")
                    metrics.flush()
                    if info["ram_available_gib"] < 8 or info["vram_free_gib"] < 1 or health_failures >= 3:
                        raise RuntimeError("memory/A1 health guard stopped this GLM process")
                    for _ in range(10):
                        if stopped or child.poll() is not None:
                            break
                        time.sleep(1)
        finally:
            # The API spawns the engine; both inherit this owned session/group.
            # Clean the group even if the API has already crashed.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if child.returncode and not stopped:
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
