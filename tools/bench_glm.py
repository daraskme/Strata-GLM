"""Explicit live API validation. Saves actual inputs, outputs, token counts and timings.

Run after tools/run_glm.py. Does not start/stop any model or execute returned tools.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "serve")]
from strata_tokenizer import Tokenizer
from frontend import ChatTemplate, openai_to_messages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:1243")
    p.add_argument("--case", choices=["smoke", "needle"], default="smoke")
    p.add_argument("--tokens", type=int, default=32000)
    a = p.parse_args()
    tok = Tokenizer.from_pack(a.pack / "tokenizer")
    tpl = ChatTemplate(a.pack / "tokenizer/chat_template.jinja")
    out = ROOT / "build/validation" / (time.strftime("%Y%m%dT%H%M%S") + "-" + a.case)
    out.mkdir(parents=True, exist_ok=False)

    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        prompt = tpl.render(messages, tools, **kwargs)
        return len(tok.encode(prompt, parse_special=True))

    def ask(label, content, *, tools=None, max_tokens=256):
        req = {"model": "glm-5.3-flash-iq4-mangai", "messages": [{"role": "user", "content": content}],
               "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "none", "stream": False}
        if tools:
            req["tools"] = tools
        n = count(req)
        (out / f"{label}-request.json").write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "start", "case": label, "prompt_tokens": n, "output": str(out)}, ensure_ascii=False), flush=True)
        start = time.monotonic()
        request = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=7200) as response:
            result = json.load(response)
        elapsed = time.monotonic()-start
        (out / f"{label}-response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        actual = result["usage"]["prompt_tokens"]
        if actual != n:
            raise RuntimeError(f"tokenizer mismatch: local {n} / API {actual}")
        summary = {"case": label, "prompt_tokens": n, "usage": result["usage"], "timings": result.get("timings"),
                   "elapsed_s": elapsed, "finish_reason": result["choices"][0]["finish_reason"]}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return result["choices"][0]["message"], summary

    records = []
    if a.case == "smoke":
        msg, record = ask("japanese", "次の文だけをそのまま返してください。説明は不要です。\n桜の花が咲きました。", max_tokens=64)
        record["passed"] = (msg.get("content") or "").strip() == "桜の花が咲きました。"
        records.append(record)
        msg, record = ask("code", "Pythonのclamp(x, lo, hi)関数を書いてください。xをlo以上hi以下に制限して返します。lo<=hiは保証済み。関数定義だけを出力し、説明・import・型注釈は付けないでください。", max_tokens=256)
        source = (msg.get("content") or "").strip()
        source = re.sub(r"^```(?:python)?\s*|\s*```$", "", source).strip()
        try:
            tree = ast.parse(source)
            allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Call, ast.Name,
                       ast.Load, ast.If, ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Constant)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "clamp":
                raise ValueError("expected one clamp function")
            for node in ast.walk(tree):
                if not isinstance(node, allowed):
                    raise ValueError("unsupported code shape")
                if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in {"min", "max"}):
                    raise ValueError("unexpected call")
                if isinstance(node, ast.Name) and node.id not in {"x", "lo", "hi", "min", "max"}:
                    raise ValueError("unexpected identifier")
            scope = {"__builtins__": {"min": min, "max": max}}
            exec(compile(tree, "<validated-clamp>", "exec"), scope)
            cases = [(-1,0,10,0), (5,0,10,5), (12,0,10,10), (0,0,10,0), (10,0,10,10), (-5,-4,-2,-4), (1.5,1,2,1.5), (7,3,3,3)]
            record["passed"] = all(scope["clamp"](x,l,h) == expected for x,l,h,expected in cases)
        except (SyntaxError, ValueError, TypeError, KeyError) as exc:
            record.update(passed=False, error=str(exc))
        records.append(record)
        tools = [{"type": "function", "function": {"name": "lookup_asset", "description": "Find a project asset by ID.",
                  "parameters": {"type": "object", "properties": {"asset_id": {"type": "string"}, "revision": {"type": "integer"}},
                                 "required": ["asset_id", "revision"]}}}]
        msg, record = ask("tool", "lookup_assetツールを呼び出し、asset_idがmanga-007、revisionが3の素材を確認してください。", tools=tools, max_tokens=256)
        calls = msg.get("tool_calls") or []
        record["passed"] = len(calls) == 1 and calls[0]["function"]["name"] == "lookup_asset" and json.loads(calls[0]["function"]["arguments"]) == {"asset_id":"manga-007", "revision":3}
        records.append(record)
    else:
        # Three needles with unrelated values among numbered distractors; NO_REUSE
        # in the launcher ensures these are actual fresh prompt tokens.
        keys = {"早朝": "R7K2-5941", "正午": "P3X8-2167", "夕方": "W9C4-8032"}
        lines = [f"記録{i:05d}: 資料箱は棚{i%97:02d}にあり、担当番号は{(i*7919)%100003:06d}です。" for i in range(a.tokens//8+100)]

        def make(n):
            rows = lines[:n].copy()
            for fraction, (key, value) in zip([.1,.5,.9], keys.items()):
                rows[int(n*fraction)] += f" 特別記録: {key}の合言葉は{value}。"
            return "次の記録を読み、末尾の質問に答えてください。\n" + "\n".join(rows) + '\n質問: 早朝・正午・夕方の合言葉をJSONオブジェクトで返してください。キーは「早朝」「正午」「夕方」。説明は不要です。'
        lo, hi = 10, len(lines)
        while lo < hi:
            mid = (lo+hi+1)//2
            req = {"messages":[{"role":"user", "content":make(mid)}], "reasoning_effort":"none"}
            if count(req) <= a.tokens:
                lo = mid
            else:
                hi = mid-1
        msg, record = ask(f"needle-{a.tokens}", make(lo), max_tokens=256)
        content = (msg.get("content") or "").strip()
        try:
            answer = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip())
        except ValueError:
            answer = None
        record.update(passed=answer == keys, expected=keys, answer=answer)
        records.append(record)
    (out / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": records, "output": str(out)}, ensure_ascii=False), flush=True)
    if not all(x["passed"] for x in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
