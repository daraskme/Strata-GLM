"""Fresh long-context retrieval plus sustained generation; saves all evidence.

This never runs generated code. The three lookup answers are checked separately
from code completion and FP8-equivalence, neither of which this fixture proves.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ENGINE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ENGINE / 'tools'), str(ENGINE / 'serve')]
from strata_tokenizer import Tokenizer
from frontend import ChatTemplate, openai_to_messages

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:1244')
    p.add_argument('--model')
    p.add_argument('--context', type=int, default=262144)
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--output-tokens', type=int, default=512)
    args = p.parse_args()
    if args.tokens < 256 or args.output_tokens < 1 or args.tokens + args.output_tokens > args.context:
        p.error('positive output and at least 256 input tokens must fit within context')
    args.output.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(args.url.rstrip('/') + '/v1/models', timeout=10) as response:
        models = json.load(response)['data']
    ids = [item['id'] for item in models]
    model = args.model or (ids[0] if len(ids) == 1 else None)
    if model not in ids:
        p.error('select an available --model, especially for a multi-model broker')
    tok = Tokenizer.from_pack(args.pack / 'tokenizer')
    tpl = ChatTemplate(args.pack / 'tokenizer/chat_template.jinja')
    keys = {'早朝': 'R7K2-5941', '正午': 'P3X8-2167', '夕方': 'W9C4-8032'}
    lines = [f'記録{i:05d}: 資料箱は棚{i%97:02d}にあり、担当番号は{(i*7919)%100003:06d}です。'
             for i in range(args.tokens//8+100)]
    def request(n):
        rows = lines[:n].copy()
        for fraction, (key, value) in zip((.1, .5, .9), keys.items()):
            rows[int(n*fraction)] += f' 特別記録: {key}の合言葉は{value}。'
        content = ('次の記録を読み、末尾の質問に答えてください。\n' + '\n'.join(rows) +
                   '\n質問: 最初に早朝・正午・夕方の合言葉だけをJSONオブジェクトで返してください。'
                   'キーは「早朝」「正午」「夕方」。その後、これらの記録から合言葉を検索する'
                   'Python標準ライブラリだけのプログラムを、型注釈・説明・境界条件のunittestを8件以上含めて完全に書いてください。')
        return {'model': model, 'messages': [{'role': 'user', 'content': content}],
                'temperature': 0, 'max_tokens': args.output_tokens, 'reasoning_effort': 'none', 'stream': False, 'seed': 42}
    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        return len(tok.encode(tpl.render(messages, tools, **kwargs), parse_special=True))
    lo, hi = 10, len(lines)
    while lo < hi:
        mid = (lo+hi+1)//2
        if count(request(mid)) <= args.tokens:
            lo = mid
        else:
            hi = mid-1
    req = request(lo)
    n = count(req)
    assert n + args.output_tokens <= args.context
    (args.output / 'request.json').write_text(json.dumps(req, ensure_ascii=False, indent=2))
    print(json.dumps({'phase': 'request', 'prompt_tokens': n, 'max_output_tokens': args.output_tokens}, ensure_ascii=False), flush=True)
    start = time.monotonic()
    http = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(req).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(http, timeout=5400) as response:
        result = json.load(response)
    elapsed = time.monotonic()-start
    (args.output / 'response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    text = (result['choices'][0]['message'].get('content') or '').strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    try:
        answer, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        answer = None
    summary = {'prompt_tokens': n, 'usage': result['usage'], 'timings': result['timings'],
               'elapsed_seconds': elapsed, 'answer': answer, 'expected': keys,
               'retrieval_passed': answer == keys,
               'tokenizer_matched': n == result['usage']['prompt_tokens'],
               'prefix_reused': result['timings']['cache_n'],
               'finish_reason': result['choices'][0]['finish_reason'],
               'note': 'Checks lookup values and throughput only. Generated program not executed; truncated output is not a coding success. No FP8 quality comparison.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary['retrieval_passed'] or not summary['tokenizer_matched'] or summary['prefix_reused']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
