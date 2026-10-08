"""Summarize GLM route traces without changing routing, models or usage profiles.

Trace format from STRATA_GLM_ROUTE_LOG: layer, eight expert IDs, RAM-fetch mask,
SSD-miss mask, CPU-compute mask. The last mask is RAM data computed on the CPU;
it must not be counted as a GPU hit or an SSD miss.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def analyze(paths, allow_truncated_tail=False):
    tiers = Counter()
    layers = defaultdict(Counter)
    experts = defaultdict(Counter)
    routes = 0
    truncated_tails = []
    for path in paths:
        with Path(path).open() as source:
            for number, line in enumerate(source, 1):
                if not line.strip() or line.startswith('#'):
                    continue
                if allow_truncated_tail and not line.endswith('\n'):
                    truncated_tails.append({'path': str(path), 'line': number})
                    continue
                try:
                    values = list(map(int, line.split()))
                    if len(values) != 12:
                        raise ValueError('expected 12 integers')
                    layer, *rest = values
                    ids, masks = rest[:8], rest[8:]
                    fetch, miss, cpu = masks
                    if layer < 0 or len(set(ids)) != 8 or any(e < 0 or e >= 288 for e in ids):
                        raise ValueError('invalid layer/expert IDs')
                    if any(mask < 0 or mask > 255 for mask in masks) or (fetch & miss) or (fetch & cpu) or (miss & cpu):
                        raise ValueError('invalid or overlapping masks')
                except ValueError as error:
                    raise ValueError(f'{path}:{number}: {error}') from error
                routes += 1
                for i, expert in enumerate(ids):
                    bit = 1 << i
                    tier = 'ssd_miss' if miss & bit else 'ram_cpu' if cpu & bit else 'ram_to_gpu' if fetch & bit else 'vram_hit'
                    tiers[tier] += 1
                    layers[layer][tier] += 1
                    experts[layer][expert] += 1
    total = routes * 8
    def counts(value):
        return {key: value[key] for key in ('vram_hit', 'ram_cpu', 'ram_to_gpu', 'ssd_miss')}
    return {'routes': routes, 'selected_experts': total, 'counts': counts(tiers), 'truncated_tails': truncated_tails,
            'fractions': {key: n / total if total else 0 for key, n in counts(tiers).items()},
            'layers': {str(layer): {'counts': counts(value),
                'top_experts': [{'id': e, 'selections': n} for e, n in experts[layer].most_common(16)]}
                for layer, value in sorted(layers.items())},
            'note': 'Observed complete routes only. Fractions are selections, not bytes or elapsed time. Process termination can discard buffered records. SSD misses may be satisfied by lookahead/page cache; no claim about physical SSD reads.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-truncated-tail', action='store_true', help='Exclude and report the final non-newline-terminated record after a stopped engine')
    args = parser.parse_args()
    data = json.dumps(analyze(args.traces, args.allow_truncated_tail), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as target:
            target.write(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
