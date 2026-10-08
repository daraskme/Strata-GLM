"""Summarize diagnostic CUDA-stream intervals; not end-to-end token latency.

The main stream includes time waiting for host CPU/disk replies. It excludes the
embedding, output head, fast_boundary and HTTP work outside the marked region.
Instrumentation itself adds overhead. Position is the consumed token's index;
for one request, --min-position PROMPT_TOKENS selects generated-token passes
(the first output is produced by the final prompt token and is not included).
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def analyze(path, min_position=0, max_position=None):
    if min_position < 0 or (max_position is not None and max_position < min_position):
        raise ValueError('invalid inclusive position range')
    phase_ms = defaultdict(float)
    layer_phases = defaultdict(lambda: defaultdict(float))
    layer_samples = defaultdict(lambda: defaultdict(float))
    sample_ms = defaultdict(float)
    seen = set()
    positions = {}
    selected = 0
    with Path(path).open(newline='') as source:
        if source.readline().strip() != '# glm_gpu_profile_v1':
            raise ValueError('unsupported GPU profile version')
        def complete_lines():
            for number, line in enumerate(source, 2):
                if not line.endswith('\n'):
                    raise ValueError(f'{path}:{number}: truncated final line')
                yield line
        reader = csv.DictReader(complete_lines())
        if reader.fieldnames != ['sample', 'position', 'layer', 'phase', 'milliseconds']:
            raise ValueError('unexpected CSV columns')
        previous_sample = 0
        for number, row in enumerate(reader, 3):
            try:
                sample, position, layer = (int(row[key]) for key in ('sample', 'position', 'layer'))
                phase, ms = row['phase'], float(row['milliseconds'])
                if None in row or sample <= 0 or position < 0 or layer < -1 or not re.fullmatch(r'[a-z_]+', phase):
                    raise ValueError('invalid row')
                if not math.isfinite(ms) or ms < 0:
                    raise ValueError('invalid event measurement')
                if sample < previous_sample:
                    raise ValueError('sample order regressed')
                previous_sample = sample
                if sample in positions and positions[sample] != position:
                    raise ValueError('mixed token positions in one fast pass; unsupported trace')
                positions[sample] = position
                key = (sample, layer, phase)
                if key in seen:
                    raise ValueError('duplicate interval in fast pass; unsupported trace')
                seen.add(key)
            except (TypeError, ValueError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
            if position < min_position or (max_position is not None and position > max_position) or phase == 'start':
                continue
            selected += 1
            sample_ms[sample] += ms
            phase_ms[phase] += ms
            layer_phases[layer][phase] += ms
            layer_samples[layer][sample] += ms
    count = len(sample_ms)
    if not count:
        raise ValueError('no intervals in the selected position range')
    phases = {key: value / count for key, value in sorted(phase_ms.items())}
    layers = {str(layer): {'samples': len(layer_samples[layer]),
        'stream_ms_per_pass': sum(values.values()) / count,
        'stream_ms_p95': percentile(list(layer_samples[layer].values()), .95),
        'phases_ms_per_pass': {key: value / count for key, value in sorted(values.items())}}
        for layer, values in sorted(layer_phases.items())}
    return {'input': str(path), 'min_position': min_position, 'max_position': max_position,
        'intervals': selected, 'fast_passes': count,
        'consumed_position_range': [min(positions[s] for s in sample_ms), max(positions[s] for s in sample_ms)],
        'stream_ms_mean': sum(sample_ms.values()) / count,
        'stream_ms_p50': percentile(list(sample_ms.values()), .5),
        'stream_ms_p95': percentile(list(sample_ms.values()), .95),
        'phases_ms_per_pass': phases, 'layers': layers,
        'top_cpu_wait_layers': sorted(layers, key=lambda key: layers[key]['phases_ms_per_pass'].get('moe_cpu_wait', 0), reverse=True)[:10],
        'note': __doc__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--min-position', type=int, default=0)
    parser.add_argument('--max-position', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.dumps(analyze(args.trace, args.min_position, args.max_position), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as target:
            target.write(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
