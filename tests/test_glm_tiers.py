import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from analyze_glm_tiers import analyze


class TierTests(unittest.TestCase):
    def test_cpu_ram_is_distinct_from_gpu_hit_and_ssd_miss(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'routes.0'
            path.write_text('3 0 1 2 3 4 5 6 7 3 4 24\n3 0 1 2 3 4 5 6 7 0 0 0\n')
            result = analyze([path])
            self.assertEqual(result['counts'], {'vram_hit': 11, 'ram_to_gpu': 2, 'ram_cpu': 2, 'ssd_miss': 1})
            self.assertEqual(result['selected_experts'], 16)
            self.assertEqual(result['layers']['3']['top_experts'][0], {'id': 0, 'selections': 2})
            self.assertEqual(sum(result['fractions'].values()), 1)

    def test_bad_mask_is_not_silently_classified(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'routes.0'
            path.write_text('3 0 1 2 3 4 5 6 7 3 1 0\n')
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                analyze([path])

    def test_interrupted_trace_requires_explicit_partial_mode(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'routes.0'
            path.write_text('3 0 1 2 3 4 5 6 7 0 0 0\n4 3 2')
            with self.assertRaisesRegex(ValueError, 'expected 12'):
                analyze([path])
            result = analyze([path], allow_truncated_tail=True)
            self.assertEqual(result['routes'], 1)
            self.assertEqual(result['truncated_tails'], [{'path': str(path), 'line': 2}])
            path.write_text('3 0 1 2 3 4 5 6 7 3 1 0\n')
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                analyze([path], allow_truncated_tail=True)


if __name__ == '__main__':
    unittest.main()
