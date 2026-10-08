"""GPU diagnostic parsing: filtered accounting and refusal of misleading data."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from analyze_glm_gpu_profile import analyze

HEADER = '# glm_gpu_profile_v1\nsample,position,layer,phase,milliseconds\n'


class ProfileTests(unittest.TestCase):
    def parse(self, rows, **kwargs):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'profile.csv'
            path.write_text(HEADER + rows)
            return analyze(path, **kwargs)

    def test_prompt_filter_and_per_pass_not_per_interval_averaging(self):
        data = self.parse('1,3,3,moe_cpu_wait,100\n'
            '2,4,3,moe_cpu_wait,2\n2,4,3,moe_down,1\n2,4,4,moe_cpu_wait,4\n'
            '3,5,3,moe_cpu_wait,6\n3,5,3,moe_down,1\n3,5,4,moe_cpu_wait,2\n', min_position=4)
        self.assertEqual(data['fast_passes'], 2)
        self.assertEqual(data['stream_ms_mean'], 8)
        self.assertEqual(data['phases_ms_per_pass']['moe_cpu_wait'], 7)
        self.assertEqual(data['layers']['3']['stream_ms_per_pass'], 5)
        self.assertEqual(data['top_cpu_wait_layers'], ['3', '4'])
        self.assertEqual(data['consumed_position_range'], [4, 5])

    def test_rejects_failed_duplicate_and_truncated_measurements(self):
        for rows in ('1,4,3,moe_cpu_wait,nan\n', '1,4,3,moe_cpu_wait,-1\n',
            '1,4,3,moe_cpu_wait,1\n1,4,3,moe_cpu_wait,2\n', '1,4,3,moe_cpu_wait,\n',
            '2,4,3,moe_cpu_wait,1\n1,5,3,moe_cpu_wait,2\n',
            '1,4,3,moe_cpu_wait,1\n1,5,4,moe_down,2\n', '1,4,3,moe_cpu_wait,1.2'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.parse(rows)

    def test_empty_or_invalid_range_is_not_a_zero_cost_result(self):
        for kwargs in ({'min_position': 9}, {'min_position': -1}, {'min_position': 8, 'max_position': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.parse('1,4,3,moe_cpu_wait,1\n', **kwargs)


if __name__ == '__main__':
    unittest.main()
