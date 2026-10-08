"""Launcher guards and lease ownership; no model or GPU allocations."""
from contextlib import redirect_stdout, redirect_stderr
import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_glm


class LauncherTests(unittest.TestCase):
    def test_cpu_plan_rejects_missing_counts_and_impossible_assignments(self):
        self.assertEqual(run_glm.parse_cpu_plan('001122334'), '001122334')
        self.assertEqual(run_glm.parse_cpu_plan('012345678'), '012345678')
        for invalid in ('00112233', '0011223340', '101122334', '003122334', '00x122334', '００１１２２３３４'):
            with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                run_glm.parse_cpu_plan(invalid)

    def test_native_reasoning_is_default_and_smoke_cap_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            (root / 'build').mkdir()
            (root / 'build/strata').touch()
            pack = root / 'pack'
            (pack / 'tokenizer').mkdir(parents=True)
            for name in ('tokenizer.json', 'chat_template.jinja'):
                (pack / 'tokenizer' / name).touch()
            info = {'ram_available_gib': 110, 'vram_free_gib': 85, 'vram_total_gib': 95.6}
            with patch.object(run_glm, 'ROOT', root), patch.object(run_glm, 'resources', return_value=info), patch.object(run_glm, 'resident_ready', return_value=True):
                for extra, expected in (([], 0), (['--thinking-budget', '256'], 256)):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        run_glm.main(['--pack', str(pack), *extra])
                    self.assertEqual(json.loads(output.getvalue())['thinking_budget'], expected)
                    self.assertIsNone(json.loads(output.getvalue())['profile_gpu_after'])
                output = io.StringIO()
                with redirect_stdout(output):
                    run_glm.main(['--pack', str(pack), '--profile-gpu-after', '142'])
                self.assertEqual(json.loads(output.getvalue())['profile_gpu_after'], 142)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    run_glm.main(['--pack', str(pack), '--profile-gpu-after', '-1'])
                output = io.StringIO()
                with redirect_stdout(output):
                    run_glm.main(['--pack', str(pack), '--cpu-plan', '001122334'])
                self.assertEqual(json.loads(output.getvalue())['cpu_plan'], '001122334')
                output = io.StringIO()
                with redirect_stdout(output):
                    run_glm.main(['--pack', str(pack), '--prefetch-experts', '2', '--lookahead-layers', '4'])
                planned = json.loads(output.getvalue())
                self.assertEqual((planned['prefetch_experts'], planned['lookahead_layers']), (2, 4))
                for flags in (['--prefetch-experts', '3'], ['--lookahead-layers', '5'], ['--prefetch-experts', '-1']):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        run_glm.main(['--pack', str(pack), *flags])
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    run_glm.main(['--pack', str(pack), '--cpu-plan', '001122334', '--cpu-lane', 'off'])
                self.assertEqual(list((root / 'build').iterdir()), [root / 'build/strata'])

    def test_fixed_budgets_preserve_headroom_and_reject_nonfinite_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            (root / 'build').mkdir()
            (root / 'build/strata').touch()
            pack = root / 'pack'
            (pack / 'tokenizer').mkdir(parents=True)
            for name in ('tokenizer.json', 'chat_template.jinja'):
                (pack / 'tokenizer' / name).touch()
            info = {'ram_available_gib': 109, 'vram_free_gib': 85, 'vram_total_gib': 95.6}
            with patch.object(run_glm, 'ROOT', root), patch.object(run_glm, 'resources', return_value=info), patch.object(run_glm, 'resident_ready', return_value=True):
                output = io.StringIO()
                with redirect_stdout(output):
                    run_glm.main(['--pack', str(pack), '--expert-pool-gib', '67', '--ram-tier-gib', '92'])
                plan = json.loads(output.getvalue())
                self.assertEqual((plan['expert_pool_gib'], plan['ram_tier_gib']), (67, 92))
                for flags in (['--ram-tier-gib', '94'], ['--expert-pool-gib', '91'], ['--ram-tier-gib', 'nan'], ['--expert-pool-gib', 'inf'], ['--ram-tier-gib', '0']):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        run_glm.main(['--pack', str(pack), *flags])
                    self.assertEqual(error.exception.code, 2)
                self.assertEqual(list((root / 'build').iterdir()), [root / 'build/strata'])

    def test_external_lease_must_be_inherited_locked_and_for_this_queue(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            (root / 'work/gpu').mkdir(parents=True)
            with (root / 'work/gpu/heavy.lock').open('a+b') as lock:
                with patch.dict(os.environ, {'MANGAI_GPU_LEASE_FD': str(lock.fileno())}):
                    with self.assertRaisesRegex(ValueError, 'not locked'):
                        run_glm.validate_external_lease(root)
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    self.assertEqual(run_glm.validate_external_lease(root), lock.fileno())
                    with (root / 'other.lock').open('a+b') as other:
                        with patch.dict(os.environ, {'MANGAI_GPU_LEASE_FD': str(other.fileno())}):
                            with self.assertRaisesRegex(ValueError, 'different queue'):
                                run_glm.validate_external_lease(root)

    def test_external_lease_requires_broker_descriptor(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                run_glm.validate_external_lease(ROOT)


if __name__ == '__main__':
    unittest.main()
