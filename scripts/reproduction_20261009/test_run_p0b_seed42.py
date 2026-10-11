"""CPU-only protocol/output and queue safety tests; no native training."""
from argparse import Namespace
from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch
import run_p0b_seed42 as launch
import bootstrap_p0b_seed42 as bootstrap


class Seed42Safety(unittest.TestCase):
    def test_only_output_config_changes(self):
        spec = importlib.util.spec_from_file_location('r1_config_for_test',
            launch.BASE / 'source/configs/__init__.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = vars(module.load_config('MOSI'))
        revised = vars(bootstrap.output_config(Namespace(**original)))
        changed = {key for key in original if original[key] != revised[key]}
        self.assertEqual(changed, {'checkpoints_dir', 'logs_dir', 'results_dir'})
        self.assertNotEqual(launch.OUTPUT, launch.legacy.OUTPUT)

    def test_failed_dependency_does_not_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'Research2 failed'):
            launch.dependency_ready({'phase':'failed'})

    def test_live_dependency_waits(self):
        with patch.object(launch, 'live', return_value=True):
            self.assertFalse(launch.dependency_ready({'phase':'running', 'pid':123}))

    def test_lost_dependency_fails(self):
        with patch.object(launch, 'live', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'controller lost'):
                launch.dependency_ready({'phase':'running', 'pid':123})

    def test_incomplete_completion_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'lacks required jobs'):
            launch.dependency_ready({'phase':'complete', 'completed':[]})

    def test_source_fingerprints(self):
        launch.legacy.verify_sources(launch.legacy.read(launch.BASE / 'recovery_manifest.json'))


if __name__ == '__main__':
    unittest.main()
