"""CPU-only single-card orchestration safety tests."""
from argparse import Namespace
from contextlib import ExitStack
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import restart_p0b_seed42_gpu1 as launch
import bootstrap_p0b_seed42 as bootstrap


class GPU1RestartSafety(unittest.TestCase):
    def environment_mock(self):
        manifest = launch.native_launcher.legacy.read(launch.native_launcher.BASE / 'recovery_manifest.json')
        return manifest['current_gpu1_uuid']

    def test_gpu0_compute_does_not_block_gpu1(self):
        uuid = self.environment_mock()
        with patch.object(launch.fcntl, 'flock'), patch.object(launch.subprocess, 'check_output', side_effect=[
            f'1, {uuid}\n', '123, GPU-07abbe75-7a55-4623-70aa-a9572c83957e\n']):
            locks = launch.acquire_gpu1_only()
            self.assertEqual(len(locks), 1)
            self.assertTrue(locks[0].name.endswith('research2_gpu1.lock'))
            for lock in locks:
                lock.close()

    def test_gpu1_compute_is_rejected(self):
        uuid = self.environment_mock()
        with patch.object(launch.fcntl, 'flock'), patch.object(launch.subprocess, 'check_output', side_effect=[f'1, {uuid}\n', f'123, {uuid}\n']):
            with self.assertRaisesRegex(RuntimeError, 'GPU1 is busy'):
                launch.acquire_gpu1_only()

    def test_only_output_directories_change(self):
        args = Namespace(batch_size=4, epochs=120, checkpoints_dir='old1', logs_dir='old2', results_dir='old3')
        with patch.object(bootstrap, 'OUTPUT', launch.OUTPUT):
            revised = bootstrap.output_config(args)
            self.assertEqual(revised.batch_size, 4)
            self.assertEqual(revised.epochs, 120)
            self.assertTrue(revised.logs_dir.endswith('run_02/logs'))


if __name__ == '__main__':
    unittest.main()
