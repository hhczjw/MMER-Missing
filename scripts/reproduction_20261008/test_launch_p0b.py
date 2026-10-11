"""CPU-only safety tests; never launch a subprocess or initialize CUDA."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch_p0b as launch


class LaunchSafetyTests(unittest.TestCase):
    def test_pending_review_prevents_resource_acquisition(self):
        with patch('sys.argv', ['launch_p0b.py', '--launch']), \
             patch.object(launch, 'acquire_idle_gpus') as acquire:
            with self.assertRaisesRegex(RuntimeError, 'Pending historical'):
                launch.main()
            acquire.assert_not_called()

    def test_existing_output_is_never_modified(self):
        manifest = launch.read(launch.BASE / 'recovery_manifest.json')
        expected = manifest['expected_current_environment']
        cpu = {k:expected[k] for k in ['python','packages','cuda','cudnn']}
        cpu['cuda_initialized'] = False
        original_read = launch.read
        def fake_read(path):
            if path.name == 'environment_review.json':
                return dict(state='user_authorized_candidate', evidence_or_user_direction='TEST ONLY; not real approval')
            return original_read(path)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            output = Path(temp) / 'existing_run'
            output.mkdir()
            status = output / 'status.json'
            old = '{"phase":"train","user_data":"keep"}'
            status.write_text(old)
            lock = stack.enter_context((Path(temp) / 'lock').open('a'))
            stack.enter_context(patch('sys.argv', ['launch_p0b.py', '--launch']))
            stack.enter_context(patch.object(launch, 'OUTPUT', output))
            stack.enter_context(patch.object(launch, 'read', side_effect=fake_read))
            stack.enter_context(patch.object(launch, 'acquire_idle_gpus', return_value=[lock]))
            stack.enter_context(patch.object(launch, 'verify_inputs'))
            stack.enter_context(patch.dict(launch.os.environ, {}, clear=True))
            stack.enter_context(patch.object(launch.subprocess, 'check_output', side_effect=[
                f"1, {manifest['current_gpu1_uuid']}, NVIDIA GeForce RTX 3070, {expected['driver']}\n",
                json.dumps(cpu)]))
            spawn = stack.enter_context(patch.object(launch.subprocess, 'Popen'))
            with self.assertRaises(FileExistsError):
                launch.main()
            self.assertEqual(status.read_text(), old)
            self.assertEqual([p.name for p in output.iterdir()], ['status.json'])
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
