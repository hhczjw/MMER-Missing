"""CPU-only safety checks for queued native research-two replay."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_research2 as replay


class ReplaySafety(unittest.TestCase):
    def test_frozen_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base/'source').mkdir()
            path = base/'source/train.py'
            path.write_text('original')
            m = {'files': {'train.py': {'sha256':replay.sha(path)}}, 'helper_sha256':{}}
            with patch.object(replay,'BASE',base):
                replay.verify_sources(m)
                path.write_text('modified')
                with self.assertRaisesRegex(RuntimeError,'Frozen source changed'):
                    replay.verify_sources(m)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'already_exists';out.mkdir()
            sentinel=out/'sentinel';sentinel.write_text('retain')
            with patch.object(replay,'OUTPUT',out), patch.object(replay,'verify_sources'), \
                    patch.object(replay,'read',side_effect=[{}, {'state':'user_authorized_candidate','evidence_or_user_direction':'yes'}]), \
                    patch.object(replay,'acquire_idle_gpus') as resources:
                with self.assertRaises(FileExistsError):replay.main()
                resources.assert_not_called()
                self.assertEqual(sentinel.read_text(),'retain')

    def test_r1_failure_does_not_launch_r2(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'new_output'
            manifest={'authority':'human','modality':{'seeds':[43,45,48,42,44]}}
            decision={'state':'user_authorized_candidate','evidence_or_user_direction':'yes'}
            r1={'phase':'failed','pid':1,'worker_pid':2}
            with patch.object(replay,'OUTPUT',out), patch.object(replay,'verify_sources'), \
                    patch.object(replay,'read',side_effect=[manifest,decision,r1,r1]), \
                    patch.object(replay.shutil,'copy2'), patch.object(replay,'acquire_idle_gpus') as resources, \
                    patch.object(replay,'run_native') as native:
                with self.assertRaisesRegex(RuntimeError,'R1 failed'):replay.main()
                resources.assert_not_called();native.assert_not_called()
                state=json.loads((out/'status.json').read_text())
                self.assertEqual(state['phase'],'failed')

    def test_pending_review_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'not_created'
            with patch.object(replay,'OUTPUT',out), patch.object(replay,'verify_sources'), \
                    patch.object(replay,'read',side_effect=[{}, {'state':'pending'}]):
                with self.assertRaisesRegex(RuntimeError,'authorization missing'):replay.main()
                self.assertFalse(out.exists())


if __name__=='__main__':
    unittest.main()
