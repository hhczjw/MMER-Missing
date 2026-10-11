"""CPU-only completion-gate checks; no real queues or GPU state are mutated."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from queue_guard import WORKS, check_queues


class QueueGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name, key in WORKS:
            folder = self.root / 'results' / name
            (folder / 'train/job').mkdir(parents=True)
            self.write(folder / 'plan.json', {'jobs':[{'job_id':'job'}]})
            self.write(folder / 'status.json', {'phase':'complete', key:['job']})
            self.write(folder / 'train/job/status.json', {'phase':'complete'})
            self.write(folder / 'train/job/test.json', {'MAE':1.0})

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def tearDown(self):
        self.temp.cleanup()

    def test_all_complete(self):
        self.assertEqual(len(check_queues(self.root)), 2)

    def test_single_worker_not_full_queue(self):
        name, key = WORKS[0]
        self.write(self.root / 'results' / name / 'status.json',
                   {'phase':'train', key:['job']})
        with self.assertRaisesRegex(RuntimeError, 'not complete'):
            check_queues(self.root)

    def test_missing_plan_job(self):
        name, key = WORKS[1]
        self.write(self.root / 'results' / name / 'status.json',
                   {'phase':'complete', key:[]})
        with self.assertRaisesRegex(RuntimeError, 'not complete'):
            check_queues(self.root)

    def test_live_controller(self):
        name, key = WORKS[0]
        self.write(self.root / 'results' / name / 'status.json',
                   {'phase':'complete', key:['job'], 'pid':os.getpid()})
        with self.assertRaisesRegex(RuntimeError, 'still alive'):
            check_queues(self.root)

    def test_failed_result_marker(self):
        name, _ = WORKS[1]
        self.write(self.root / 'results' / name / 'train/job/status.json',
                   {'phase':'failed'})
        with self.assertRaisesRegex(RuntimeError, 'Missing successful'):
            check_queues(self.root)

    def paused_fixture(self, authorize=True):
        policy = self.root / 'scripts/reproduction_20261008/scheduling_policy.json'
        policy.parent.mkdir(parents=True)
        if authorize:
            self.write(policy, dict(execution_gate='complete_or_explicitly_paused_after_current_success', authority='test human instruction'))
        for name,key in WORKS:
            folder=self.root/'results'/name
            self.write(folder/'plan.json',{'jobs':[{'job_id':'job'},{'job_id':'not_started'}]})
            self.write(folder/'status.json',dict(phase='paused',current_job='job',**{key:['job']}))
            self.write(folder/'pause_request.json',dict(phase='paused',current_job='job',authority='test human instruction',
                controller_exited=True,worker_exited=True,
                plan_sha256=hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),
                current_worker_result_sha256=hashlib.sha256((folder/'train/job/test.json').read_bytes()).hexdigest()))

    def test_explicit_successful_pause_allows_remaining_jobs(self):
        self.paused_fixture()
        self.assertEqual(len(check_queues(self.root)),2)

    def test_unrequested_pause_is_rejected(self):
        self.paused_fixture(authorize=False)
        with self.assertRaisesRegex(RuntimeError,'without human'):
            check_queues(self.root)

    def test_paused_result_change_is_rejected(self):
        self.paused_fixture()
        self.write(self.root/'results'/WORKS[0][0]/'train/job/test.json',{'MAE':2.0})
        with self.assertRaisesRegex(RuntimeError,'result changed'):
            check_queues(self.root)

    def test_failed_boundary_is_rejected(self):
        self.paused_fixture()
        self.write(self.root/'results'/WORKS[0][0]/'train/job/status.json',{'phase':'failed'})
        with self.assertRaisesRegex(RuntimeError,'Missing successful'):
            check_queues(self.root)


if __name__ == '__main__':
    unittest.main()
