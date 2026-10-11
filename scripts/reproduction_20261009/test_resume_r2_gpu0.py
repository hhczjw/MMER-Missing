"""CPU-only safety checks for original-worker device adaptation."""
import unittest
from unittest.mock import patch
import r2_gpu0_worker as worker
import resume_r2_gpu0 as resume


class ResumeGPU0Safety(unittest.TestCase):
    def test_exactly_two_device_only_line_changes(self):
        original = worker.ORIGINAL.read_text()
        revised = worker.adapted_source(original)
        pairs = [(a,b) for a,b in zip(original.splitlines(), revised.splitlines()) if a != b]
        self.assertEqual(len(pairs), 2)
        self.assertIn("CUDA_VISIBLE_DEVICES", pairs[0][0])
        self.assertIn('physical_gpu=1', pairs[1][0])
        self.assertEqual(revised.replace("CUDA_VISIBLE_DEVICES')=='0'", "CUDA_VISIBLE_DEVICES')=='1'")
            .replace('pid=os.getpid(),physical_gpu=0,','pid=os.getpid(),physical_gpu=1,'), original)
        compile(revised, str(worker.ORIGINAL), 'exec')

    def test_source_shape_change_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            worker.adapted_source('unrecognized worker')

    def test_wrong_queue_phase_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Expected paused'):
            resume.validate_queue({'jobs':[]}, {'phase':'train'})

    def test_live_old_worker_rejected(self):
        with patch.object(resume, 'alive', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'still alive'):
                resume.validate_queue({'jobs':[]}, {'phase':'paused', 'pid':123})

    def test_actual_paused_queue_and_remaining_jobs(self):
        remaining = resume.validate_queue(resume.read(resume.OUTPUT / 'plan.json'),
            resume.read(resume.OUTPUT / 'status.json'))
        self.assertEqual(len(remaining), 19)
        self.assertEqual(remaining[0]['job_id'], 'R2-M-12_seed48')


if __name__ == '__main__':
    unittest.main()
