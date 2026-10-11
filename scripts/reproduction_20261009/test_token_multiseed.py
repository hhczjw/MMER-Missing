"""CPU-only queue safety; no CUDA diagnostics or training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import token_multiseed_queue as q

class TokenQueueSafety(unittest.TestCase):
    def test_seeds_and_reuse_are_fixed(self):
        self.assertEqual(q.SEEDS,[42,43,44,45,46,47])
        self.assertEqual(q.NEW_SEEDS,[42,43,44,46,47])
        flags=q.native_flags(42,Path('/tmp/token-test'))
        self.assertEqual(flags[flags.index('--batch-size')+1],'8')
        self.assertEqual(flags[flags.index('--eval-batch-size')+1],'16')
        self.assertEqual(flags[flags.index('--epochs')+1],'60')
        self.assertNotIn('--resume',flags)
    def test_running_r1_is_waited(self):
        with patch.object(q.legacy,'alive',return_value=True):
            self.assertFalse(q.dependency_ready({'phase':'train','pid':1}))
    def test_failed_r1_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError,'did not complete'):
            q.dependency_ready({'phase':'failed','pid':1})
    def test_lost_r1_is_not_started_over(self):
        with patch.object(q.legacy,'alive',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'lost'):
                q.dependency_ready({'phase':'train','pid':1})
    def test_gpu0_busy_does_not_block_gpu1(self):
        manifest={'gpu1_uuid':'GPU-one','expected_environment':{'driver':'580'}}
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'results').mkdir()
            with patch.object(q,'ROOT',Path(temp)),patch.object(q.fcntl,'flock'),patch.object(q.subprocess,'check_output',side_effect=['1, GPU-one, 580\n','1, GPU-zero\n']):
                lock=q.acquire_gpu1(manifest);self.assertTrue(lock.name.endswith('research2_gpu1.lock'));lock.close()
    def test_gpu1_busy_is_waited(self):
        manifest={'gpu1_uuid':'GPU-one','expected_environment':{'driver':'580'}}
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'results').mkdir()
            with patch.object(q,'ROOT',Path(temp)),patch.object(q.fcntl,'flock'),patch.object(q.subprocess,'check_output',side_effect=['1, GPU-one, 580\n','1, GPU-one\n']):
                with self.assertRaises(BlockingIOError):q.acquire_gpu1(manifest)
    def test_source_change_is_rejected(self):
        with patch.object(q.legacy,'verify_sources'),patch.object(q.legacy,'sha',return_value='wrong'):
            with self.assertRaisesRegex(RuntimeError,'helper changed'):
                q.verify_pin({'frozen_manifest':{},'helpers':{'some-helper':'expected'},'reused45_sha256':{}})

if __name__=='__main__':unittest.main()
