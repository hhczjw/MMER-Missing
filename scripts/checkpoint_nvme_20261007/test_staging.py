"""CPU-only checks: best selection, mmap load, IO isolation, archive failure retention."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from staging import CheckpointStaging


class StagingTest(unittest.TestCase):
    def test_redirect_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination, local = root / 'archive/job', root / 'nvme/job'
            helper = CheckpointStaging(destination, local)
            helper.install(torch)
            try:
                destination.mkdir(parents=True, exist_ok=False)
                checkpoint = destination / 'best.pth'
                temporary = checkpoint.with_suffix('.pth.tmp')
                tensor = torch.tensor([1., 2.])
                torch.save({'model': tensor, 'epoch': 1}, temporary)
                temporary.replace(checkpoint)
                tensor.add_(10)
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
                self.assertEqual(saved['model'].tolist(), [1., 2.])
                self.assertEqual(saved['epoch'], 1)
                torch.save({'model': tensor, 'epoch': 3}, temporary)
                temporary.replace(checkpoint)
                # Normal logging and unrelated checkpoint paths stay untouched.
                unrelated = root / 'other.pth'
                torch.save({'epoch': 9}, unrelated)
                self.assertTrue(unrelated.exists())
                self.assertFalse(checkpoint.exists())
                self.assertTrue((local / 'best.pth').exists())
            finally:
                helper.restore()
            helper.archive(root / 'receipt.json')
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual(saved['epoch'], 3)
            self.assertEqual(saved['model'].tolist(), [11., 12.])
            self.assertFalse((local / 'best.pth').exists())
            self.assertEqual(json.loads((root / 'receipt.json').read_text())['phase'], 'archived')

    def test_archive_failure_retains_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / 'nvme/job'
            local.mkdir(parents=True)
            torch.save({'epoch': 3}, local / 'best.pth')
            invalid_parent = root / 'not_a_directory'
            invalid_parent.write_text('keep')
            helper = CheckpointStaging(invalid_parent / 'job', local)
            with self.assertRaises(OSError):
                helper.archive(root / 'receipt.json')
            self.assertTrue((local / 'best.pth').exists())
            self.assertEqual(invalid_parent.read_text(), 'keep')

    def test_real_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'fake_worker.py'
            fake.write_text('''import sys\nfrom pathlib import Path\nimport torch\np=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=False)\nc=p/'best.pth';t=c.with_suffix('.pth.tmp')\ntorch.save({'epoch':7,'weights':torch.tensor([5.])},t);t.replace(c)\nassert torch.load(c,weights_only=False,mmap=True)['epoch']==7\nassert not c.exists()\n''')
            destination, local = root / 'archive/job', root / 'nvme/job'
            command = [sys.executable, str(Path(__file__).with_name('worker.py')),
                       '--destination', str(destination), '--local', str(local),
                       '--receipt', str(root / 'receipt.json'), str(fake), str(destination)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((destination / 'best.pth').exists())
            self.assertEqual(torch.load(local / 'best.pth', weights_only=False)['epoch'], 7)
            self.assertEqual(json.loads((root / 'receipt.json').read_text())['phase'], 'staged')
            CheckpointStaging(destination, local).archive(root / 'receipt.json')
            self.assertEqual(torch.load(destination / 'best.pth', weights_only=False)['epoch'], 7)


    def test_independent_archiver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'queue'
            output.mkdir()
            local, destination = root / 'nvme/job', root / 'archive/job'
            local.mkdir(parents=True)
            torch.save({'epoch': 9}, local / 'best.pth')
            helper = CheckpointStaging(destination, local)
            receipt = output / 'nvme_archive_job.json'
            helper.stage(receipt)
            (output / 'nvme_transition.json').write_text(json.dumps({'phase': 'complete'}))
            command = [sys.executable, str(Path(__file__).with_name('archiver.py')),
                       '--output', str(output), '--state-directory', str(root / 'state')]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(receipt.read_text())['phase'], 'archived')
            self.assertEqual(torch.load(destination / 'best.pth', weights_only=False)['epoch'], 9)
            self.assertFalse((local / 'best.pth').exists())


    def test_authorized_batch_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'configured_worker.py'
            fake.write_text("import sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\nimport torch\ndef configure(track):return SimpleNamespace(batch_size=4)\ndef main():\n a=configure('R1');assert a.batch_size==2;assert a.original_batch_size==4\n p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=False)\n c=p/'best.pth';t=c.with_suffix('.pth.tmp')\n torch.save({'batch':a.batch_size,'original':a.original_batch_size},t);t.replace(c)\nif __name__=='__main__':main()\n")
            destination, local = root / 'archive/job', root / 'nvme/job'
            command = [sys.executable, str(Path(__file__).with_name('worker.py')),
                       '--destination', str(destination), '--local', str(local),
                       '--receipt', str(root / 'receipt.json'), '--batch-size', '2', str(fake), str(destination)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = torch.load(local / 'best.pth', weights_only=False)
            self.assertEqual(value, {'batch': 2, 'original': 4})


if __name__ == '__main__':
    unittest.main()
