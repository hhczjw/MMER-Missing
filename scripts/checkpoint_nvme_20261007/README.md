# NVMe checkpoint staging and asynchronous archival

The frozen R1/R2 training workers and their manifests remain unchanged. A separate
worker launcher redirects only its checkpoint directory to the project's local
NVMe filesystem (/dev/nvme0n1p2). Validation selection, training, RNG, and testing
continue through the original worker. Checkpoint paths recorded by the original
worker retain their final archive destination.

During training, each improved best checkpoint is saved and loaded on NVMe.
After successful testing, the launcher fsyncs the local checkpoint and records a
`staged` receipt. The GPU controller immediately starts the next job. One separate
archiver consumes receipts from both queues, writes a temporary file on the
original destination, fsyncs and atomically replaces the final file. Only after
successful archival are the local checkpoint files removed. A failed archival
keeps the local files and retries in the background.

The current workers (R1-04_seed2024 and R2-T-02_seed45) continue unchanged.
Transition controllers hold the original queue locks and wait for those workers
to exit successfully, then run only the remaining jobs. Completed jobs are retained.

Runtime state:

- `results/checkpoint_nvme_staging_20261007/`: NVMe staging, process metadata, and logs.
- Each queue's `nvme_transition.json`: controller state.
- Each queue's `nvme_archive_<job>.json`: staged / archiving / archived / archive_failed.
- `results/checkpoint_nvme_staging_20261007/archiver_status.json`: background archiver state.

Training completion and archive completion are distinct. Receipts report the
local checkpoint location while archival is pending. If NVMe free space falls
below 12 GiB, the controller waits before starting another experiment, allowing
background archival to free space. The archiver exits after both queues and all
successful-job archives complete.

CPU-only validation: `test_staging.py` checks checkpoint selection/load parity,
normal-path isolation, archive-failure retention, and the independent archiver.

## GPU0 OOM recovery (2026-10-07)

`resume_gpu0.py` restarts the failed R1-05_seed2024 experiment from the original
seed with batch_size=2 instead of 4, then runs the remaining GPU0 jobs with the
same explicit override. Original frozen worker sources and manifests are retained.
The launcher records original_batch_size and batch_override_reason in config.json.
This is a training-protocol change; completed batch=4 results must not be treated
as a uniform batch=2 cohort. No optimizer/RNG resume state was available.

The original failed attempt, its log, and its best epoch-16 NVMe checkpoint are
preserved under the queue's failed_attempts/ directory. batch_recovery.json records
that location and the retained completed jobs. NVMe staging and independent
asynchronous archival remain enabled; GPU1 is unaffected.

## Emergency-gradient memory repair and input cache

The independent `r1_worker_recovery.py` releases the previous forward graph and
the chained clipping exception (which retained the old gradient tensor list)
before the emergency backward. Adam moment tensors temporarily move to CPU and
return to their original device without arithmetic changes. Native forward and
reference-backward equivalence checks remain mandatory; no batch is skipped.

The GPU0-exclusive second-batch forced-failure validation passed with Adam state
already allocated. See results/gpu0_memory_recovery_smoke_20261007_v2/smoke.json.

After each successful epoch, latest_resume.pth on NVMe stores model, optimizer,
scheduler, early-stopping counters, history, and Python/NumPy/CPU/CUDA RNG states.
`worker.py --resume` permits existing result/checkpoint directories only when this
full state exists; the recovery worker validates the batch, seed, and source hash.
The temporary full state is removed after audited final testing, before archival.

Two HDD input files have verified, byte-identical NVMe copies. data_cache.py redirects
read-only opens and file-presence checks, including frozen data hashing, to those
copies. The source paths and input files remain unchanged. GPU1's cache-aware
controller waits for the existing worker before starting the next worker.

## Offline model loading after controller handover

All wrapped workers force HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before
importing the model libraries and remove HTTP/SOCKS proxy variables only from
their own process environment. This prevents local RoBERTa initialization from
entering a proxy client dependency path. The system proxy settings are unchanged.
`resume_gpu1.py` preserves the failed initialization attempt and retains all
completed experiments before restarting the same original job and seed.
