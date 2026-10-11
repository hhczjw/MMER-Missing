# Published research-two source snapshot

This is the current local source snapshot published on 2026-10-11. Reference literature, reference codebases, datasets, experiment caches, logs and generated presentation artifacts are excluded. The modified Mamba runtime source is included because the model imports it directly; upstream license files are retained. Binary installation wheels are excluded.

Selected historical checkpoints are stored with Git LFS under `published_checkpoints/`. They are not newly trained models. Current source is not claimed to be a complete dated snapshot of the historical training environment. Dataset and pretrained-text paths in configurations must be adapted on a new machine. Diagnostic/queue scripts can reference intentionally excluded local experiment directories.

Run `git lfs install` and `git lfs pull` to obtain checkpoint contents.
