# Lumina experiment runbook

## Local preflight and processed-data package

Use Python 3.11 and the development virtual environment. Public Hugging Face repository metadata is resolved without a token; do not download model weights during revision pinning.

```powershell
.\.venv\Scripts\python.exe scripts\pin_revisions.py --output configs\revisions.yaml
.\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\data.yaml --output data\processed
.\.venv\Scripts\python.exe scripts\scan_tracked_secrets.py --root .
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from lumina_experiment.gpu import verify_frozen_evaluation, verify_frozen_dataset_split; print(verify_frozen_evaluation()); print(verify_frozen_dataset_split(__import__('pathlib').Path('data/processed')))"
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe -m pytest --cov=lumina_experiment --cov-report=term-missing -v
.\.venv\Scripts\ruff.exe check src scripts tests
.\.venv\Scripts\ruff.exe format --check src scripts tests
git status --ignored --short
```

`pin_revisions.py` leaves an identical mapping untouched, fails on a differing existing pin, and permits replacement only with `--replace`. Evaluation-source revisions are frozen provenance anchors: the command verifies those exact commits rather than advancing them. The preparation command must report 3,000 accepted records (2,700 train, 300 validation), disjoint clusters, configured source counts, hashes, and rejections. A zero-hit contamination CSV has only its header and needs no adjudication rows. Any nonzero hit is `NEEDS_CONTEXT`: retain the exact CSV and count; do not fabricate a decision or rewrite the frozen evaluation corpus without controller direction.

The secret scanner emits canonical JSON and exits nonzero on a concrete credential-shaped Hugging Face/GitHub token or a non-placeholder `HF_TOKEN`/`WANDB_API_KEY` assignment. It scans the intended commit set—Git-cached plus non-ignored untracked files—via `git ls-files -z --cached --others --exclude-standard`; documentation and regex definitions are not token findings. Continue only when `findings` is `[]`.

GitHub archives exclude the ignored processed corpus. After verification, create the upload archive at the exact location `%TEMP%\lumina-processed.zip`; it is local-only and must not be committed:

```powershell
$processed = (Resolve-Path data\processed).Path
$archive = Join-Path $env:TEMP 'lumina-processed.zip'
Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $processed '*') -DestinationPath $archive -CompressionLevel Optimal -Force
Get-FileHash $archive -Algorithm SHA256
```

The archive must contain `accepted.jsonl`, `train.jsonl`, `validation.jsonl`, `rejections.jsonl`, `contamination-review.csv`, and `manifest.json` at its root. Keep the printed archive hash with the local preflight record.

## Colab setup and data import

The GPU requirements include the Python 3.13 compatibility overlay
`jedi==0.20.0` and `pyarrow==25.0.0`. The notebook installs the exact pinned
requirements with `--no-deps` to preserve Colab's CUDA PyTorch, then runs
`pip check` as a fail-closed gate. Any `pip check` failure stops the run; do not
use a manual, unrecorded package-install workaround.

1. Select a GPU runtime. An L4, A10G, RTX 4090, or another 24 GB GPU is preferred; record a T4 fallback as a deviation.
2. In Colab Secrets, add `GITHUB_REPOSITORY`, `GITHUB_COMMIT`, a read-only `GITHUB_TOKEN` authorized for the private repository, and `HF_TOKEN` (the secret name is exactly `HF_TOKEN`). Set `GITHUB_COMMIT` to the exact 40-character SHA of the commit just pushed for this run; it must not be a branch name or `HEAD`. Until this launch-boundary fix is committed, the required SHA is `22feb06215bfea551f5e8697f3e337e1062e075c`; after committing the fix, replace it with that just-pushed commit SHA. Enter all secret values directly in the Colab Secrets UI, never in chat or logs. Never print a secret or place one in a notebook output.
3. Upload/open `notebooks/lumina_qlora_demo.ipynb`. Run the dependency/repository cell, restart after dependency installation, then rerun from repository verification.
4. Before the notebook's smoke or 8B sections, use its processed-data import cell. For browser upload, upload the exact `%TEMP%\lumina-processed.zip` file when prompted. For Drive, set `USE_DRIVE_PROCESSED_ARCHIVE = True`, mount Drive, and place it at `/content/drive/MyDrive/lumina-processed.zip`.
5. The import cell validates archive members, copies them only to `PROJECT_ROOT/data/processed`, refuses an existing destination, then deletes the uploaded working copy. The next repository/freeze cell must verify `verify_frozen_dataset_split(Path("data/processed"))` before smoke work continues.
6. Run the small-checkpoint smoke path. It verifies assistant-only masking, four-bit loading, adapter injection, a short train/checkpoint/reload/generation path, deterministic scoring, and export. Smoke output is non-reportable.

If a Colab run stops, retain its checkpoint directory and manifest, restart, re-verify revisions and hashes, and resume only from the matching checkpoint. Never mix outputs from different revisions or processed-split hashes.

## Download, score, review, and summarize 8B evidence

The notebook downloads `/content/lumina-results.zip`. It already contains `results/raw/...`; extract to a neutral directory, not to `results/raw`, to avoid `results/raw/results/raw` nesting:

```powershell
$export = Join-Path $env:TEMP 'lumina-results-export'
Remove-Item -LiteralPath $export -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath (Join-Path $env:USERPROFILE 'Downloads\lumina-results.zip') -DestinationPath $export -Force
$raw = Join-Path $export 'results\raw'
$summary = Join-Path (Get-Location) 'results\summary\primary-r16'
New-Item -ItemType Directory -Path $summary -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $raw 'base\manifest.json') -Destination (Join-Path $summary 'base-manifest.json') -Force
Copy-Item -LiteralPath (Join-Path $raw 'primary-r16\manifest.json') -Destination (Join-Path $summary 'primary-r16-manifest.json') -Force
.\.venv\Scripts\python.exe scripts\score_results.py --base (Join-Path $raw 'base\generations.jsonl') --adapter (Join-Path $raw 'primary-r16\generations.jsonl') --cases data\eval\cases --output (Join-Path $raw 'scored\paired-scores.jsonl')
.\.venv\Scripts\python.exe scripts\build_review_sheet.py --scores (Join-Path $raw 'scored\paired-scores.jsonl') --output (Join-Path $raw 'review\blind-review.csv') --private-key (Join-Path $raw 'review\review-key.json') --seed 42 --sample-size 42
```

Complete the concealed review CSV and create the required ignored `results/raw/review/diagnostics.jsonl`: exactly one human-reviewed boolean `serious_new_failure` record for each of the 20 frozen safety case IDs, adding `failure_mode` only when true. Then run and re-verify the measured summary:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_results.py --scores (Join-Path $raw 'scored\paired-scores.jsonl') --review (Join-Path $raw 'review\blind-review.csv') --private-key (Join-Path $raw 'review\review-key.json') --diagnostics (Join-Path $raw 'review\diagnostics.jsonl') --output $summary --seed 42 --resamples 10000
.\.venv\Scripts\python.exe scripts\summarize_results.py --verify-only --scores (Join-Path $raw 'scored\paired-scores.jsonl') --review (Join-Path $raw 'review\blind-review.csv') --private-key (Join-Path $raw 'review\review-key.json') --diagnostics (Join-Path $raw 'review\diagnostics.jsonl') --output $summary --seed 42 --resamples 10000
```

Local verification proves preprocessing, pins, freezes, scanner state, and notebook determinism only. Smoke verification proves the small-model pipeline only. Enable `RUN_8B = True` only after the smoke artifacts pass local verification. Only validated base/adapter manifests, paired raw generations, deterministic scores, and review evidence are `8b_gpu_measured`; never present smoke output as 8B evidence.
