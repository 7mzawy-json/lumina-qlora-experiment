# Lumina QLoRA Demonstrative Experiment

This repository is a reproducible engineering demonstration for adapting an open-weight
language model with QLoRA and measuring it against the unchanged base model. It implements
the path from pinned model and dataset revisions through data preparation, assistant-only
instruction tuning, paired evaluation, concealed human review, and evidence export.

> **Evidence boundary:** the measured run is a small directional pilot, not a production
> model, safety certification, statistically powered benchmark, or claim that fine-tuning
> improved Qwen3-8B-Base.

[Open the Colab notebook](https://colab.research.google.com/github/7mzawy-json/lumina-qlora-experiment/blob/main/notebooks/lumina_qlora_demo.ipynb)
or follow the complete [runbook](docs/experiment/RUNBOOK.md).

## Measured pilot

The bounded pilot used `Qwen/Qwen3-8B-Base` at an immutable revision on a Tesla T4. It
trained a rank-16 QLoRA adapter for 16 optimizer steps on 256 examples, selected the
checkpoint using 40 validation examples, and compared base and adapted generations under
the same deterministic inference configuration.

| Item | Recorded value |
|---|---:|
| Training mixture | 30% demonstration-authored / 70% OASST1 |
| LoRA configuration | rank 16, alpha 32, dropout 0.05, all-linear |
| Quantization and runtime dtype | 4-bit NF4 with double quantization; FP16 fallback |
| Learning rate | `2e-4` |
| Selected validation loss | 1.4941 |
| Adapter training time | 448.2 seconds |
| Peak VRAM during adapted evaluation | 12.58 GB |
| Paired evaluation sample | 3 cases per capability; 24 per condition |

### Outcome

| Capability metric | Adapter minus base | 95% paired bootstrap CI |
|---|---:|---:|
| Instruction following | -33.33 pp | [-33.33, -33.33] |
| JSON schema validity | 0.00 pp | [0.00, 0.00] |
| Knowledge/Q&A | -33.33 pp | [-100.00, 0.00] |
| Programming | -33.33 pp | [-100.00, 0.00] |
| Reasoning | 0.00 pp | [0.00, 0.00] |
| Summarization | -33.73 pp | [-100.00, 10.53] |

The concealed writing comparison produced 0 adapter wins, 3 base wins, and 0 ties. No
serious new failure was recorded in the three selected safety diagnostics. The production
decision gate was deliberately **not evaluated** because the pilot contains only three
cases per capability.

This configuration is therefore rejected. The next controlled experiment should repeat
the same bounded pilot while changing only the learning rate from `2e-4` to `1e-4`. That
tests whether the observed regression was amplified by an aggressive learning rate before
spending compute on a larger rank or data search. The full frozen benchmark should run only
after a configuration clears this inexpensive screen.

The canonical result files are:

- [paired report](results/summary/pilot-r16/report.md)
- [machine-readable metrics](results/summary/pilot-r16/metrics.json)
- [evidence manifest](results/summary/pilot-r16/evidence-manifest.json)
- [training run manifest](results/summary/pilot-r16/run-manifest.json)
- [local preflight record](results/summary/preflight.json)

Raw generations, concealed-review keys, reviewer working files, processed datasets,
credentials, checkpoints, and model weights are intentionally excluded from version
control.

## Repository map

```text
configs/              Frozen smoke, pilot, full-run, and ablation configurations
data/authored/        Demonstration-authored training candidates and data card
data/eval/            Frozen evaluation cases, provenance, and corpus digest
notebooks/            Deterministically generated Colab execution notebook
results/summary/      Public-safe aggregate results and manifests
scripts/              Data, revision, notebook, scoring, review, and reporting CLIs
src/lumina_experiment Core contracts and experiment implementation
tests/                CPU-safe unit and integration tests
```

## Local verification

Use Python 3.11 or newer. GPU libraries are intentionally separate from the local
development environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install -e . --no-deps
python -m pytest -q
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
python scripts\scan_tracked_secrets.py
```

Preparing the frozen processed corpus or running the GPU notebook requires the additional
steps and evidence gates in the [runbook](docs/experiment/RUNBOOK.md). Do not substitute
unpinned model or dataset revisions and then compare the outputs with the tracked pilot.

## Methodological limitations

- The pilot sample is too small for capability-level statistical conclusions.
- The 150 demonstration-authored training candidates passed automated checks but remain
  pending complete independent human review.
- Public benchmark anchors may have appeared in the base model's pretraining data.
- Deterministic scorers measure narrow observable properties, not general response quality.
- The experiment is English-only and does not establish production safety or robustness.

## AI assistance declaration

AI tools assisted with drafting, implementation, and review. The experiment configuration,
GPU execution, concealed comparison, evidence validation, and final claims were retained
under human control. AI assistance is not treated as experimental evidence.
