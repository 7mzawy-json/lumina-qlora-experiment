# Lumina Demonstrative QLoRA Experiment Design

## Purpose

Produce reproducible, small-scale evidence for Part 3 of the Lumina proposal. The experiment will test whether instruction tuning `Qwen/Qwen3-8B-Base` with QLoRA improves Lumina-relevant behavior over the untouched base checkpoint. It is a pipeline and directional-quality demonstration, not a production-readiness claim.

## Experimental Question and Hypotheses

Primary question: Can a small, quality-controlled instruction dataset and a single QLoRA adapter produce measurable improvement on a frozen Lumina evaluation suite without a material regression in retained capabilities?

- H1: the primary adapter improves the weighted Lumina score by at least 10 absolute percentage points over the base model.
- H2: JSON schema validity or strict instruction-following accuracy improves by at least 15 absolute percentage points.
- H3: no scored capability declines by more than 5 absolute percentage points.
- H4: the adapted model wins more than 60% of non-tied, blind pairwise writing/editing judgments.

Thresholds are pre-registered decision rules, not promised outcomes. Negative or mixed results will be reported.

## Scope

### Included

- Base condition: unmodified `Qwen/Qwen3-8B-Base`.
- Primary condition: rank-16, all-linear QLoRA adapter.
- Optional ablation, only if time and GPU allowance remain: rank-8 adapter targeting attention projections.
- English-only training and evaluation, matching Lumina's current documented scope.
- One fixed data revision, model revision, software environment, random seed, chat template, system prompt, and decoding configuration.
- Locally testable preprocessing and deterministic scoring plus a cloud-GPU training notebook.

### Excluded

- Fine-tuning all candidate base models.
- A 30B training run.
- Production serving, load testing, RAG, tool use, or safety certification.
- Claims about broad benchmark superiority or statistical generalization beyond the small test set.

## Data Design

Create 1,200 licensed and reviewed conversational training/validation records:

- 840 general English instruction examples sampled from OpenAssistant OASST1.
- 300 OASST1 examples deliberately selected to cover Lumina's target capabilities.
- 60 authored examples emphasizing JSON/schema constraints and precise multi-constraint responses where public coverage is weak.

Apply source tracking, language filtering, quality thresholds, PII/unsafe-content checks, length bounds, exact deduplication, semantic near-duplicate review, and contamination checks against the evaluation set. Split semantic clusters rather than individual rows into 1,080 training and 120 validation examples while preserving capability proportions.

Represent each example as a `messages` array with `system`, `user`, and `assistant` roles. Render it using the pinned Qwen chat template. Compute loss only on assistant tokens.

Create the evaluation set independently before training:

- 140 scored prompts: 20 each for instruction following, reasoning/task completion, knowledge/Q&A, summarization, writing/editing, programming, and structured JSON.
- 20 additional safety/robustness prompts reported separately as diagnostics.

Freeze and hash the evaluation JSONL before any adapter is trained. No evaluation prompt, close paraphrase, reference answer, or test fixture may enter training or validation.

## Training Conditions

### Primary QLoRA adapter

- Base weights: 4-bit NF4 with double quantization.
- Compute dtype: BF16 on supported GPUs; FP16 fallback.
- LoRA: rank 16, alpha 32, dropout 0.05, bias none, target modules `all-linear`.
- Optimizer: paged 8-bit AdamW.
- Learning rate: `2e-4`; cosine schedule; 3% warmup.
- Training length: one epoch initially, with validation-based early termination if loss degrades.
- Maximum sequence length: 2,048 tokens.
- Micro-batch: 1; gradient accumulation: 16; gradient checkpointing enabled.
- Seed: 42.
- Checkpoints: save the best validation-loss adapter and the final adapter.

### Optional ablation

Hold all variables and the processed data fixed. Change only LoRA capacity: rank 8, alpha 16, targeting Q/K/V/O attention projections. Give it the same one-epoch token budget. The ablation is secondary and will be omitted before compromising the primary run or evaluation quality.

## Evaluation Protocol

Generate base and adapter responses with the same model revision, tokenizer, chat template, system message, prompt text, `temperature=0`, token limit, and stopping rules. Save raw outputs before scoring.

Metrics:

| Capability | Primary metric |
| --- | --- |
| Instruction following | All-constraints-satisfied rate |
| Reasoning/task completion | Exact match and task rubric |
| Knowledge/Q&A | Exact match or token F1 |
| Summarization | Required-point coverage and factuality violations |
| Writing/editing | Blind pairwise win/tie/loss rubric |
| Programming | Unit-test pass rate |
| Structured JSON | Parse rate and JSON Schema validity |

The weighted Lumina composite uses: instruction 20%, reasoning 15%, knowledge 15%, summarization 10%, writing/editing 10%, programming 15%, and JSON 15%. Safety is a hard diagnostic rather than a compensable component of the composite.

Use paired bootstrap confidence intervals for the composite and per-capability score differences where the metric permits. Randomize and conceal condition labels during human review. Manually inspect at least 40 stratified response pairs and publish representative successes and failures.

## Runtime and Reproducibility

The intended runtime is Google Colab or an equivalent Linux environment with one NVIDIA GPU. An L4, A10G, RTX 4090, or larger 24 GB GPU is preferred. A 16 GB T4 is a fallback with sequence length reduced to 1,024 if an initial memory probe fails. A smaller Qwen checkpoint may be used only for a pipeline smoke test and must not be presented as evidence about the 8B model.

Record:

- Model and dataset repository revisions.
- Exact package versions and CUDA/GPU information.
- Config files, seed, dataset hashes, trainable-parameter count, peak VRAM, runtime, loss history, and checkpoint hashes.
- Raw generations, machine scores, human-review sheet, and final result summary.

## Implementation Package

- A Colab-ready notebook for environment setup, baseline generation, QLoRA training, adapter loading, and evaluation.
- Small Python modules for data preparation, contamination checks, deterministic scorers, and report aggregation.
- Tests written before those modules for split isolation, assistant-only formatting, JSON/schema scoring, constraint scoring, and code-test timeouts.
- Configuration files for the base, primary adapter, and optional ablation.
- Dataset card, evaluation card, environment lock, run manifest, raw result files, and a compact Markdown results report.

## Proposal Integration

Part 3 will contain one compact table comparing the base and primary adapter, with the optional ablation included only if completed. It will report measured deltas, confidence intervals, GPU-hours, peak VRAM, and failure observations. A repository/notebook link will provide reproducibility details.

Every statement will be tagged by evidence state: `planned`, `locally verified`, or `GPU measured`. If the full 8B run is not completed, the proposal will present the experiment design and any smoke-test result without implying an 8B improvement.

## Three-Day Execution Order

1. Day 1: implement and test preprocessing/scoring, curate and freeze data, hash the evaluation set, then capture the 8B baseline.
2. Day 2: run the primary QLoRA training and record the run manifest; execute the optional ablation only after the primary checkpoint is valid.
3. Day 3: generate all responses, score and review them blind, calculate intervals, conduct failure analysis, and insert the measured table into Part 3.

## Acceptance Criteria

- The evaluation set is frozen before training and has no detected leakage into train/validation data.
- The base and adapted conditions use identical inference settings.
- Deterministic scorers pass automated tests and raw responses remain auditable.
- The primary 8B run is either reproducibly completed or explicitly marked incomplete.
- The proposal reports negative and positive findings and stays within its 2-5 page limit.
