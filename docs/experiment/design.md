# Lumina Demonstrative QLoRA Experiment Design

## Purpose

Produce reproducible, small-scale evidence for Part 3 of the Lumina proposal. The experiment tests whether instruction tuning an open-weight 8B checkpoint with QLoRA improves Lumina-relevant behavior over the untouched checkpoint. It is a directional pilot and pipeline demonstration, not evidence of production readiness or broad benchmark superiority.

## Model Decision

The default experimental checkpoint is `Qwen/Qwen3-8B-Base`, interpreting the assignment's term "base model" as a raw pretrained checkpoint. If Parallax Horizon confirms that an instruction-tuned checkpoint is eligible, the experiment will instead use the official Qwen3 8B post-trained checkpoint and reduce the proportion of general instruction data because basic alignment is already present.

The selected model identifier is configuration, not hard-coded application logic. Only one 8B checkpoint will be trained in the pilot; the proposal's other candidates remain a documented model-selection comparison.

## Experimental Question and Pre-Registered Decisions

Primary question: Can a small, quality-controlled instruction dataset and a single QLoRA adapter measurably improve strict instruction following and structured JSON output without a material regression in other Lumina capabilities?

Primary outcomes:

- Strict all-constraints-satisfied rate for instruction-following prompts.
- JSON Schema validity rate. JSON parse rate is reported as a supporting diagnostic.

Secondary outcomes are reasoning/task completion, knowledge/Q&A, summarization, writing/editing, and programming. Results remain separate by capability; the experiment will not use an arbitrary weighted composite as its headline.

The pilot is considered directionally successful only if:

- At least one primary outcome improves by 15 or more absolute percentage points and the other has a positive absolute change.
- No secondary capability declines by more than 5 absolute percentage points.
- The adapted model wins more than 60% of non-tied, blind writing/editing comparisons.
- Safety diagnostics reveal no serious new failure mode.

These thresholds are decision rules, not promised outcomes. Negative and mixed findings will be reported.

## Scope

### Included

- Base condition: the unmodified selected 8B checkpoint.
- Primary condition: rank-16, all-linear QLoRA adapter.
- Optional ablation, only after the primary run and evaluation succeed: rank-8, all-linear QLoRA.
- English-only training and evaluation, matching Lumina's currently documented scope.
- A disposable small-checkpoint smoke test before the 8B GPU run.
- Fixed model and dataset revisions, random seed, chat template, system prompt, inference configuration, and evaluation set.
- Locally tested preprocessing and scoring plus cloud-GPU baseline generation and training.

### Excluded

- Fine-tuning every candidate model or any 30B model.
- Production serving, load testing, RAG, tool use, or safety certification.
- Treating smoke-test output as evidence about the selected 8B model.
- Claims of statistical generalization beyond the small, purpose-built test set.

## Data Design

Create 3,000 licensed and reviewed conversational records:

- 2,100 general English instruction examples sampled from OpenAssistant OASST1.
- 750 OASST1 examples deliberately selected for Lumina capabilities.
- 150 authored and fully reviewed examples emphasizing JSON/schema constraints and precise multi-constraint responses where public coverage is weak.

If an instruction-tuned checkpoint is approved, change the mixture before freezing the processed dataset to 30% general and 70% capability-targeted examples while keeping the 3,000-record budget.

Apply source and license tracking, English filtering, quality thresholds, PII and unsafe-content checks, length bounds, exact deduplication, semantic near-duplicate review, and evaluation-contamination checks. Split semantic clusters rather than individual rows into 2,700 training and 300 validation examples while preserving source and capability proportions.

Represent every example as a `messages` array with `system`, `user`, and `assistant` roles. Render it with the tokenizer's pinned Qwen chat template. Compute loss only on assistant tokens.

## Frozen Evaluation Set

Create the evaluation set independently before training:

- 210 scored prompts: 30 each for instruction following, reasoning/task completion, knowledge/Q&A, summarization, writing/editing, programming, and structured JSON.
- 20 additional safety/robustness prompts reported separately as diagnostics.

The set will combine fresh authored tasks with a small number of clearly attributed public benchmark anchors. Freeze and hash the evaluation JSONL and scoring fixtures before adapter training. No evaluation prompt, close paraphrase, reference answer, rubric answer, schema, or hidden test fixture may enter training or validation.

## Smoke Test

Before allocating an 8B GPU run, execute the entire path with the smallest compatible Qwen3 base checkpoint, approximately 50 training records, and 10 non-reportable test prompts. The smoke test must verify:

- Conversational formatting and assistant-only loss masks.
- Four-bit loading and adapter injection.
- One short training pass, checkpoint save, reload, and generation.
- Deterministic scoring and result export.

Smoke-test outputs are discarded from the proposal's evidence table and labeled as pipeline verification only.

## Training Conditions

### Primary QLoRA adapter

- Base weights: 4-bit NF4 with double quantization.
- Compute dtype: BF16 on supported GPUs; FP16 fallback.
- LoRA: rank 16, alpha 32, dropout 0.05, bias none, target modules `all-linear`.
- Optimizer: paged 8-bit AdamW.
- Learning rate: `2e-4`; cosine schedule; 3% warmup.
- Maximum training length: two epochs, with validation evaluation during training.
- Maximum sequence length: 2,048 tokens.
- Micro-batch: 1; gradient accumulation: 16; gradient checkpointing enabled.
- Seed: 42.
- Checkpoints: save intermediate, best-validation, and final adapters. Choose between the one- and two-epoch checkpoints using validation evidence only.

### Optional rank ablation

Hold the model revision, processed data, token budget, seed, optimizer, learning rate, target modules, and evaluation protocol fixed. Change only LoRA capacity to rank 8 and alpha 16. Omit this run before compromising the primary evaluation or the deadline.

The broader proposal may describe attention-only versus all-linear and other rank experiments as future work, but the demonstrative ablation will not confound rank with target-module selection.

## Evaluation Protocol

Generate base and adapter responses using the same model revision, tokenizer, chat template, system message, prompt text, `temperature=0`, maximum output length, and stopping rules. Save raw outputs before scoring and assign concealed, randomized condition labels for human review.

| Capability | Pilot measurement |
| --- | --- |
| Instruction following | Strict all-constraints-satisfied rate |
| Reasoning/task completion | Exact match and task-specific rubric |
| Knowledge/Q&A | Exact match or token F1 |
| Summarization | Required-point coverage and factuality violations |
| Writing/editing | Blind pairwise win, tie, or loss rubric |
| Programming | Sandboxed unit-test pass rate, or syntax plus blind review if no safe sandbox exists |
| Structured JSON | Parse rate and JSON Schema validity |

Report paired absolute differences and paired bootstrap confidence intervals where appropriate. With only 30 prompts per capability, category findings are directional and their uncertainty must remain visible. Manually inspect at least 42 stratified response pairs and publish representative improvements, regressions, and unchanged failures.

Generated code must not execute in the training notebook or a credentialed environment. Unit-test pass rate may be reported only if execution occurs in a disposable, network-disabled sandbox with time, memory, process, and filesystem limits. Otherwise, programming results are limited to syntax validity and blinded rubric review.

## Runtime and Reproducibility

The intended runtime is Google Colab or an equivalent Linux environment with one NVIDIA GPU. An L4, A10G, RTX 4090, or larger 24 GB GPU is preferred. A 16 GB T4 is a fallback only after a memory probe; reduce sequence length to 1,024 if necessary and record that deviation.

Record:

- Model and dataset repository revisions.
- Exact package versions and CUDA/GPU information.
- Configuration, seed, dataset and evaluation hashes, trainable-parameter count, peak VRAM, runtime, loss history, and checkpoint hashes.
- Raw generations, deterministic scores, concealed human-review sheet, and result summary.
- Any deviation from the pre-registered design and the reason it occurred.

Every result statement is tagged as `planned`, `locally verified`, `smoke-test verified`, or `8B GPU measured`.

## Implementation Package

- A Colab-ready notebook for environment setup, smoke testing, baseline generation, QLoRA training, adapter loading, and evaluation.
- Focused Python modules for data preparation, split isolation, contamination checks, deterministic scoring, blinded review preparation, and report aggregation.
- Tests written before the modules for message validation, cluster split isolation, contamination detection, JSON/schema scoring, constraint scoring, result pairing, and unsafe code-execution prevention.
- Configuration files for the smoke test, base inference, primary adapter, and optional rank ablation.
- Dataset card, evaluation card, environment lock, run manifest, raw result files, and compact Markdown results report.

## Proposal Integration

Part 3 will contain one compact table comparing the untouched 8B checkpoint with the primary adapter. The optional ablation appears only if completed and clearly secondary. The table reports primary and secondary metrics, paired deltas, uncertainty, GPU-hours, peak VRAM, and material failure observations; a private or shared repository link provides the audit trail.

If the complete 8B run is not finished, the proposal will present the experiment design and verified pipeline state without implying measured 8B improvement.

## Three-Day Execution Order

1. Day 1: implement and test preprocessing/scoring; curate data; author, review, freeze, and hash the evaluation set; complete the small-checkpoint smoke test.
2. Day 2: capture the untouched 8B baseline, train the primary adapter, validate its checkpoints, and record the run manifest. Run the optional ablation only if the primary evidence is complete.
3. Day 3: generate all adapted responses, score them, conduct blinded review and failure analysis, calculate intervals, and insert the measured table into Part 3.

## Acceptance Criteria

- The evaluation set and scoring fixtures are frozen before 8B training and have no detected leakage into train/validation data.
- Base and adapter conditions use identical inference settings.
- Deterministic scorers pass automated tests and raw responses remain auditable.
- The smoke test proves the complete pipeline without being presented as 8B evidence.
- Programming code is either evaluated in a constrained sandbox or not executed.
- The primary 8B run is reproducibly completed or explicitly marked incomplete.
- The proposal reports improvements, regressions, uncertainty, and design deviations while staying within its 2-5 page limit.
