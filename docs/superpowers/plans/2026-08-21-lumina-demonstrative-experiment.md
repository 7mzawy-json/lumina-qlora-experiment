# Lumina Demonstrative Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, small-scale QLoRA experiment that supplies honest base-versus-adapter evidence for Part 3 of the Lumina proposal.

**Architecture:** A local, test-first Python package prepares licensed instruction data, freezes an independent evaluation set, scores paired generations, and produces an auditable result summary. A generated Colab notebook imports the same package to smoke-test the GPU path, capture the untouched 8B baseline, train one primary adapter, and export raw evidence; no GPU-only result is treated as measured until its run manifest and artifacts validate locally.

**Tech Stack:** Python 3.11.15, pytest, Ruff, Hugging Face Datasets 5.0.1, Transformers 5.14.1, TRL 1.9.2, PEFT 0.20.0, Accelerate 1.14.0, bitsandbytes 0.50.0, PyTorch supplied by the Colab CUDA runtime, PyYAML, jsonschema, scikit-learn, NumPy, pandas, matplotlib, and nbformat.

**Spec:** `docs/experiment/design.md`

## Global Constraints

- Default 8B checkpoint: `Qwen/Qwen3-8B-Base`; switch to the official Qwen3 8B post-trained checkpoint only after written confirmation from Parallax Horizon.
- Training corpus: exactly 3,000 accepted records, split by semantic cluster into 2,700 training and 300 validation records.
- Raw-base mixture: 2,100 general OASST1, 750 capability-targeted OASST1, and 150 authored records.
- Evaluation: 210 scored prompts, exactly 30 per capability, plus 20 separately reported safety/robustness diagnostics.
- Primary adapter: QLoRA rank 16, alpha 32, dropout 0.05, `all-linear`, NF4, double quantization, one-to-two-epoch selection using validation evidence only.
- Optional ablation: rank 8, alpha 16, `all-linear`; all other experimental variables remain fixed.
- The evaluation corpus and scoring fixtures must be hashed and frozen before any 8B adapter training.
- Base and adapter inference must use identical prompts, template, system message, token limit, stopping conditions, and deterministic decoding.
- Generated code must not execute inside Colab or any credentialed environment. Unit-test pass rate requires a disposable network-disabled sandbox; otherwise report syntax and blind-review results only.
- Never commit `.env`, Hugging Face tokens, model weights, raw generations, checkpoints, or private reviewer identities.
- Evidence labels are exactly `planned`, `locally_verified`, `smoke_test_verified`, and `8b_gpu_measured`.
- The proposal must not claim measured 8B improvement unless the validated run manifest, raw paired generations, and score summary all exist.

## File and Interface Map

- `src/lumina_experiment/contracts.py`: immutable message, record, split, evaluation, score, generation, GPU-probe, and manifest dataclasses plus JSON conversion.
- `src/lumina_experiment/config.py`: YAML loading and strict experiment-configuration validation.
- `src/lumina_experiment/data_pipeline.py`: OASST1 extraction, filtering, capability selection, authored-data merge, and deterministic sampling.
- `src/lumina_experiment/isolation.py`: exact deduplication, semantic clusters, group split, contamination detection, and canonical hashing.
- `src/lumina_experiment/scoring.py`: deterministic constraint, JSON Schema, exact-match, token-F1, summary-coverage, and Python-syntax scorers.
- `src/lumina_experiment/review.py`: concealed pair generation and completed-review validation.
- `src/lumina_experiment/statistics.py`: paired bootstrap differences and directional success-gate evaluation.
- `src/lumina_experiment/reporting.py`: condition summaries, evidence-state validation, tables, plots, and Markdown report rendering.
- `src/lumina_experiment/gpu.py`: lazily imported GPU environment probe, model loading, QLoRA configuration, SFT, checkpoint selection, and deterministic generation.
- `scripts/prepare_data.py`, `freeze_eval.py`, `score_results.py`, `build_review_sheet.py`, `summarize_results.py`: thin CLIs over package functions.
- `scripts/build_notebook.py`: deterministic generator for `notebooks/lumina_qlora_demo.ipynb`.
- `configs/data.yaml`, `eval.yaml`, `smoke.yaml`, `primary-r16.yaml`, `ablation-r8.yaml`: frozen inputs to each pipeline stage.
- `data/authored/training.jsonl`: 150 reviewed Lumina-targeted training records.
- `data/eval/cases/*.jsonl`: seven 30-case capability sets plus the 20-case safety set.
- `data/eval/sources.yaml`, `data/eval/FROZEN.sha256`, `data/eval/CARD.md`: attribution, freeze hash, and evaluation limitations.
- `results/summary/`: tracked manifests, aggregate CSV/JSON, plots, review rubric without reviewer identity, and proposal-ready Markdown; raw outputs stay ignored in `results/raw/`.

---

### Task 1: Lock contracts, configurations, and dependency boundaries

**Files:**
- Create: `src/lumina_experiment/contracts.py`
- Create: `src/lumina_experiment/config.py`
- Create: `configs/data.yaml`
- Create: `configs/eval.yaml`
- Create: `configs/smoke.yaml`
- Create: `configs/primary-r16.yaml`
- Create: `configs/ablation-r8.yaml`
- Create: `configs/revisions.yaml`
- Create: `requirements-gpu.txt`
- Modify: `requirements-dev.txt`
- Modify: `.env.example`
- Test: `tests/test_contracts.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Message`, `InstructionRecord`, `DatasetSplit`, `EvalCase`, `Score`, `Generation`, `GpuProbe`, `RunManifest`, and `ExperimentConfig` dataclasses.
- Produces: `load_experiment_config(path: Path) -> ExperimentConfig` and `canonical_json(value: object) -> str`.
- Produces: `ExperimentConfig.without(*field_names: str) -> dict[str, object]` for ablation comparisons.
- Consumes: no application interfaces.

- [ ] **Step 1: Write failing contract tests**

```python
def test_instruction_record_rejects_non_alternating_roles():
    payload = {
        "id": "oasst:1",
        "source": "OpenAssistant/oasst1",
        "license": "Apache-2.0",
        "capability": "general",
        "cluster_id": "tree-1",
        "messages": [
            {"role": "user", "content": "Explain gravity."},
            {"role": "user", "content": "Briefly."},
        ],
    }
    with pytest.raises(ValueError, match="alternate"):
        InstructionRecord.from_dict(payload)


def test_generation_identity_is_condition_and_case():
    generation = Generation(
        case_id="json-001",
        condition="base",
        output="{}",
        evidence_state="8b_gpu_measured",
    )
    assert generation.identity == ("base", "json-001")
```

- [ ] **Step 2: Run the contract tests and observe the missing-module failure**

Run: `pytest tests/test_contracts.py -v`

Expected: FAIL because `lumina_experiment.contracts` does not exist.

- [ ] **Step 3: Implement immutable contracts and canonical JSON conversion**

Use frozen dataclasses, validate non-empty IDs/content, permit only `system`, `user`, and `assistant` roles, require the final training message to be `assistant`, constrain capabilities to the seven named capabilities plus `general` and `safety`, and constrain evidence state to the four global values. Serialize with sorted keys, UTF-8, and compact separators.

```python
@dataclass(frozen=True)
class Score:
    case_id: str
    condition: str
    metric: str
    value: float
    passed: bool | None
    details: Mapping[str, object] = field(default_factory=dict)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
```

- [ ] **Step 4: Write failing configuration tests**

```python
def test_primary_config_matches_preregistered_adapter():
    cfg = load_experiment_config(Path("configs/primary-r16.yaml"))
    assert cfg.model_id == "Qwen/Qwen3-8B-Base"
    assert (cfg.lora_rank, cfg.lora_alpha) == (16, 32)
    assert cfg.target_modules == "all-linear"
    assert cfg.max_epochs == 2


def test_ablation_changes_only_rank_and_alpha():
    primary = load_experiment_config(Path("configs/primary-r16.yaml"))
    ablation = load_experiment_config(Path("configs/ablation-r8.yaml"))
    assert primary.without("name", "lora_rank", "lora_alpha") == ablation.without(
        "name", "lora_rank", "lora_alpha"
    )
```

- [ ] **Step 5: Run configuration tests and observe the missing-loader failure**

Run: `pytest tests/test_config.py -v`

Expected: FAIL because `load_experiment_config` is undefined.

- [ ] **Step 6: Implement strict YAML parsing and add exact configurations**

`primary-r16.yaml` must specify model ID, `revision_file: configs/revisions.yaml`, seed 42, NF4, double quantization, BF16 preference, rank 16, alpha 32, dropout 0.05, `all-linear`, learning rate `2e-4`, cosine schedule, warmup ratio 0.03, maximum two epochs, sequence length 2,048, micro-batch 1, accumulation 16, and temperature 0. `ablation-r8.yaml` changes only name, rank, and alpha. `smoke.yaml` uses the smallest compatible Qwen3 base checkpoint, 50 records, 10 cases, 20 optimizer steps, and an isolated output directory. `configs/revisions.yaml` is initially an empty mapping and is populated once by the preflight revision-pinning command before data preparation.

- [ ] **Step 7: Add GPU dependency pins without replacing Colab's CUDA PyTorch**

```text
accelerate==1.14.0
bitsandbytes==0.50.0
datasets==5.0.1
peft==0.20.0
transformers==5.14.1
trl==1.9.2
```

Add `nbformat` to `requirements-dev.txt`, install it locally, refresh `requirements-dev.lock`, and verify `python -m pip check` reports no broken requirements.

Add `GITHUB_TOKEN=` to `.env.example` with a comment restricting it to a fine-grained, read-only token for this private repository. Real GitHub and Hugging Face tokens remain only in `.env` or Colab Secrets.

- [ ] **Step 8: Run task verification and commit**

Run: `pytest tests/test_contracts.py tests/test_config.py -v`

Run: `ruff check src tests`

Expected: all tests pass and Ruff reports no errors.

Run: `git add src/lumina_experiment/contracts.py src/lumina_experiment/config.py configs requirements-gpu.txt requirements-dev.txt requirements-dev.lock .env.example tests/test_contracts.py tests/test_config.py`

Commit: `git commit -m "feat: define experiment contracts and configurations"`.

### Task 2: Build deterministic OASST1 preparation and authored-data validation

**Files:**
- Create: `src/lumina_experiment/data_pipeline.py`
- Create: `scripts/prepare_data.py`
- Create: `data/authored/training.jsonl`
- Create: `data/authored/CARD.md`
- Test: `tests/fixtures/oasst_rows.json`
- Test: `tests/test_data_pipeline.py`

**Interfaces:**
- Consumes: `InstructionRecord`, `ExperimentConfig`.
- Produces: `extract_oasst_pairs(rows: Iterable[Mapping[str, object]]) -> list[InstructionRecord]`.
- Produces: `filter_records(records, max_characters) -> tuple[list[InstructionRecord], list[Rejection]]`.
- Produces: `prepare_dataset(oasst_rows, authored_rows, config) -> list[InstructionRecord]`.
- Produces: CLI outputs `data/processed/accepted.jsonl` and `data/processed/rejections.jsonl`.

- [ ] **Step 1: Write failing extraction and filtering tests**

```python
def test_extracts_english_root_to_ranked_assistant_pair(oasst_rows):
    records = extract_oasst_pairs(oasst_rows)
    assert records[0].messages == (
        Message(role="user", content="Give two uses of solar energy."),
        Message(role="assistant", content="Electricity generation and water heating."),
    )
    assert records[0].cluster_id == "tree-001"


def test_rejection_log_records_reason():
    accepted, rejected = filter_records(
        [valid_record(), valid_record(id="bad", messages=overlong_messages())],
        max_characters=12_000,
    )
    assert [record.id for record in accepted] == ["valid"]
    assert rejected[0].reason == "over_length"
```

- [ ] **Step 2: Run the data tests and observe failure**

Run: `pytest tests/test_data_pipeline.py -v`

Expected: FAIL because the extraction functions do not exist.

- [ ] **Step 3: Implement OASST1 tree reconstruction and deterministic selection**

Reconstruct message ancestry by `parent_id`, retain English roots and assistant replies, choose the highest-ranked valid assistant child with deterministic ID tie-breaking, preserve `message_tree_id` as `cluster_id`, and never concatenate branches from the same tree into independent clusters. Record the dataset repository revision and row IDs in every output manifest.

- [ ] **Step 4: Implement explicit quality filters and rejection reasons**

Reject empty content, invalid role order, language other than English, known spam/deleted rows, prompt/response pairs above configured bounds, obvious credential patterns, and exact duplicate pairs. Do not silently remove records; write ID, source, and one enumerated reason to `rejections.jsonl`.

- [ ] **Step 5: Add 150 authored training records and validate them**

Create exactly 75 JSON/schema records and 75 multi-constraint instruction records. Each record must include author/reviewer status, provenance `lumina-demonstration-authored`, license `CC-BY-4.0`, a unique ID, capability, cluster ID, and reviewed prompt/answer messages. `data/authored/CARD.md` states that these records were authored for this evaluation, names no private company information, and records the review date.

- [ ] **Step 6: Implement the preparation CLI**

```powershell
python scripts/prepare_data.py --config configs/data.yaml --output data/processed
```

The command downloads the pinned OASST1 revision, produces exactly 3,000 accepted records in the raw-base configuration, fails closed on count mismatch, and writes selection counts by source/capability plus canonical SHA-256 hashes.

- [ ] **Step 7: Verify and commit**

Run: `pytest tests/test_data_pipeline.py -v`

Run: `ruff check src scripts tests`

Expected: tests and Ruff pass; a fixture-only dry run produces stable hashes on two consecutive executions.

Commit: `git commit -m "feat: add deterministic instruction data preparation"`.

### Task 3: Enforce cluster isolation, deduplication, and contamination gates

**Files:**
- Create: `src/lumina_experiment/isolation.py`
- Modify: `scripts/prepare_data.py`
- Create: `scripts/freeze_eval.py`
- Test: `tests/test_isolation.py`

**Interfaces:**
- Consumes: `InstructionRecord`, `EvalCase`, canonical JSON.
- Produces: `deduplicate(records) -> tuple[list[InstructionRecord], list[Duplicate]]`.
- Produces: `cluster_split(records, validation_size=300, seed=42) -> DatasetSplit`.
- Produces: `find_contamination(train, validation, evaluation, threshold=0.92) -> list[ContaminationHit]`.
- Produces: `freeze_files(paths: Sequence[Path], destination: Path) -> str`.

- [ ] **Step 1: Write failing isolation tests**

```python
def test_cluster_never_crosses_train_validation_boundary():
    split = cluster_split(records_with_repeated_clusters(), validation_size=2, seed=42)
    train_clusters = {record.cluster_id for record in split.train}
    validation_clusters = {record.cluster_id for record in split.validation}
    assert train_clusters.isdisjoint(validation_clusters)


def test_contamination_detects_close_paraphrase():
    hits = find_contamination(
        train=[record("Summarize this article in three bullets")],
        validation=[],
        evaluation=[case("Provide a three-bullet summary of this article")],
        threshold=0.80,
    )
    assert hits[0].evaluation_id == "case-1"
```

- [ ] **Step 2: Run tests and observe the missing-isolation failure**

Run: `pytest tests/test_isolation.py -v`

Expected: FAIL because `lumina_experiment.isolation` is missing.

- [ ] **Step 3: Implement canonical exact deduplication and group-aware splitting**

Normalize Unicode, line endings, and surrounding whitespace for comparison while preserving original text in outputs. Use `GroupShuffleSplit` candidates with seed 42, select the candidate closest to 300 validation records while preserving capability proportions, and fail if any cluster crosses boundaries.

- [ ] **Step 4: Implement semantic contamination screening**

Use character n-gram TF-IDF cosine similarity as the reproducible first-pass screen. Flag every train/evaluation or validation/evaluation pair at or above 0.92 and every exact normalized match. Write the complete review list; no flagged pair is automatically deleted without recording the adjudication.

- [ ] **Step 5: Implement canonical freezing**

Sort JSONL by ID, serialize each record canonically, hash each file and then a manifest containing relative path, record count, and file hash. `freeze_eval.py` refuses to overwrite `data/eval/FROZEN.sha256` unless `--replace-freeze` is supplied and records the previous hash in `data/eval/freeze-history.jsonl`.

- [ ] **Step 6: Verify and commit**

Run: `pytest tests/test_isolation.py -v`

Run: `pytest -v`

Expected: all tests pass; two fixture freezes yield the same SHA-256.

Commit: `git commit -m "feat: enforce dataset isolation and freeze hashes"`.

### Task 4: Create the 230-case evaluation corpus and deterministic scorers

**Files:**
- Create: `src/lumina_experiment/scoring.py`
- Create: `data/eval/cases/instruction.jsonl`
- Create: `data/eval/cases/reasoning.jsonl`
- Create: `data/eval/cases/knowledge.jsonl`
- Create: `data/eval/cases/summarization.jsonl`
- Create: `data/eval/cases/writing.jsonl`
- Create: `data/eval/cases/programming.jsonl`
- Create: `data/eval/cases/json.jsonl`
- Create: `data/eval/cases/safety.jsonl`
- Create: `data/eval/sources.yaml`
- Create: `data/eval/CARD.md`
- Test: `tests/test_scoring.py`
- Test: `tests/test_evaluation_corpus.py`

**Interfaces:**
- Consumes: `EvalCase`, `Generation`, `Score`.
- Produces: `score_case(case: EvalCase, generation: Generation) -> list[Score]`.
- Produces: `score_json_schema`, `score_constraints`, `score_exact_match`, `score_token_f1`, `score_required_points`, and `score_python_syntax`.
- Produces: `load_eval_directory(path: Path) -> list[EvalCase]`.

- [ ] **Step 1: Write failing scorer tests**

```python
def test_json_schema_rejects_extra_properties():
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
        "additionalProperties": False,
    }
    score = score_json_schema('{"answer":"yes","extra":1}', schema)
    assert score.passed is False
    assert score.details["parsed"] is True


def test_constraint_score_requires_every_constraint():
    constraints = [
        {"kind": "max_words", "value": 8},
        {"kind": "required_substring", "value": "solar"},
        {"kind": "forbidden_substring", "value": "fossil"},
    ]
    assert score_constraints("Solar energy reduces emissions.", constraints).passed is True
```

- [ ] **Step 2: Run scoring tests and observe failure**

Run: `pytest tests/test_scoring.py -v`

Expected: FAIL because scorers are missing.

- [ ] **Step 3: Implement deterministic scorers**

JSON scoring separates parse success from schema validity. Constraint scoring supports exact list length, maximum words, required/forbidden strings, required headings, and regex shape. Token F1 lowercases and normalizes punctuation. Summary coverage matches explicitly listed key facts and counts explicitly listed contradiction phrases. Python scoring calls `ast.parse` only; it never imports or executes generated code.

- [ ] **Step 4: Write the failing corpus-integrity test**

```python
def test_frozen_corpus_has_exact_capability_counts():
    corpus = load_eval_directory(Path("data/eval/cases"))
    counts = Counter(case.capability for case in corpus)
    assert counts == {
        "instruction": 30,
        "reasoning": 30,
        "knowledge": 30,
        "summarization": 30,
        "writing": 30,
        "programming": 30,
        "json": 30,
        "safety": 20,
    }
    assert len({case.id for case in corpus}) == 230
```

- [ ] **Step 5: Author and review the exact evaluation corpus**

Use 20 fresh authored cases plus 10 attributed public anchors for instruction following, reasoning, knowledge, summarization, and programming. Use 30 fresh authored cases each for writing and JSON, plus 20 fresh safety diagnostics. `sources.yaml` records dataset name, upstream item ID, revision, license, and retrieval date for each public anchor. Every deterministic case includes its complete reference answer, schema, constraints, required facts, or syntax expectation. Every rubric case includes a bounded 0-2 rubric with observable criteria.

- [ ] **Step 6: Document evaluation limitations**

`CARD.md` must state that the set is English-only, small, partly authored for this pilot, vulnerable to pretraining contamination in public anchors, not a safety certification, and intended for paired directional comparison rather than leaderboard claims.

- [ ] **Step 7: Verify, freeze, and commit**

Run: `pytest tests/test_scoring.py tests/test_evaluation_corpus.py -v`

Run: `python scripts/freeze_eval.py --cases data/eval/cases --sources data/eval/sources.yaml --output data/eval/FROZEN.sha256`

Save the first printed hash, run the same command again, and assert that the second printed hash is byte-for-byte identical before staging the freeze file.

Expected: 230 unique cases, exact category counts, stable hash, and no test failures.

Commit: `git commit -m "feat: freeze Lumina evaluation corpus and scorers"`.

### Task 5: Add concealed review, paired statistics, and success-gate reporting

**Files:**
- Create: `src/lumina_experiment/review.py`
- Create: `src/lumina_experiment/statistics.py`
- Create: `src/lumina_experiment/reporting.py`
- Create: `scripts/build_review_sheet.py`
- Create: `scripts/score_results.py`
- Create: `scripts/summarize_results.py`
- Test: `tests/test_review.py`
- Test: `tests/test_statistics.py`
- Test: `tests/test_reporting.py`

**Interfaces:**
- Consumes: paired `Generation` and `Score` records.
- Produces: `build_blind_pairs(generations, seed=42, sample_size=42) -> ReviewPacket`.
- Produces: `paired_bootstrap_difference(base, adapter, seed=42, resamples=10_000) -> Interval`.
- Produces: `evaluate_success(summary: ResultSummary) -> Decision`.
- Produces: `render_report(summary, manifest) -> str`.

- [ ] **Step 1: Write failing concealment and pairing tests**

```python
def test_review_packet_conceals_condition_and_randomizes_side():
    packet = build_blind_pairs(paired_generations(), seed=42, sample_size=2)
    serialized = packet.to_public_dict()
    assert "condition" not in json.dumps(serialized)
    assert {item.left_label for item in packet.items} == {"A"}
    assert {item.right_label for item in packet.items} == {"B"}


def test_missing_pair_fails_closed():
    with pytest.raises(ValueError, match="missing paired generation"):
        pair_conditions(base_generations(), adapter_generations()[:-1])
```

- [ ] **Step 2: Run review tests and observe failure**

Run: `pytest tests/test_review.py -v`

Expected: FAIL because review functions are absent.

- [ ] **Step 3: Implement concealed review packets**

Sample six writing/editing pairs and six pairs from each other capability for 42 total. Randomize left/right deterministically, store the private key only under ignored `results/raw/`, and export a public CSV containing case ID, prompt, A, B, rubric, winner field, and reviewer notes without condition names.

- [ ] **Step 4: Write failing statistics and decision tests**

```python
def test_success_requires_both_primary_metrics_to_improve():
    summary = result_summary(instruction_delta=16.7, json_delta=0.0)
    assert evaluate_success(summary).passed is False


def test_bootstrap_is_reproducible():
    first = paired_bootstrap_difference([0, 1, 0], [1, 1, 1], seed=42)
    second = paired_bootstrap_difference([0, 1, 0], [1, 1, 1], seed=42)
    assert first == second
```

- [ ] **Step 5: Implement statistics and the exact success gate**

Use paired resampling of case indices for 10,000 bootstrap replicates. Return observed absolute percentage-point difference and 2.5/97.5 percentiles. The decision object passes only when at least one primary delta is at least 15 points, both primary deltas are positive, every secondary delta is at least -5 points, writing non-tied adapter win rate exceeds 60%, and no diagnostic is marked `serious_new_failure=true`.

- [ ] **Step 6: Implement evidence-aware reporting**

Reject a report labeled `8b_gpu_measured` unless both conditions contain all 210 scored case IDs, the evaluation hash matches `FROZEN.sha256`, inference configuration hashes match, and a run manifest supplies GPU, runtime, peak VRAM, model revision, package versions, adapter hash, and generation hash. Render separate primary/secondary tables and never compute a weighted aggregate.

- [ ] **Step 7: Verify and commit**

Run: `pytest tests/test_review.py tests/test_statistics.py tests/test_reporting.py -v`

Run: `ruff check src scripts tests`

Expected: tests pass; reports fail closed for incomplete or mismatched evidence.

Commit: `git commit -m "feat: add blind review and paired result reporting"`.

### Task 6: Build the GPU runtime and generated Colab notebook

**Files:**
- Create: `src/lumina_experiment/gpu.py`
- Create: `scripts/build_notebook.py`
- Generate: `notebooks/lumina_qlora_demo.ipynb`
- Test: `tests/test_gpu_contracts.py`
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: frozen train/validation JSONL, evaluation JSONL, `ExperimentConfig`.
- Produces: `probe_gpu() -> GpuProbe`, `resolve_hub_revision(repo_id) -> str`, `train_adapter(config, datasets) -> RunManifest`, and `generate_condition(config, cases, adapter_path=None) -> list[Generation]`; each completed generation condition exports the shared typed `ConditionManifest` contract consumed by Task 5 reporting.
- Produces: `resolve_runtime_config(config: ExperimentConfig, gpu: GpuProbe) -> RuntimeConfig` and `build_generation_kwargs(config: ExperimentConfig) -> dict[str, object]`.
- Produces: a Colab notebook that calls package interfaces rather than duplicating pipeline logic.

- [ ] **Step 1: Write failing pure-contract tests**

```python
def test_t4_fallback_is_recorded_not_silent():
    adjusted = resolve_runtime_config(primary_config(), gpu(name="Tesla T4", bf16=False, vram_gb=15.0))
    assert adjusted.compute_dtype == "float16"
    assert adjusted.max_sequence_length == 1024
    assert adjusted.deviations == (
        "BF16 unsupported; used FP16",
        "VRAM below 20 GB; reduced max sequence length from 2048 to 1024",
    )


def test_generation_config_is_deterministic():
    generation = build_generation_kwargs(primary_config())
    assert generation == {"do_sample": False, "temperature": 0.0, "max_new_tokens": 512}
```

- [ ] **Step 2: Run GPU contract tests and observe failure**

Run: `pytest tests/test_gpu_contracts.py -v`

Expected: FAIL because GPU configuration functions are absent.

- [ ] **Step 3: Implement GPU functions with lazy imports**

Keep `torch`, `transformers`, `trl`, `peft`, and `bitsandbytes` imports inside GPU-only functions so the Windows test environment remains lightweight. Build `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`, `LoraConfig` from YAML, and `SFTConfig` with assistant-only loss, gradient checkpointing, deterministic seed, validation evaluation, and checkpoint retention. Assert that the tokenizer template exposes an assistant-generation mask before training.

- [ ] **Step 4: Make checkpoint selection validation-only**

Write validation metrics and checkpoint path to the manifest. The selection function reads only validation loss and configured validation task metrics; it cannot accept evaluation-set paths. Resolve and record immutable Hugging Face repository revisions before loading datasets or models.

- [ ] **Step 5: Write the failing notebook-structure test**

```python
def test_notebook_has_ordered_evidence_gates():
    notebook = nbformat.read("notebooks/lumina_qlora_demo.ipynb", as_version=4)
    headings = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    assert headings == [
        "# Lumina QLoRA Demonstrative Experiment",
        "## 1. Runtime and dependency verification",
        "## 2. Repository and frozen-data verification",
        "## 3. Small-checkpoint smoke test",
        "## 4. Untouched 8B baseline",
        "## 5. Primary rank-16 QLoRA training",
        "## 6. Adapted-model generation",
        "## 7. Export and local verification",
        "## 8. Optional rank-8 ablation",
    ]
```

- [ ] **Step 6: Generate the notebook deterministically**

`build_notebook.py` creates cells that retrieve a read-only repository archive through the GitHub API using the Colab Secret `GITHUB_TOKEN` in an HTTP authorization header, never a URL or printed command. The cells install `requirements-gpu.txt` without replacing the runtime's CUDA PyTorch, run `pip check`, print GPU/package versions, verify Git commit and freeze hashes, run the smoke test, require the user to change `RUN_8B = False` to `RUN_8B = True` before 8B work, execute baseline then training then adapted generation, delete token-bearing variables, and download a ZIP containing manifests/results but no token or model weights.

- [ ] **Step 7: Verify and commit**

Run: `python scripts/build_notebook.py`

Run: `pytest tests/test_gpu_contracts.py tests/test_notebook.py -v`

Run: `git diff --exit-code notebooks/lumina_qlora_demo.ipynb` after rebuilding a second time.

Expected: deterministic notebook, passing tests, no embedded credential values.

Commit: `git commit -m "feat: add reproducible Colab QLoRA pipeline"`.

### Task 7: Complete the local preflight and freeze the experiment

**Files:**
- Modify: `README.md`
- Create: `docs/experiment/RUNBOOK.md`
- Create: `data/processed/manifest.json` (ignored, retained locally and exported to Colab)
- Create: `results/summary/preflight.json`
- Create: `scripts/pin_revisions.py`

**Interfaces:**
- Consumes: all local modules, configs, authored data, and evaluation corpus.
- Produces: a committed `preflight.json` with `locally_verified` evidence state and hashes; it contains no model-quality result.

- [ ] **Step 1: Run the full automated suite**

Run: `pytest --cov=lumina_experiment --cov-report=term-missing -v`

Run: `ruff check src scripts tests`

Run: `ruff format --check src scripts tests`

Expected: zero failures/errors; any uncovered branch protecting hashes, evidence state, or contamination receives an explicit test before continuing.

- [ ] **Step 2: Prepare and split the 3,000-record corpus**

Run: `python scripts/pin_revisions.py --output configs/revisions.yaml`

Expected: immutable commit SHAs are recorded for the selected 8B model, smoke model, OASST1, and every public evaluation source; a second run refuses to change an existing pin without `--replace`.

Run: `python scripts/prepare_data.py --config configs/data.yaml --output data/processed`

Expected: exactly 2,700 train and 300 validation records, disjoint clusters, source counts matching the spec, stable hashes, and a non-empty rejection log where applicable.

- [ ] **Step 3: Run contamination review and freeze evaluation assets**

Adjudicate every flagged similarity hit in `data/processed/contamination-review.csv`. Replace or rewrite an evaluation case rather than admitting a close training/evaluation match. Re-run preparation until the unresolved contamination count is zero, then run `freeze_eval.py` and record the final hash in configs and `preflight.json`.

- [ ] **Step 4: Verify notebook and secret boundaries**

Run: `python scripts/build_notebook.py`

Run: `git grep -n -I -E "hf_[A-Za-z0-9]{20,}|github_pat_|ghp_|WANDB_API_KEY=.+|HF_TOKEN=.+" -- ':!requirements-dev.lock'`

Expected: no output. Confirm `git status --ignored --short` marks `.env`, `data/processed`, `results/raw`, and `artifacts` with `!!`.

- [ ] **Step 5: Write the runbook and commit preflight evidence**

Document Colab GPU selection, GitHub private-repo authorization, Colab Secret name `HF_TOKEN`, restart procedure after dependency installation, smoke-test acceptance, 8B confirmation gate, artifact download, local scoring commands, recovery from interrupted checkpoints, and the rule against presenting smoke results as 8B evidence.

Commit: `git commit -m "docs: add verified experiment preflight and runbook"`.

### Task 8: Execute smoke test, baseline, primary QLoRA run, and paired evaluation

**Files:**
- Create via Colab: `results/raw/smoke/`, `results/raw/base/`, `results/raw/primary-r16/` (ignored)
- Create: `results/summary/smoke-manifest.json`
- Create: `results/summary/base-manifest.json`
- Create: `results/summary/primary-r16-manifest.json`
- Create: `results/summary/evidence-manifest.json`
- Create: `results/summary/metrics.json`
- Create: `results/summary/metrics.csv`
- Create locally: `results/raw/scores.jsonl`, `results/raw/diagnostics.jsonl`, `results/raw/review.csv`, and `results/raw/review-key.json` (ignored)
- Create: `results/summary/failure-analysis.md`

**Interfaces:**
- Consumes: committed notebook, frozen hashes, configs, and processed-data bundle.
- Produces: validated paired evidence and `Decision` for proposal integration.

- [ ] **Step 1: Run the Colab environment and smoke-test gates**

Select an NVIDIA runtime, run notebook sections 1-3 only, and stop if dependency checks, GPU probe, revision resolution, trainable-parameter count, checkpoint reload, generation export, or smoke manifest validation fails. Download the smoke artifacts and validate them locally before enabling the 8B confirmation cell.

- [ ] **Step 2: Capture the untouched 8B baseline before training**

Run section 4 with the frozen 230 cases, retain all 210 scored outputs plus 20 diagnostics, and export the exact inference configuration hash. Do not regenerate only failed cases; restart the entire condition if the run is incomplete or settings change.

- [ ] **Step 3: Train and validate the primary adapter**

Run section 5, retain one-epoch and two-epoch checkpoints, select using validation evidence only, record runtime/peak VRAM/loss history, and calculate the adapter SHA-256. If interrupted, resume only from a manifest-matching checkpoint.

- [ ] **Step 4: Generate adapted outputs with the baseline inference hash**

Run section 6. The notebook must verify identical evaluation hash, template hash, generation kwargs, and case order before generation. Export all paired outputs and manifests.

- [ ] **Step 5: Score, review, and summarize locally**

Run:

```powershell
python scripts/score_results.py --base results/raw/base --adapter results/raw/primary-r16 --output results/raw/scores.jsonl
python scripts/build_review_sheet.py --scores results/raw/scores.jsonl --output results/raw/review.csv
python scripts/summarize_results.py --scores results/raw/scores.jsonl --review results/raw/review.csv --diagnostics results/raw/diagnostics.jsonl --output results/summary
```

Complete the 42 concealed review pairs before unblinding. If no safe code sandbox was prepared, keep programming execution disabled and report syntax plus rubric scores.

- [ ] **Step 6: Decide whether the ablation is justified**

Run rank 8 only when the primary evidence is complete, manifests validate, at least six hours remain before proposal finalization, and sufficient GPU allowance remains. Otherwise mark the ablation `planned` and preserve the primary evidence.

- [ ] **Step 7: Verify result integrity and commit summaries**

Run: `pytest -v`

Run: `python scripts/summarize_results.py --verify-only --scores results/raw/scores.jsonl --review results/raw/review.csv --diagnostics results/raw/diagnostics.jsonl --output results/summary`

Expected: 210 complete paired scored cases, 20 paired diagnostics, matching hashes, completed concealed review, and an explicit success/failure decision. Commit only manifests with sanitized paths, aggregate metrics, plots, and failure analysis; never commit raw generations or adapters.

Commit: `git commit -m "results: add validated Lumina pilot evidence"`.

### Task 9: Integrate measured evidence into proposal Part 3 and perform final audit

**Files:**
- Create: `docs/proposal/part-3-experiments.md`
- Modify: `README.md`
- Create: `results/summary/proposal-table.csv`
- Create: `results/summary/proposal-figure.png`
- Test: `tests/test_proposal_evidence.py`

**Interfaces:**
- Consumes: validated result summary, decision, evidence states, and manifests.
- Produces: proposal-ready Part 3 prose, table, figure, and repository reproduction instructions.
- Produces: `load_validated_bundle(path: Path) -> EvidenceBundle` and `extract_measured_claims(markdown: str) -> list[MeasuredClaim]` in `reporting.py`.

- [ ] **Step 1: Write the failing proposal-evidence test**

```python
def test_gpu_claims_require_validated_evidence_bundle():
    document = Path("docs/proposal/part-3-experiments.md").read_text(encoding="utf-8")
    bundle = load_validated_bundle(Path("results/summary"))
    for claim in extract_measured_claims(document):
        assert claim.evidence_state == "8b_gpu_measured"
        assert bundle.supports(claim.metric, claim.value)
```

- [ ] **Step 2: Run the test and observe failure**

Run: `pytest tests/test_proposal_evidence.py -v`

Expected: FAIL because the Part 3 document and claim extractor do not exist.

- [ ] **Step 3: Render Part 3 from validated results**

Include experimental question, frozen dataset/evaluation counts, primary configuration, identical-inference control, a compact base-versus-adapter table, absolute deltas and intervals, GPU-hours/peak VRAM, the success-gate outcome, and two representative failure observations. Label small-sample results as directional. Describe the optional ablation as measured only when its validated bundle exists.

- [ ] **Step 4: Add repository reproduction instructions**

README must distinguish local setup, evaluation freeze, Colab smoke test, 8B execution, local scoring, evidence labels, ignored artifacts, and how Parallax can reproduce the run without receiving any personal token.

- [ ] **Step 5: Run final verification**

Run: `pytest -v`

Run: `ruff check src scripts tests`

Run: `ruff format --check src scripts tests`

Run: `python scripts/summarize_results.py --verify-only --scores results/raw/scores.jsonl --review results/raw/review.csv --diagnostics results/raw/diagnostics.jsonl --output results/summary`

Run: `git grep -n -I -E "hf_[A-Za-z0-9]{20,}|github_pat_|ghp_|WANDB_API_KEY=.+|HF_TOKEN=.+"`

Expected: all tests and checks pass, evidence bundle validates, secret scan is empty, and every numeric proposal claim maps to an aggregate result record.

- [ ] **Step 6: Commit and tag the submission evidence**

Commit: `git commit -m "docs: integrate measured Lumina experiment evidence"`.

Create an annotated tag only after the proposal PDF is rendered and checked: `git tag -a lumina-proposal-v1 -m "Lumina proposal evidence snapshot"`.

Push `main` and the tag, then record the commit SHA in the PDF references or submission email.
