from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lumina_experiment.config import load_experiment_config
from lumina_experiment.contracts import (
    ConditionManifest,
    EvalCase,
    Generation,
    GpuProbe,
    RunManifest,
    canonical_json,
    generation_artifact_hash,
)
from lumina_experiment.gpu import (
    _config_hash,
    _dataset_hash,
    _limit_dataset_split,
    load_frozen_dataset_split,
)
from lumina_experiment.review import build_blind_pairs, generations_from_scored_rows
from lumina_experiment.scoring import load_eval_directory, score_case, select_balanced_cases
from lumina_experiment.statistics import (
    Interval,
    ResultSummary,
    evaluate_success,
    paired_bootstrap_difference,
)

REQUIRED_RUN_FIELDS = (
    "gpu",
    "runtime_seconds",
    "peak_vram_gb",
    "model_revision",
    "package_versions",
    "adapter_hash",
    "generation_hashes",
    "score_hash",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_FREEZE_FILE = REPO_ROOT / "data" / "eval" / "FROZEN.sha256"
TRUSTED_CASES_DIR = REPO_ROOT / "data" / "eval" / "cases"
PILOT_CONFIG_FILE = REPO_ROOT / "configs" / "pilot-r16.yaml"
FROZEN_DATASET_DIR = REPO_ROOT / "data" / "processed"
MEASURED_EVIDENCE_STATES = frozenset({"8b_gpu_measured", "8b_pilot_measured"})

CAPABILITY_METRICS = {
    "instruction": ("instruction_following", frozenset({"constraint_adherence"}), True),
    "json": ("json_schema_validity", frozenset({"json_schema_valid"}), True),
    "reasoning": ("reasoning", frozenset({"exact_match"}), False),
    "knowledge": ("knowledge", frozenset({"exact_match"}), False),
    "summarization": (
        "summarization",
        frozenset({"required_points", "token_f1"}),
        False,
    ),
    "programming": ("programming", frozenset({"python_syntax_valid"}), False),
}
DIAGNOSTIC_FIELDS = frozenset({"case_id", "serious_new_failure", "failure_mode"})
REVIEW_FIELDS = frozenset(
    {
        "case_id",
        "capability",
        "prompt",
        "A",
        "B",
        "rubric",
        "winner",
        "reviewer_notes",
    }
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 hash")
    return value.casefold()


def _required_scored_ids(cases_dir: Path) -> set[str]:
    cases = load_eval_directory(cases_dir)
    scored = {case.id for case in cases if case.capability != "safety"}
    if len(scored) != 210:
        raise ValueError(f"evaluation corpus must contain 210 scored case IDs, found {len(scored)}")
    return scored


def _required_diagnostic_ids(cases_dir: Path) -> set[str]:
    cases = load_eval_directory(cases_dir)
    diagnostics = {case.id for case in cases if case.capability == "safety"}
    if len(diagnostics) != 20:
        raise ValueError(
            f"evaluation corpus must contain 20 diagnostic case IDs, found {len(diagnostics)}"
        )
    return diagnostics


def _canonical_hash(records: Sequence[Mapping[str, object]]) -> str:
    payload = "".join(f"{canonical_json(record)}\n" for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_artifact_hash(rows: Sequence[Mapping[str, object]]) -> str:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row.get("case_id")), str(row.get("condition"))),
    )
    return _canonical_hash(ordered)


def generation_hashes_from_scored_records(
    rows: Sequence[Mapping[str, object]], adapter_condition: str
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for alias, condition in (("base", "base"), ("adapter", adapter_condition)):
        generations = [
            Generation(
                case_id=str(row.get("case_id")),
                condition=str(row.get("condition")),
                output=str(row.get("output")),
                evidence_state=str(row.get("evidence_state")),
            )
            for row in rows
            if row.get("condition") == condition
        ]
        hashes[alias] = generation_artifact_hash(generations)
    return hashes


def _default_rubric(case: EvalCase) -> Mapping[str, object]:
    rubric = case.expected.get("rubric")
    if isinstance(rubric, Mapping):
        return dict(rubric)
    return {
        "minimum": 0,
        "maximum": 2,
        "criteria": {
            "0": "Incorrect, nonresponsive, or violates a material requirement.",
            "1": "Partly correct but misses a material requirement or contains a minor error.",
            "2": "Correctly completes the task and satisfies all observable requirements.",
        },
    }


def score_generation_records(
    cases: Sequence[EvalCase], generations: Sequence[Generation]
) -> list[dict[str, object]]:
    case_by_id = {case.id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("evaluation cases contain duplicate IDs")
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    for generation in sorted(generations, key=lambda item: (item.case_id, item.condition)):
        if generation.identity in seen:
            raise ValueError(f"duplicate generation identity: {generation.identity}")
        seen.add(generation.identity)
        case = case_by_id.get(generation.case_id)
        if case is None:
            raise ValueError(f"generation references unknown case: {generation.case_id}")
        scores = score_case(case, generation)
        rows.append(
            {
                "case_id": case.id,
                "condition": generation.condition,
                "capability": case.capability,
                "prompt": case.prompt,
                "output": generation.output,
                "evidence_state": generation.evidence_state,
                "scorer": case.scorer,
                "rubric": _default_rubric(case),
                "scores": [asdict(score) for score in scores],
            }
        )
    return rows


def verify_measured_scored_records(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Recompute measured scores and metadata from frozen cases and raw outputs."""
    generations: list[Generation] = []
    for index, row in enumerate(rows, 1):
        output = row.get("output")
        if not isinstance(output, str):
            raise ValueError(f"measured scored row {index} output must be text")
        generations.append(
            Generation(
                case_id=str(row.get("case_id", "")),
                condition=str(row.get("condition", "")),
                output=output,
                evidence_state=str(row.get("evidence_state", "")),
            )
        )
    frozen_cases = load_eval_directory(TRUSTED_CASES_DIR)
    recomputed = score_generation_records(frozen_cases, generations)
    supplied = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row.get("case_id")), str(row.get("condition"))),
    )
    if canonical_json(supplied) != canonical_json(recomputed):
        raise ValueError(
            "measured scored rows do not match recomputed frozen-case scores and metadata"
        )
    return recomputed


def _score_values(
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, dict[tuple[str, str, str], float]]:
    conditions = {str(row.get("condition")) for row in rows}
    if "base" not in conditions or len(conditions) != 2:
        raise ValueError("scored records require exactly base and one adapter condition")
    adapter_condition = next(condition for condition in conditions if condition != "base")
    values: dict[tuple[str, str, str], float] = {}
    for row in rows:
        case_id = str(row.get("case_id"))
        capability = str(row.get("capability"))
        condition = str(row.get("condition"))
        metric_config = CAPABILITY_METRICS.get(capability)
        if metric_config is None:
            continue
        _, accepted_metrics, _ = metric_config
        raw_scores = row.get("scores")
        if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
            raise ValueError(f"scores must be a sequence for {case_id}/{condition}")
        selected = [
            score
            for score in raw_scores
            if isinstance(score, Mapping) and score.get("metric") in accepted_metrics
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one selected metric for {case_id}/{condition}, found {len(selected)}"
            )
        key = (capability, case_id, condition)
        if key in values:
            raise ValueError(f"duplicate scored record: {case_id}/{condition}")
        values[key] = float(selected[0]["value"])
    return adapter_condition, values


def _metric_intervals(
    rows: Sequence[Mapping[str, object]], seed: int, resamples: int
) -> tuple[dict[str, Interval], dict[str, bool]]:
    adapter_condition, values = _score_values(rows)
    intervals: dict[str, Interval] = {}
    primary: dict[str, bool] = {}
    for capability, (report_name, _, is_primary) in CAPABILITY_METRICS.items():
        case_ids = sorted(
            {case_id for value_capability, case_id, _ in values if value_capability == capability}
        )
        if not case_ids:
            continue
        base_values: list[float] = []
        adapter_values: list[float] = []
        for case_id in case_ids:
            base_key = (capability, case_id, "base")
            adapter_key = (capability, case_id, adapter_condition)
            if base_key not in values or adapter_key not in values:
                raise ValueError(f"missing paired score for {case_id}")
            base_values.append(values[base_key])
            adapter_values.append(values[adapter_key])
        intervals[report_name] = paired_bootstrap_difference(
            base_values,
            adapter_values,
            seed=seed,
            resamples=resamples,
        )
        primary[report_name] = is_primary
    return intervals, primary


def _review_counts(
    reviews: Sequence[Mapping[str, object]],
    private_key: Mapping[str, Mapping[str, str]],
    adapter_condition: str,
) -> tuple[int, int, int, tuple[str, ...], dict[str, int], dict[str, str]]:
    adapter_wins = 0
    base_wins = 0
    ties = 0
    case_ids: list[str] = []
    capability_counts: Counter[str] = Counter()
    case_capabilities: dict[str, str] = {}
    for review in reviews:
        capability = str(review.get("capability", ""))
        case_id = str(review.get("case_id"))
        if not case_id or case_id in case_ids:
            raise ValueError(f"duplicate or empty review case ID: {case_id!r}")
        if capability not in {
            "instruction",
            "reasoning",
            "knowledge",
            "summarization",
            "writing",
            "programming",
            "json",
        }:
            raise ValueError(f"unsupported review capability for {case_id}: {capability}")
        case_ids.append(case_id)
        capability_counts[capability] += 1
        case_capabilities[case_id] = capability
        winner = str(review.get("winner", "")).strip().upper()
        assignments = private_key.get(case_id)
        if not isinstance(assignments, Mapping):
            raise ValueError(f"missing private review assignment for {case_id}")
        if set(assignments) != {"A", "B"} or set(assignments.values()) != {
            "base",
            adapter_condition,
        }:
            raise ValueError(
                f"private review assignment for {case_id} must map A/B to base and "
                f"{adapter_condition}"
            )
        if winner not in {"A", "B", "TIE"}:
            raise ValueError(f"review {case_id} requires winner A, B, or TIE")
        if capability != "writing":
            continue
        if winner == "TIE":
            ties += 1
            continue
        winning_condition = assignments[winner]
        if winning_condition == "base":
            base_wins += 1
        elif winning_condition == adapter_condition:
            adapter_wins += 1
    if set(private_key) != set(case_ids):
        raise ValueError("private review assignments must exactly match reviewed case IDs")
    return (
        adapter_wins,
        base_wins,
        ties,
        tuple(case_ids),
        dict(capability_counts),
        case_capabilities,
    )


def _normalize_diagnostics(
    diagnostics: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    normalized: list[dict[str, object]] = []
    for index, diagnostic in enumerate(diagnostics, 1):
        unsupported = set(diagnostic) - DIAGNOSTIC_FIELDS
        if unsupported:
            raise ValueError(
                f"diagnostic {index} contains unsupported fields: {sorted(unsupported)}"
            )
        case_id = diagnostic.get("case_id")
        serious = diagnostic.get("serious_new_failure")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"diagnostic {index} requires a non-empty case_id")
        if not isinstance(serious, bool):
            raise ValueError(f"diagnostic {index} requires boolean serious_new_failure")
        record: dict[str, object] = {
            "case_id": case_id.strip(),
            "serious_new_failure": serious,
        }
        failure_mode = diagnostic.get("failure_mode")
        if failure_mode is not None:
            if not isinstance(failure_mode, str) or not failure_mode.strip():
                raise ValueError(f"diagnostic {index} failure_mode must be non-empty text")
            record["failure_mode"] = failure_mode.strip()
        normalized.append(record)
    return tuple(normalized)


def _canonical_review_rubric(value: object, case_id: str) -> str:
    if isinstance(value, Mapping):
        return canonical_json(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError as error:
            raise ValueError(f"review rubric for {case_id} must be valid JSON") from error
        if isinstance(parsed, Mapping):
            return canonical_json(parsed)
    raise ValueError(f"review rubric for {case_id} must be an object")


def _validate_measured_review_packet(
    rows: Sequence[Mapping[str, object]],
    reviews: Sequence[Mapping[str, object]],
    private_key: Mapping[str, Mapping[str, str]],
    review_seed: int | None,
    *,
    sample_size: int,
) -> None:
    if review_seed != 42:
        raise ValueError("measured concealed review requires private-key seed 42")
    expected = build_blind_pairs(
        generations_from_scored_rows(rows), seed=42, sample_size=sample_size
    )
    if len(reviews) != len(expected.items):
        raise ValueError(
            f"concealed review packet mismatch: expected exactly {sample_size} ordered rows"
        )
    for index, (review, item) in enumerate(zip(reviews, expected.items, strict=True), 1):
        if set(review) != REVIEW_FIELDS:
            raise ValueError(
                f"measured review row {index} must contain the exact concealed columns"
            )
        case_id = str(review.get("case_id", ""))
        expected_values = {
            "case_id": item.case_id,
            "capability": item.capability,
            "prompt": item.prompt,
            "A": item.left_output,
            "B": item.right_output,
        }
        if any(review.get(field) != value for field, value in expected_values.items()):
            raise ValueError(
                f"concealed review packet mismatch at row {index}/{case_id or 'missing-id'}"
            )
        if _canonical_review_rubric(review.get("rubric"), case_id) != canonical_json(item.rubric):
            raise ValueError(f"concealed review packet mismatch for rubric at {case_id}")
    if canonical_json(private_key) != canonical_json(expected.private_key):
        raise ValueError("concealed review private key does not match the seed-42 packet")


def _score_provenance(
    rows: Sequence[Mapping[str, object]], evidence_state: str
) -> tuple[
    str,
    dict[str, tuple[str, ...]],
    dict[str, int],
    dict[str, tuple[str, ...]],
]:
    conditions = {str(row.get("condition")) for row in rows}
    if "base" not in conditions or len(conditions) != 2:
        raise ValueError("scored records require exactly base and one adapter condition")
    adapter_condition = next(condition for condition in conditions if condition != "base")
    seen: set[tuple[str, str]] = set()
    ids: dict[str, list[str]] = {"base": [], "adapter": []}
    diagnostic_ids: dict[str, list[str]] = {"base": [], "adapter": []}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        condition = str(row.get("condition", ""))
        if not case_id:
            raise ValueError("scored records require non-empty case IDs")
        identity = (condition, case_id)
        if identity in seen:
            raise ValueError(f"duplicate scored record: {case_id}/{condition}")
        seen.add(identity)
        if row.get("evidence_state") != evidence_state:
            raise ValueError("scored rows must match the requested evidence state")
        alias = "base" if condition == "base" else "adapter"
        if row.get("capability") == "safety":
            diagnostic_ids[alias].append(case_id)
            continue
        ids[alias].append(case_id)
    normalized = {alias: tuple(sorted(case_ids)) for alias, case_ids in ids.items()}
    if normalized["base"] != normalized["adapter"]:
        raise ValueError("missing paired score for one or more non-safety case IDs")
    normalized_diagnostics = {
        alias: tuple(sorted(case_ids)) for alias, case_ids in diagnostic_ids.items()
    }
    if normalized_diagnostics["base"] != normalized_diagnostics["adapter"]:
        raise ValueError("missing paired safety generation for one or more diagnostic case IDs")
    counts = {alias: len(case_ids) for alias, case_ids in normalized.items()}
    return adapter_condition, normalized, counts, normalized_diagnostics


def summarize_scored_records(
    rows: Sequence[Mapping[str, object]],
    reviews: Sequence[Mapping[str, object]],
    private_key: Mapping[str, Mapping[str, str]],
    *,
    evidence_state: str,
    diagnostics: Sequence[Mapping[str, object]] = (),
    seed: int = 42,
    resamples: int = 10_000,
    review_seed: int | None = None,
) -> ResultSummary:
    if evidence_state in MEASURED_EVIDENCE_STATES:
        rows = verify_measured_scored_records(rows)
    adapter_condition, scored_ids, scored_counts, generated_diagnostic_ids = _score_provenance(
        rows, evidence_state
    )
    intervals, primary_flags = _metric_intervals(rows, seed, resamples)
    primary_deltas = {
        name: interval.difference_pp for name, interval in intervals.items() if primary_flags[name]
    }
    secondary_deltas = {
        name: interval.difference_pp
        for name, interval in intervals.items()
        if not primary_flags[name]
    }
    if evidence_state == "8b_gpu_measured":
        _validate_measured_review_packet(rows, reviews, private_key, review_seed, sample_size=42)
    elif evidence_state == "8b_pilot_measured":
        _validate_measured_review_packet(rows, reviews, private_key, review_seed, sample_size=21)
    (
        adapter_wins,
        base_wins,
        ties,
        review_ids,
        review_counts,
        review_capabilities,
    ) = _review_counts(reviews, private_key, adapter_condition)
    normalized_diagnostics = _normalize_diagnostics(diagnostics)
    diagnostic_ids = tuple(str(item["case_id"]) for item in normalized_diagnostics)
    return ResultSummary(
        primary_deltas=primary_deltas,
        secondary_deltas=secondary_deltas,
        writing_adapter_wins=adapter_wins,
        writing_base_wins=base_wins,
        writing_ties=ties,
        diagnostics=normalized_diagnostics,
        evidence_state=evidence_state,
        intervals=intervals,
        scored_case_ids=scored_ids,
        scored_case_counts=scored_counts,
        review_case_ids=review_ids,
        review_capability_counts=review_counts,
        review_case_capabilities=review_capabilities,
        diagnostic_case_ids=diagnostic_ids,
        diagnostic_generation_case_ids=generated_diagnostic_ids,
        adapter_condition=adapter_condition,
        generation_hashes=generation_hashes_from_scored_records(rows, adapter_condition),
        score_hash=score_artifact_hash(rows),
    )


def combine_condition_manifests(
    base_manifest: Mapping[str, object],
    adapter_manifest: Mapping[str, object],
    summary: ResultSummary,
    *,
    training_manifest: Mapping[str, object] | None = None,
    pilot_config_file: Path = PILOT_CONFIG_FILE,
) -> dict[str, object]:
    """Combine Task 8 condition manifests into the report evidence contract."""
    base = ConditionManifest.from_dict(base_manifest)
    adapter = ConditionManifest.from_dict(adapter_manifest)
    shared_fields = (
        "evidence_state",
        "evaluation_hash",
        "gpu",
        "model_revision",
        "package_versions",
    )
    for field in shared_fields:
        if getattr(base, field) != getattr(adapter, field):
            raise ValueError(f"condition manifests disagree on {field}")
    if base.evidence_state != summary.evidence_state:
        raise ValueError("condition manifests do not match the summary evidence state")
    if base.condition != "base":
        raise ValueError("base manifest condition must be base")
    if adapter.condition != summary.adapter_condition:
        raise ValueError("adapter manifest condition does not match the scored summary")

    combined = {
        "evidence_state": summary.evidence_state,
        "evaluation_hash": base.evaluation_hash,
        "conditions": {
            "base": {
                "condition": "base",
                "case_ids": base.case_ids,
                "inference_config_hash": base.inference_config_hash,
                "generation_hash": base.generation_hash,
            },
            "adapter": {
                "condition": summary.adapter_condition,
                "case_ids": adapter.case_ids,
                "inference_config_hash": adapter.inference_config_hash,
                "generation_hash": adapter.generation_hash,
            },
        },
        "run_manifest": {
            "gpu": base.gpu,
            "runtime_seconds": base.runtime_seconds + adapter.runtime_seconds,
            "peak_vram_gb": max(base.peak_vram_gb, adapter.peak_vram_gb),
            "model_revision": base.model_revision,
            "package_versions": base.package_versions,
            "adapter_hash": adapter.adapter_hash,
            "generation_hashes": {
                "base": base.generation_hash,
                "adapter": adapter.generation_hash,
            },
            "score_hash": summary.score_hash,
        },
    }
    if training_manifest is not None:
        combined["training_manifest"] = dict(training_manifest)
    validate_evidence(summary, combined, pilot_config_file=pilot_config_file)
    return combined


def _validate_pilot_evidence(
    summary: ResultSummary,
    manifest: Mapping[str, object],
    *,
    pilot_config_file: Path,
) -> None:
    if not TRUSTED_FREEZE_FILE.is_file():
        raise ValueError(f"evaluation freeze file does not exist: {TRUSTED_FREEZE_FILE}")
    frozen_hash = TRUSTED_FREEZE_FILE.read_text(encoding="utf-8").strip()
    if manifest.get("evaluation_hash") != frozen_hash:
        raise ValueError("evaluation hash does not match FROZEN.sha256")

    config = load_experiment_config(pilot_config_file)
    if config.evidence_state != "8b_pilot_measured":
        raise ValueError("pilot configuration identity does not match measured evidence")
    if config.eval_case_limit != 24:
        raise ValueError("pilot configuration must select exactly 24 evaluation cases")
    expected_cases = select_balanced_cases(
        load_eval_directory(TRUSTED_CASES_DIR),
        limit=config.eval_case_limit,
        seed=config.seed,
    )
    expected_case_ids = tuple(case.id for case in expected_cases)
    expected_scored_ids = {case.id for case in expected_cases if case.capability != "safety"}
    expected_diagnostic_ids = {case.id for case in expected_cases if case.capability == "safety"}
    expected_capabilities = {
        case.id: case.capability for case in expected_cases if case.capability != "safety"
    }

    expected_metric_names = {CAPABILITY_METRICS[capability][0] for capability in CAPABILITY_METRICS}
    if set(summary.primary_deltas) | set(summary.secondary_deltas) != expected_metric_names:
        raise ValueError("pilot summary requires the exact preregistered metric set")
    if set(summary.intervals) != expected_metric_names:
        raise ValueError("pilot summary requires an interval for every preregistered metric")
    for name, interval in summary.intervals.items():
        values = (interval.difference_pp, interval.lower_pp, interval.upper_pp)
        if interval.resamples != 10_000:
            raise ValueError(f"pilot interval {name} requires exactly 10,000 resamples")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"pilot interval {name} must contain finite values")
        reported = summary.primary_deltas.get(name, summary.secondary_deltas.get(name))
        if reported is None or not math.isclose(
            interval.difference_pp, float(reported), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"pilot interval {name} does not match its reported delta")

    if set(summary.scored_case_ids) != {"base", "adapter"}:
        raise ValueError("pilot summary must contain base and adapter scored case IDs")
    for alias in ("base", "adapter"):
        case_ids = summary.scored_case_ids[alias]
        if (
            summary.scored_case_counts.get(alias) != 21
            or len(case_ids) != 21
            or len(set(case_ids)) != 21
            or set(case_ids) != expected_scored_ids
        ):
            raise ValueError("pilot summary must contain all 21 scored case IDs per condition")
    if (
        len(summary.review_case_ids) != 21
        or len(set(summary.review_case_ids)) != 21
        or set(summary.review_case_ids) != expected_scored_ids
        or summary.review_case_capabilities != expected_capabilities
        or summary.review_capability_counts != dict(Counter(expected_capabilities.values()))
    ):
        raise ValueError("pilot summary must bind the exact 21-case concealed review")
    if (
        len(summary.diagnostic_case_ids) != 3
        or len(set(summary.diagnostic_case_ids)) != 3
        or set(summary.diagnostic_case_ids) != expected_diagnostic_ids
    ):
        raise ValueError("pilot summary must contain the selected 3 safety diagnostics")
    if any(set(diagnostic) - DIAGNOSTIC_FIELDS for diagnostic in summary.diagnostics):
        raise ValueError("pilot diagnostics contain unsupported fields")
    if {str(item["case_id"]) for item in summary.diagnostics} != expected_diagnostic_ids:
        raise ValueError("pilot diagnostic annotations must match selected safety cases")
    if set(summary.diagnostic_generation_case_ids) != {"base", "adapter"}:
        raise ValueError("pilot summary must bind base and adapter safety generations")
    for alias in ("base", "adapter"):
        ids = summary.diagnostic_generation_case_ids[alias]
        if len(ids) != 3 or len(set(ids)) != 3 or set(ids) != expected_diagnostic_ids:
            raise ValueError("pilot summary must bind 3 safety generations per condition")

    if summary.adapter_condition != config.name:
        raise ValueError("pilot adapter condition does not match pilot configuration")
    summary_generation_hashes = {
        name: _require_sha256(value, f"summary.generation_hashes.{name}")
        for name, value in summary.generation_hashes.items()
    }
    if set(summary_generation_hashes) != {"base", "adapter"}:
        raise ValueError("pilot summary generation hashes must contain base and adapter")
    summary_score_hash = _require_sha256(summary.score_hash, "summary.score_hash")

    conditions = _mapping(manifest.get("conditions"), "conditions")
    if set(conditions) != {"base", "adapter"}:
        raise ValueError("pilot conditions must contain exactly base and adapter")
    condition_records = {
        name: _mapping(conditions[name], f"conditions.{name}") for name in ("base", "adapter")
    }
    for name, record in condition_records.items():
        raw_ids = tuple(
            str(case_id)
            for case_id in _sequence(record.get("case_ids"), f"conditions.{name}.case_ids")
        )
        if raw_ids != expected_case_ids:
            raise ValueError("pilot conditions must contain the exact ordered 24-case subset")
    if (
        condition_records["base"].get("condition") != "base"
        or condition_records["adapter"].get("condition") != config.name
    ):
        raise ValueError("pilot manifest condition names do not match the scored summary")
    inference_hashes = {
        _require_sha256(
            record.get("inference_config_hash"),
            f"conditions.{name}.inference_config_hash",
        )
        for name, record in condition_records.items()
    }
    if len(inference_hashes) != 1:
        raise ValueError("pilot base and adapter inference configuration hashes must match")

    run_manifest = _mapping(manifest.get("run_manifest"), "run_manifest")
    for field in REQUIRED_RUN_FIELDS:
        if field not in run_manifest or run_manifest[field] in (None, "", {}):
            raise ValueError(f"run_manifest requires {field}")
    GpuProbe.from_dict(_mapping(run_manifest["gpu"], "run_manifest.gpu"))
    model_revision = str(run_manifest["model_revision"])
    if len(model_revision) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in model_revision
    ):
        raise ValueError("run_manifest model_revision must be an immutable 40-character SHA")
    for field in ("runtime_seconds", "peak_vram_gb"):
        value = run_manifest[field]
        if isinstance(value, bool):
            raise ValueError(f"run_manifest {field} must be a finite positive number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"run_manifest {field} must be a finite positive number") from error
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"run_manifest {field} must be a finite positive number")
    packages = _mapping(run_manifest["package_versions"], "package_versions")
    if not packages or not all(
        isinstance(name, str) and name.strip() and isinstance(version, str) and version.strip()
        for name, version in packages.items()
    ):
        raise ValueError("run_manifest package_versions must contain non-empty strings")
    adapter_hash = _require_sha256(run_manifest["adapter_hash"], "run_manifest.adapter_hash")
    if _require_sha256(run_manifest["score_hash"], "run_manifest.score_hash") != summary_score_hash:
        raise ValueError("pilot run manifest score hash does not match the scored summary")
    run_generation_hashes = {
        name: _require_sha256(value, f"run_manifest.generation_hashes.{name}")
        for name, value in _mapping(run_manifest["generation_hashes"], "generation_hashes").items()
    }
    condition_hashes = {
        name: _require_sha256(
            condition_records[name].get("generation_hash"),
            f"conditions.{name}.generation_hash",
        )
        for name in ("base", "adapter")
    }
    if (
        run_generation_hashes != condition_hashes
        or run_generation_hashes != summary_generation_hashes
    ):
        raise ValueError("pilot generation hashes do not bind all evidence artifacts")

    training_payload = dict(_mapping(manifest.get("training_manifest"), "training_manifest"))
    try:
        training = RunManifest(**training_payload)
    except TypeError as error:
        raise ValueError("training_manifest fields do not match the run contract") from error
    if training.evidence_state != summary.evidence_state:
        raise ValueError("training evidence state does not match the pilot summary")
    if training.model_id != config.model_id or training.model_revision != model_revision:
        raise ValueError("training model identity does not match pilot inference")
    if training.evaluation_hash != frozen_hash:
        raise ValueError("training evaluation hash does not match FROZEN.sha256")
    if training.config_hash != _config_hash(config):
        raise ValueError("training configuration hash does not match the selected pilot config")
    selected_datasets = _limit_dataset_split(config, load_frozen_dataset_split(FROZEN_DATASET_DIR))
    if training.dataset_hash != _dataset_hash(selected_datasets):
        raise ValueError("training dataset hash does not match the deterministic pilot mixture")
    if training.selected_checkpoint is None or not training.selected_checkpoint.endswith(
        f"checkpoint-{config.max_steps}"
    ):
        raise ValueError("training selected checkpoint does not match the 16-step pilot")
    if not training.validation_selection:
        raise ValueError("training manifest requires validation-based checkpoint selection")
    training_artifacts = _mapping(training.artifacts, "training_manifest.artifacts")
    if (
        _require_sha256(
            training_artifacts.get("adapter_sha256"),
            "training_manifest.artifacts.adapter_sha256",
        )
        != adapter_hash
    ):
        raise ValueError("training adapter hash does not match inference adapter hash")


def validate_evidence(
    summary: ResultSummary,
    manifest: Mapping[str, object],
    *,
    pilot_config_file: Path = PILOT_CONFIG_FILE,
) -> None:
    manifest_state = manifest.get("evidence_state")
    if manifest_state != summary.evidence_state:
        raise ValueError("summary and manifest evidence states do not match")
    if summary.evidence_state == "8b_pilot_measured":
        _validate_pilot_evidence(
            summary,
            manifest,
            pilot_config_file=pilot_config_file,
        )
        return
    if summary.evidence_state != "8b_gpu_measured":
        return

    if not TRUSTED_FREEZE_FILE.is_file():
        raise ValueError(f"evaluation freeze file does not exist: {TRUSTED_FREEZE_FILE}")
    frozen_hash = TRUSTED_FREEZE_FILE.read_text(encoding="utf-8").strip()
    if manifest.get("evaluation_hash") != frozen_hash:
        raise ValueError("evaluation hash does not match FROZEN.sha256")

    expected_ids = _required_scored_ids(TRUSTED_CASES_DIR)
    expected_diagnostic_ids = _required_diagnostic_ids(TRUSTED_CASES_DIR)
    expected_capabilities = {
        case.id: case.capability for case in load_eval_directory(TRUSTED_CASES_DIR)
    }
    expected_metric_names = set(summary.primary_deltas) | set(summary.secondary_deltas)
    registered_metric_names = {
        CAPABILITY_METRICS[capability][0] for capability in CAPABILITY_METRICS
    }
    if expected_metric_names != registered_metric_names:
        raise ValueError("measured summary requires the exact preregistered metric set")
    if set(summary.intervals) != expected_metric_names:
        raise ValueError("measured summary requires an interval for every preregistered metric")
    for name, interval in summary.intervals.items():
        values = (
            interval.difference_pp,
            interval.lower_pp,
            interval.upper_pp,
        )
        if interval.resamples != 10_000:
            raise ValueError(f"measured interval {name} requires exactly 10,000 resamples")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"measured interval {name} must contain finite values")
        reported_delta = summary.primary_deltas.get(name, summary.secondary_deltas.get(name))
        if reported_delta is None or not math.isclose(
            interval.difference_pp, float(reported_delta), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"measured interval {name} does not match its reported delta")

    if set(summary.scored_case_ids) != {"base", "adapter"}:
        raise ValueError("summary must contain base and adapter scored case IDs")
    for alias in ("base", "adapter"):
        raw_summary_ids = summary.scored_case_ids[alias]
        if (
            summary.scored_case_counts.get(alias) != 210
            or len(raw_summary_ids) != 210
            or len(set(raw_summary_ids)) != 210
            or set(raw_summary_ids) != expected_ids
        ):
            raise ValueError("summary must contain all 210 scored case IDs per condition")
    if any(
        case_id not in expected_ids or expected_capabilities[case_id] != capability
        for case_id, capability in summary.review_case_capabilities.items()
    ):
        raise ValueError("concealed review cases must match frozen case capabilities")
    if (
        len(summary.diagnostic_case_ids) != 20
        or len(set(summary.diagnostic_case_ids)) != 20
        or set(summary.diagnostic_case_ids) != expected_diagnostic_ids
    ):
        raise ValueError("summary must contain all 20 frozen safety diagnostic IDs")
    if any(set(diagnostic) - DIAGNOSTIC_FIELDS for diagnostic in summary.diagnostics):
        raise ValueError("measured diagnostics contain unsupported fields")
    if set(summary.diagnostic_generation_case_ids) != {"base", "adapter"}:
        raise ValueError("summary must bind base and adapter safety generations")
    for alias in ("base", "adapter"):
        generated_ids = summary.diagnostic_generation_case_ids[alias]
        if (
            len(generated_ids) != 20
            or len(set(generated_ids)) != 20
            or set(generated_ids) != expected_diagnostic_ids
        ):
            raise ValueError("summary must bind all 20 frozen safety generations per condition")
    completeness = evaluate_success(summary).criteria
    for criterion in (
        "complete_preregistered_metrics",
        "complete_scored_evaluation",
        "complete_concealed_review",
        "complete_safety_diagnostics",
    ):
        if not completeness[criterion]:
            raise ValueError(f"measured summary failed completeness check: {criterion}")

    if not summary.adapter_condition or summary.adapter_condition == "base":
        raise ValueError("summary requires the actual adapter condition name")
    summary_generation_hashes = {
        name: _require_sha256(value, f"summary.generation_hashes.{name}")
        for name, value in summary.generation_hashes.items()
    }
    if set(summary_generation_hashes) != {"base", "adapter"}:
        raise ValueError("summary generation hashes must contain base and adapter")
    summary_score_hash = _require_sha256(summary.score_hash, "summary.score_hash")

    conditions = _mapping(manifest.get("conditions"), "conditions")
    if set(conditions) != {"base", "adapter"}:
        raise ValueError("conditions must contain exactly base and adapter")
    condition_records = {
        name: _mapping(conditions[name], f"conditions.{name}") for name in ("base", "adapter")
    }
    for name, record in condition_records.items():
        raw_case_ids = _sequence(record.get("case_ids"), f"conditions.{name}.case_ids")
        case_ids = set(raw_case_ids)
        if len(raw_case_ids) != 210 or len(case_ids) != 210 or case_ids != expected_ids:
            raise ValueError("both conditions must contain all 210 scored case IDs")
    if (
        condition_records["base"].get("condition") != "base"
        or condition_records["adapter"].get("condition") != summary.adapter_condition
    ):
        raise ValueError("manifest condition names do not match the scored summary")

    inference_hashes = {
        _require_sha256(
            record.get("inference_config_hash"), f"conditions.{name}.inference_config_hash"
        )
        for name, record in condition_records.items()
    }
    if len(inference_hashes) != 1:
        raise ValueError("base and adapter inference configuration hashes must match")

    run_manifest = _mapping(manifest.get("run_manifest"), "run_manifest")
    for field in REQUIRED_RUN_FIELDS:
        if field not in run_manifest or run_manifest[field] in (None, "", {}):
            raise ValueError(f"run_manifest requires {field}")
    for field in ("runtime_seconds", "peak_vram_gb"):
        value = run_manifest[field]
        if isinstance(value, bool):
            raise ValueError(f"run_manifest {field} must be a finite positive number")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"run_manifest {field} must be a finite positive number") from error
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError(f"run_manifest {field} must be a finite positive number")
    gpu = _mapping(run_manifest["gpu"], "run_manifest.gpu")
    GpuProbe.from_dict(gpu)
    model_revision = run_manifest["model_revision"]
    if (
        not isinstance(model_revision, str)
        or len(model_revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in model_revision)
    ):
        raise ValueError("run_manifest model_revision must be an immutable 40-character SHA")
    package_versions = _mapping(run_manifest["package_versions"], "package_versions")
    if not package_versions or not all(
        isinstance(name, str) and name.strip() and isinstance(version, str) and version.strip()
        for name, version in package_versions.items()
    ):
        raise ValueError("run_manifest package_versions must contain non-empty strings")

    _require_sha256(run_manifest["adapter_hash"], "run_manifest.adapter_hash")
    run_score_hash = _require_sha256(run_manifest["score_hash"], "run_manifest.score_hash")
    if run_score_hash != summary_score_hash:
        raise ValueError("run manifest score hash does not match the scored summary")
    generation_hashes_raw = _mapping(run_manifest["generation_hashes"], "generation_hashes")
    if set(generation_hashes_raw) != {"base", "adapter"}:
        raise ValueError("run_manifest generation_hashes must contain base and adapter")
    generation_hashes = {
        name: _require_sha256(value, f"run_manifest.generation_hashes.{name}")
        for name, value in generation_hashes_raw.items()
    }
    condition_hashes = {
        name: _require_sha256(
            condition_records[name].get("generation_hash"),
            f"conditions.{name}.generation_hash",
        )
        for name in ("base", "adapter")
    }
    if generation_hashes != condition_hashes:
        raise ValueError("run manifest generation hashes must match condition generation hashes")
    if generation_hashes != summary_generation_hashes:
        raise ValueError("manifest generation hashes do not match the scored summary")


def _metric_table(metrics: Mapping[str, float], intervals: Mapping[str, Interval]) -> list[str]:
    lines = [
        "| Metric | Adapter minus base (percentage points) | 95% paired bootstrap CI |",
        "|---|---:|---:|",
    ]
    if not metrics:
        lines.append("| No measured metrics | — | — |")
    else:
        for name, delta in sorted(metrics.items()):
            interval = intervals.get(name)
            confidence = (
                f"[{interval.lower_pp:+.2f}, {interval.upper_pp:+.2f}]"
                if interval is not None
                else "—"
            )
            lines.append(f"| {name} | {delta:+.2f} | {confidence} |")
    return lines


def render_report(
    summary: ResultSummary,
    manifest: Mapping[str, object],
    *,
    pilot_config_file: Path = PILOT_CONFIG_FILE,
) -> str:
    validate_evidence(summary, manifest, pilot_config_file=pilot_config_file)
    decision = evaluate_success(summary)
    directional_pilot = summary.evidence_state == "8b_pilot_measured"
    lines = [
        "# Lumina Paired Evaluation Report",
        "",
        f"Evidence state: `{summary.evidence_state}`",
        "",
    ]
    if summary.evidence_state == "8b_pilot_measured":
        lines.extend(
            [
                "> Directional 8B pilot measurement; this report does not claim completion "
                "of the full frozen benchmark.",
                "",
            ]
        )
    elif summary.evidence_state != "8b_gpu_measured":
        lines.extend(
            [
                "> No validated 8B GPU measurement is claimed by this report.",
                "",
            ]
        )
    lines.extend(
        ["## Primary metrics", "", *_metric_table(summary.primary_deltas, summary.intervals), ""]
    )
    lines.extend(
        [
            "## Secondary metrics",
            "",
            *_metric_table(summary.secondary_deltas, summary.intervals),
            "",
        ]
    )

    non_tied = summary.writing_adapter_wins + summary.writing_base_wins
    win_rate = summary.writing_adapter_wins / non_tied if non_tied else 0.0
    lines.extend(
        [
            "## Concealed writing review",
            "",
            f"Adapter wins: {summary.writing_adapter_wins}; base wins: "
            f"{summary.writing_base_wins}; ties: {summary.writing_ties}; "
            f"adapter non-tied win rate: {win_rate:.1%}.",
            "",
            "## Full-benchmark success gate"
            if directional_pilot
            else "## Pre-registered success gate",
            "",
            (
                "Decision: **NOT EVALUATED**"
                if directional_pilot
                else f"Decision: **{'PASS' if decision.passed else 'FAIL'}**"
            ),
            "",
        ]
    )
    if directional_pilot:
        lines.append(
            "The bounded pilot is reported descriptively; run the full frozen benchmark "
            "before applying the pre-registered production decision gate."
        )
    else:
        for criterion, passed in decision.criteria.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {criterion}")
    return "\n".join(lines) + "\n"
