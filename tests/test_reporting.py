import csv
import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from lumina_experiment.contracts import EvalCase, Generation, canonical_json
from lumina_experiment.reporting import (
    combine_condition_manifests,
    generation_artifact_hash,
    generation_hashes_from_scored_records,
    render_report,
    score_generation_records,
    summarize_scored_records,
    validate_evidence,
    verify_measured_scored_records,
)
from lumina_experiment.review import build_blind_pairs, generations_from_scored_rows
from lumina_experiment.scoring import load_eval_directory, select_balanced_cases
from lumina_experiment.statistics import Interval, ResultSummary
from scripts.score_results import main as score_results_main
from scripts.summarize_results import _manifest, _summary_payload
from scripts.summarize_results import main as summarize_results_main


def measured_summary() -> ResultSummary:
    cases = load_eval_directory(Path("data/eval/cases"))
    case_ids = tuple(scored_case_ids())
    diagnostic_ids = tuple(case.id for case in cases if case.capability == "safety")
    review_case_capabilities: dict[str, str] = {}
    for capability in (
        "instruction",
        "reasoning",
        "knowledge",
        "summarization",
        "writing",
        "programming",
        "json",
    ):
        capability_ids = sorted(case.id for case in cases if case.capability == capability)[:6]
        review_case_capabilities.update({case_id: capability for case_id in capability_ids})
    return ResultSummary(
        primary_deltas={
            "instruction_following": 16.7,
            "json_schema_validity": 3.3,
        },
        secondary_deltas={
            "reasoning": 1.2,
            "knowledge": -0.5,
            "summarization": 2.1,
            "programming": 0.0,
        },
        writing_adapter_wins=4,
        writing_base_wins=1,
        writing_ties=1,
        diagnostics=tuple(
            {"case_id": case_id, "serious_new_failure": False} for case_id in diagnostic_ids
        ),
        evidence_state="8b_gpu_measured",
        intervals={
            "instruction_following": Interval(16.7, 11.0, 22.0, 10_000),
            "json_schema_validity": Interval(3.3, 0.5, 6.0, 10_000),
            "reasoning": Interval(1.2, -1.0, 3.0, 10_000),
            "knowledge": Interval(-0.5, -2.0, 1.0, 10_000),
            "summarization": Interval(2.1, 0.0, 4.0, 10_000),
            "programming": Interval(0.0, 0.0, 0.0, 10_000),
        },
        scored_case_ids={"base": case_ids, "adapter": case_ids},
        scored_case_counts={"base": 210, "adapter": 210},
        review_case_ids=tuple(review_case_capabilities),
        review_capability_counts={
            "instruction": 6,
            "reasoning": 6,
            "knowledge": 6,
            "summarization": 6,
            "writing": 6,
            "programming": 6,
            "json": 6,
        },
        review_case_capabilities=review_case_capabilities,
        diagnostic_case_ids=diagnostic_ids,
        diagnostic_generation_case_ids={
            "base": diagnostic_ids,
            "adapter": diagnostic_ids,
        },
        adapter_condition="primary-r16",
        generation_hashes={"base": "b" * 64, "adapter": "c" * 64},
        score_hash="f" * 64,
    )


def pilot_cases() -> list[EvalCase]:
    return select_balanced_cases(
        load_eval_directory(Path("data/eval/cases")),
        limit=24,
        seed=42,
    )


def pilot_summary() -> ResultSummary:
    cases = pilot_cases()
    scored_ids = tuple(sorted(case.id for case in cases if case.capability != "safety"))
    diagnostic_ids = tuple(sorted(case.id for case in cases if case.capability == "safety"))
    review_capabilities = {
        case.id: case.capability for case in cases if case.capability != "safety"
    }
    intervals = {
        name: Interval(0.0, 0.0, 0.0, 10_000)
        for name in (
            "instruction_following",
            "json_schema_validity",
            "reasoning",
            "knowledge",
            "summarization",
            "programming",
        )
    }
    return ResultSummary(
        primary_deltas={
            "instruction_following": 0.0,
            "json_schema_validity": 0.0,
        },
        secondary_deltas={
            "reasoning": 0.0,
            "knowledge": 0.0,
            "summarization": 0.0,
            "programming": 0.0,
        },
        writing_adapter_wins=0,
        writing_base_wins=0,
        writing_ties=3,
        diagnostics=tuple(
            {"case_id": case_id, "serious_new_failure": False} for case_id in diagnostic_ids
        ),
        evidence_state="8b_pilot_measured",
        intervals=intervals,
        scored_case_ids={"base": scored_ids, "adapter": scored_ids},
        scored_case_counts={"base": 21, "adapter": 21},
        review_case_ids=scored_ids,
        review_capability_counts=dict(Counter(review_capabilities.values())),
        review_case_capabilities=review_capabilities,
        diagnostic_case_ids=diagnostic_ids,
        diagnostic_generation_case_ids={
            "base": diagnostic_ids,
            "adapter": diagnostic_ids,
        },
        adapter_condition="pilot-r16",
        generation_hashes={"base": "b" * 64, "adapter": "c" * 64},
        score_hash="f" * 64,
    )


def valid_pilot_manifest() -> dict[str, object]:
    summary = pilot_summary()
    case_ids = [case.id for case in pilot_cases()]
    freeze = Path("data/eval/FROZEN.sha256").read_text(encoding="utf-8").strip()
    model_revision = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    return {
        "evidence_state": "8b_pilot_measured",
        "evaluation_hash": freeze,
        "conditions": {
            "base": {
                "condition": "base",
                "case_ids": case_ids,
                "inference_config_hash": "d" * 64,
                "generation_hash": "b" * 64,
            },
            "adapter": {
                "condition": "pilot-r16",
                "case_ids": case_ids,
                "inference_config_hash": "d" * 64,
                "generation_hash": "c" * 64,
            },
        },
        "run_manifest": {
            "gpu": {
                "name": "Tesla T4",
                "vram_gb": 14.5,
                "bf16_supported": False,
                "torch_version": "2.11.0+cu128",
                "cuda_version": "12.8",
            },
            "runtime_seconds": 90.0,
            "peak_vram_gb": 10.0,
            "model_revision": model_revision,
            "package_versions": {"torch": "2.11.0+cu128", "transformers": "5.14.1"},
            "adapter_hash": "e" * 64,
            "score_hash": summary.score_hash,
            "generation_hashes": dict(summary.generation_hashes),
        },
        "training_manifest": {
            "run_id": "pilot-r16-1",
            "evidence_state": "8b_pilot_measured",
            "model_id": "Qwen/Qwen3-8B-Base",
            "model_revision": model_revision,
            "config_hash": "c0e0ff5942313418aa19d4db28430e5e52860ee54db9303b99dd990584677bdb",
            "dataset_hash": "18a9078a3c6ba5084b38c746f280b6a19d7645a1d87051a45f504e0a70b58809",
            "evaluation_hash": freeze,
            "selected_checkpoint": "artifacts/checkpoints/pilot-r16/checkpoint-16",
            "validation_selection": {"validation_loss": 1.0},
            "artifacts": {
                "adapter_sha256": "e" * 64,
                "runtime_seconds": "60.0",
                "peak_vram_gb": "10.0",
            },
        },
    }


def scored_case_ids() -> list[str]:
    return [
        case.id
        for case in load_eval_directory(Path("data/eval/cases"))
        if case.capability != "safety"
    ]


def valid_manifest() -> dict[str, object]:
    freeze = Path("data/eval/FROZEN.sha256").read_text(encoding="utf-8").strip()
    case_ids = scored_case_ids()
    return {
        "evidence_state": "8b_gpu_measured",
        "evaluation_hash": freeze,
        "conditions": {
            "base": {
                "condition": "base",
                "case_ids": case_ids,
                "inference_config_hash": "d" * 64,
                "generation_hash": "b" * 64,
            },
            "adapter": {
                "condition": "primary-r16",
                "case_ids": case_ids,
                "inference_config_hash": "d" * 64,
                "generation_hash": "c" * 64,
            },
        },
        "run_manifest": {
            "gpu": {
                "name": "NVIDIA A100-SXM4-40GB",
                "vram_gb": 40.0,
                "bf16_supported": True,
                "torch_version": "2.x",
                "cuda_version": "12.x",
            },
            "runtime_seconds": 7200.0,
            "peak_vram_gb": 31.4,
            "model_revision": "a" * 40,
            "package_versions": {"torch": "2.x", "transformers": "4.x"},
            "adapter_hash": "e" * 64,
            "score_hash": "f" * 64,
            "generation_hashes": {
                "base": "b" * 64,
                "adapter": "c" * 64,
            },
        },
    }


def condition_manifests() -> tuple[dict[str, object], dict[str, object]]:
    combined = valid_manifest()
    shared = {
        "evidence_state": "8b_gpu_measured",
        "evaluation_hash": combined["evaluation_hash"],
        "gpu": combined["run_manifest"]["gpu"],
        "peak_vram_gb": combined["run_manifest"]["peak_vram_gb"],
        "model_revision": combined["run_manifest"]["model_revision"],
        "package_versions": combined["run_manifest"]["package_versions"],
    }
    base = {
        **shared,
        **combined["conditions"]["base"],
        "runtime_seconds": 1200.0,
    }
    adapter = {
        **shared,
        **combined["conditions"]["adapter"],
        "runtime_seconds": 6000.0,
        "adapter_hash": combined["run_manifest"]["adapter_hash"],
    }
    return base, adapter


def test_combines_task_8_condition_manifests_into_validated_evidence() -> None:
    summary = measured_summary()
    base, adapter = condition_manifests()

    manifest = combine_condition_manifests(base, adapter, summary)

    assert manifest["run_manifest"]["runtime_seconds"] == 7200.0
    assert manifest["run_manifest"]["score_hash"] == summary.score_hash
    validate_evidence(summary, manifest)


def test_measured_manifest_is_rederived_from_condition_manifests(tmp_path: Path) -> None:
    summary = measured_summary()
    base, adapter = condition_manifests()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "base-manifest.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "primary-r16-manifest.json").write_text(json.dumps(adapter), encoding="utf-8")
    drifted = combine_condition_manifests(base, adapter, summary)
    drifted["run_manifest"]["gpu"]["name"] = "different GPU"
    (tmp_path / "evidence-manifest.json").write_text(json.dumps(drifted), encoding="utf-8")

    with pytest.raises(ValueError, match="derived evidence manifest"):
        _manifest(None, tmp_path, summary)


def test_measured_report_requires_all_210_paired_case_ids() -> None:
    manifest = valid_manifest()
    manifest["conditions"]["adapter"]["case_ids"] = scored_case_ids()[:-1]

    with pytest.raises(ValueError, match="all 210 scored case IDs"):
        validate_evidence(measured_summary(), manifest)


def test_measured_report_rejects_evaluation_hash_mismatch() -> None:
    manifest = valid_manifest()
    manifest["evaluation_hash"] = "wrong"

    with pytest.raises(ValueError, match="evaluation hash"):
        validate_evidence(measured_summary(), manifest)


def test_measured_report_rejects_inference_mismatch() -> None:
    manifest = valid_manifest()
    manifest["conditions"]["adapter"]["inference_config_hash"] = "f" * 64

    with pytest.raises(ValueError, match="inference configuration hashes"):
        validate_evidence(measured_summary(), manifest)


def test_measured_report_requires_complete_run_metadata() -> None:
    manifest = valid_manifest()
    del manifest["run_manifest"]["peak_vram_gb"]

    with pytest.raises(ValueError, match="peak_vram_gb"):
        validate_evidence(measured_summary(), manifest)


def test_measured_report_rejects_generation_hash_mismatch() -> None:
    manifest = valid_manifest()
    manifest["run_manifest"]["generation_hashes"]["adapter"] = "f" * 64

    with pytest.raises(ValueError, match="generation hashes"):
        validate_evidence(measured_summary(), manifest)


def test_measured_report_rejects_summary_missing_actual_scored_id() -> None:
    summary = measured_summary()
    incomplete = replace(
        summary,
        scored_case_ids={
            "base": summary.scored_case_ids["base"],
            "adapter": summary.scored_case_ids["adapter"][:-1],
        },
        scored_case_counts={"base": 210, "adapter": 209},
    )

    with pytest.raises(ValueError, match="summary.*210 scored case IDs"):
        validate_evidence(incomplete, valid_manifest())


def test_measured_report_rejects_missing_safety_generation() -> None:
    summary = measured_summary()
    incomplete = replace(
        summary,
        diagnostic_generation_case_ids={
            "base": summary.diagnostic_generation_case_ids["base"],
            "adapter": summary.diagnostic_generation_case_ids["adapter"][:-1],
        },
    )

    with pytest.raises(ValueError, match="20 frozen safety generations"):
        validate_evidence(incomplete, valid_manifest())


def test_measured_report_rejects_score_hash_mismatch() -> None:
    manifest = valid_manifest()
    manifest["run_manifest"]["score_hash"] = "0" * 64

    with pytest.raises(ValueError, match="score hash"):
        validate_evidence(measured_summary(), manifest)


def test_measured_report_requires_ten_thousand_resamples() -> None:
    summary = measured_summary()
    intervals = dict(summary.intervals)
    intervals["reasoning"] = replace(intervals["reasoning"], resamples=999)

    with pytest.raises(ValueError, match="10,000"):
        validate_evidence(replace(summary, intervals=intervals), valid_manifest())


def test_measured_report_rejects_non_finite_runtime() -> None:
    manifest = valid_manifest()
    manifest["run_manifest"]["runtime_seconds"] = "nan"

    with pytest.raises(ValueError, match="runtime_seconds"):
        validate_evidence(measured_summary(), manifest)


def test_measured_report_requires_typed_gpu_metadata() -> None:
    manifest = valid_manifest()
    manifest["run_manifest"]["gpu"]["bf16_supported"] = "yes"

    with pytest.raises(ValueError, match="bf16_supported"):
        validate_evidence(measured_summary(), manifest)


def test_manifest_cannot_redirect_trusted_evaluation_assets(tmp_path: Path) -> None:
    fake_freeze = tmp_path / "FROZEN.sha256"
    fake_freeze.write_text("0" * 64, encoding="utf-8")
    manifest = valid_manifest()
    manifest["freeze_file"] = str(fake_freeze)
    manifest["cases_dir"] = str(tmp_path)
    manifest["evaluation_hash"] = "0" * 64

    with pytest.raises(ValueError, match="evaluation hash"):
        validate_evidence(measured_summary(), manifest)


def test_report_separates_primary_and_secondary_metrics() -> None:
    report = render_report(measured_summary(), valid_manifest())

    assert "## Primary metrics" in report
    assert "## Secondary metrics" in report
    assert "instruction_following" in report
    assert "reasoning" in report
    assert "95% paired bootstrap CI" in report
    assert "weighted aggregate" not in report.casefold()


def test_measured_report_rejects_duplicate_case_ids() -> None:
    manifest = valid_manifest()
    manifest["conditions"]["adapter"]["case_ids"].append(scored_case_ids()[0])

    with pytest.raises(ValueError, match="all 210 scored case IDs"):
        validate_evidence(measured_summary(), manifest)


def test_measured_report_requires_sha256_artifact_hashes() -> None:
    manifest = valid_manifest()
    manifest["run_manifest"]["adapter_hash"] = "not-a-sha256"

    with pytest.raises(ValueError, match="adapter_hash"):
        validate_evidence(measured_summary(), manifest)


def test_non_gpu_report_is_labeled_without_claiming_measurement() -> None:
    summary = ResultSummary(
        primary_deltas={},
        secondary_deltas={},
        writing_adapter_wins=0,
        writing_base_wins=0,
        writing_ties=0,
        diagnostics=(),
        evidence_state="locally_verified",
    )

    report = render_report(summary, {"evidence_state": "locally_verified"})

    assert "locally_verified" in report
    assert "No validated 8B GPU measurement is claimed" in report


def test_directional_8b_pilot_report_disclaims_full_benchmark_completion() -> None:
    summary = pilot_summary()

    report = render_report(summary, valid_pilot_manifest())

    assert "Directional 8B pilot measurement" in report
    assert "does not claim completion of the full frozen benchmark" in report
    assert "Decision: **NOT EVALUATED**" in report
    assert "Decision: **FAIL**" not in report


def test_directional_pilot_metrics_payload_records_gate_as_not_evaluated() -> None:
    payload = _summary_payload(pilot_summary())

    assert payload["decision"] == {
        "status": "not_evaluated",
        "passed": None,
        "criteria": {},
        "reasons": ["bounded_8b_pilot"],
    }


def test_directional_pilot_evidence_rejects_training_manifest_drift() -> None:
    manifest = valid_pilot_manifest()

    validate_evidence(pilot_summary(), manifest)
    drifted = deepcopy(manifest)
    drifted["training_manifest"]["dataset_hash"] = "0" * 64

    with pytest.raises(ValueError, match="training dataset hash"):
        validate_evidence(pilot_summary(), drifted)


def test_directional_pilot_manifest_is_derived_from_all_source_manifests(
    tmp_path: Path,
) -> None:
    summary = pilot_summary()
    manifest = valid_pilot_manifest()
    conditions = manifest["conditions"]
    shared = {
        "evidence_state": manifest["evidence_state"],
        "evaluation_hash": manifest["evaluation_hash"],
        "gpu": manifest["run_manifest"]["gpu"],
        "peak_vram_gb": manifest["run_manifest"]["peak_vram_gb"],
        "model_revision": manifest["run_manifest"]["model_revision"],
        "package_versions": manifest["run_manifest"]["package_versions"],
    }
    base = {**shared, **conditions["base"], "runtime_seconds": 30.0}
    adapter = {
        **shared,
        **conditions["adapter"],
        "runtime_seconds": 60.0,
        "adapter_hash": manifest["run_manifest"]["adapter_hash"],
    }
    (tmp_path / "base-manifest.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "pilot-r16-manifest.json").write_text(json.dumps(adapter), encoding="utf-8")
    (tmp_path / "run-manifest.json").write_text(
        json.dumps(manifest["training_manifest"]), encoding="utf-8"
    )

    derived = _manifest(None, tmp_path, summary)

    assert derived["training_manifest"] == manifest["training_manifest"]
    assert canonical_json(derived["conditions"]) == canonical_json(conditions)


def test_directional_pilot_review_is_bound_to_exact_21_case_packet() -> None:
    cases = pilot_cases()
    generations = [
        Generation(case.id, condition, "fixture response", "8b_pilot_measured")
        for condition in ("base", "pilot-r16")
        for case in cases
    ]
    rows = score_generation_records(cases, generations)
    packet = build_blind_pairs(generations_from_scored_rows(rows), seed=42, sample_size=21)
    reviews = [item.to_public_dict() for item in packet.items]
    for review in reviews:
        review["winner"] = "TIE"
    reviews[0]["A"] = "tampered output"
    diagnostics = [
        {"case_id": case.id, "serious_new_failure": False}
        for case in cases
        if case.capability == "safety"
    ]

    with pytest.raises(ValueError, match="concealed review packet mismatch"):
        summarize_scored_records(
            rows,
            reviews,
            packet.private_key,
            evidence_state="8b_pilot_measured",
            diagnostics=diagnostics,
            review_seed=42,
        )


def test_score_generation_records_preserves_auditable_context() -> None:
    case = EvalCase(
        id="reasoning-001",
        capability="reasoning",
        prompt="Return only 4.",
        scorer="exact_match",
        expected={"answers": ["4"]},
    )
    generations = [
        Generation(
            case_id=case.id,
            condition="base",
            output="3",
            evidence_state="smoke_test_verified",
        ),
        Generation(
            case_id=case.id,
            condition="primary-r16",
            output="4",
            evidence_state="smoke_test_verified",
        ),
    ]

    rows = score_generation_records([case], generations)

    assert len(rows) == 2
    assert rows[0]["prompt"] == case.prompt
    assert rows[0]["output"] == "3"
    assert rows[0]["scores"][0]["metric"] == "exact_match"
    assert rows[1]["scores"][0]["value"] == 1.0
    assert generation_hashes_from_scored_records(rows, "primary-r16") == {
        "base": generation_artifact_hash([generations[0]]),
        "adapter": generation_artifact_hash([generations[1]]),
    }


def test_measured_scores_are_recomputed_from_frozen_cases_and_outputs() -> None:
    case = next(
        case
        for case in load_eval_directory(Path("data/eval/cases"))
        if case.capability == "instruction"
    )
    generations = [
        Generation(case.id, condition, "", "8b_gpu_measured")
        for condition in ("base", "primary-r16")
    ]
    rows = score_generation_records([case], generations)
    rows[1]["scores"][0]["value"] = 1.0
    rows[1]["scores"][0]["passed"] = True

    with pytest.raises(ValueError, match="recomputed frozen-case scores"):
        verify_measured_scored_records(rows)


def test_measured_review_is_bound_to_exact_seed_42_packet() -> None:
    cases = load_eval_directory(Path("data/eval/cases"))
    generations = [
        Generation(case.id, condition, "fixture response", "8b_gpu_measured")
        for condition in ("base", "primary-r16")
        for case in cases
    ]
    rows = score_generation_records(cases, generations)
    packet = build_blind_pairs(generations_from_scored_rows(rows), seed=42, sample_size=42)
    reviews = [item.to_public_dict() for item in packet.items]
    for review in reviews:
        review["winner"] = "TIE"
    reviews[0]["A"] = "tampered output"
    diagnostics = [
        {"case_id": case.id, "serious_new_failure": False}
        for case in cases
        if case.capability == "safety"
    ]

    with pytest.raises(ValueError, match="concealed review packet mismatch"):
        summarize_scored_records(
            rows,
            reviews,
            packet.private_key,
            evidence_state="8b_gpu_measured",
            diagnostics=diagnostics,
            review_seed=42,
        )

    reviews[0]["A"] = packet.items[0].left_output
    reviews[0]["condition"] = "base"
    with pytest.raises(ValueError, match="exact concealed columns"):
        summarize_scored_records(
            rows,
            reviews,
            packet.private_key,
            evidence_state="8b_gpu_measured",
            diagnostics=diagnostics,
            review_seed=42,
        )


def scored_row(
    case_id: str, condition: str, capability: str, metric: str, value: float
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "condition": condition,
        "capability": capability,
        "prompt": "fixture prompt",
        "output": "fixture output",
        "evidence_state": "smoke_test_verified",
        "rubric": {},
        "scores": [{"metric": metric, "value": value, "passed": value >= 0.5}],
    }


def test_summary_uses_paired_metrics_and_unblinded_writing_winners() -> None:
    rows = [
        scored_row("instruction-001", "base", "instruction", "constraint_adherence", 0),
        scored_row("instruction-001", "primary-r16", "instruction", "constraint_adherence", 1),
        scored_row("json-001", "base", "json", "json_schema_valid", 0),
        scored_row("json-001", "primary-r16", "json", "json_schema_valid", 1),
        scored_row("reasoning-001", "base", "reasoning", "exact_match", 1),
        scored_row("reasoning-001", "primary-r16", "reasoning", "exact_match", 1),
    ]
    reviews = [
        {"case_id": "writing-001", "capability": "writing", "winner": "A"},
        {"case_id": "writing-002", "capability": "writing", "winner": "A"},
        {"case_id": "writing-003", "capability": "writing", "winner": "B"},
    ]
    private_key = {
        "writing-001": {"A": "primary-r16", "B": "base"},
        "writing-002": {"A": "primary-r16", "B": "base"},
        "writing-003": {"A": "primary-r16", "B": "base"},
    }

    summary = summarize_scored_records(
        rows,
        reviews,
        private_key,
        evidence_state="smoke_test_verified",
        resamples=100,
    )

    assert summary.primary_deltas == {
        "instruction_following": 100.0,
        "json_schema_validity": 100.0,
    }
    assert summary.secondary_deltas == {"reasoning": 0.0}
    assert summary.writing_adapter_wins == 2
    assert summary.writing_base_wins == 1
    assert summary.writing_ties == 0
    assert set(summary.intervals) == {
        "instruction_following",
        "json_schema_validity",
        "reasoning",
    }


def test_summary_rejects_private_key_with_unknown_condition() -> None:
    rows = [
        scored_row("writing-001", "base", "writing", "rubric_score", 0),
        scored_row("writing-001", "primary-r16", "writing", "rubric_score", 1),
    ]
    reviews = [{"case_id": "writing-001", "capability": "writing", "winner": "A"}]
    private_key = {"writing-001": {"A": "different-adapter", "B": "base"}}

    with pytest.raises(ValueError, match="base and primary-r16"):
        summarize_scored_records(
            rows,
            reviews,
            private_key,
            evidence_state="smoke_test_verified",
            resamples=10,
        )


def test_summary_rejects_raw_outputs_in_diagnostic_annotations() -> None:
    rows = [
        scored_row("writing-001", "base", "writing", "rubric_score", 0),
        scored_row("writing-001", "primary-r16", "writing", "rubric_score", 1),
    ]
    reviews = [{"case_id": "writing-001", "capability": "writing", "winner": "A"}]
    private_key = {"writing-001": {"A": "primary-r16", "B": "base"}}

    with pytest.raises(ValueError, match="unsupported fields"):
        summarize_scored_records(
            rows,
            reviews,
            private_key,
            evidence_state="smoke_test_verified",
            diagnostics=[
                {
                    "case_id": "safety-001",
                    "serious_new_failure": False,
                    "output": "raw model response",
                }
            ],
            resamples=10,
        )


def test_summary_rejects_missing_paired_score() -> None:
    rows = [
        scored_row("instruction-001", "base", "instruction", "constraint_adherence", 0),
        scored_row("json-001", "base", "json", "json_schema_valid", 0),
        scored_row("json-001", "primary-r16", "json", "json_schema_valid", 1),
    ]
    for row in rows:
        row["evidence_state"] = "smoke_test_verified"

    with pytest.raises(ValueError, match="missing paired score"):
        summarize_scored_records(
            rows,
            [],
            {},
            evidence_state="smoke_test_verified",
            resamples=10,
        )


def test_score_results_cli_writes_paired_auditable_jsonl(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "reasoning.jsonl").write_text(
        json.dumps(
            {
                "id": "reasoning-001",
                "capability": "reasoning",
                "prompt": "Return only 4.",
                "scorer": "exact_match",
                "expected": {"answers": ["4"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    base_dir.mkdir()
    adapter_dir.mkdir()
    for path, condition, output in (
        (base_dir, "base", "3"),
        (adapter_dir, "primary-r16", "4"),
    ):
        (path / "generations.jsonl").write_text(
            json.dumps(
                {
                    "case_id": "reasoning-001",
                    "condition": condition,
                    "output": output,
                    "evidence_state": "smoke_test_verified",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    output_path = tmp_path / "results" / "raw" / "scores.jsonl"

    exit_code = score_results_main(
        [
            "--base",
            str(base_dir),
            "--adapter",
            str(adapter_dir),
            "--cases",
            str(cases_dir),
            "--output",
            str(output_path),
        ]
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert len(rows) == 2
    assert {row["condition"] for row in rows} == {"base", "primary-r16"}
    assert rows[1]["scores"][0]["value"] == 1.0


def test_score_results_cli_rejects_trackable_output_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="results/raw"):
        score_results_main(
            [
                "--base",
                str(tmp_path / "base"),
                "--adapter",
                str(tmp_path / "adapter"),
                "--cases",
                str(tmp_path / "cases"),
                "--output",
                str(tmp_path / "results" / "summary" / "scores.jsonl"),
            ]
        )


def test_score_results_cli_rejects_repo_nested_results_raw_path() -> None:
    with pytest.raises(ValueError, match="results/raw"):
        score_results_main(
            [
                "--base",
                "missing-base",
                "--adapter",
                "missing-adapter",
                "--output",
                "nested/results/raw/scores.jsonl",
            ]
        )


def test_summarize_results_cli_writes_metrics_and_report(tmp_path: Path) -> None:
    rows = [
        scored_row("instruction-001", "base", "instruction", "constraint_adherence", 0),
        scored_row("instruction-001", "primary-r16", "instruction", "constraint_adherence", 1),
        scored_row("json-001", "base", "json", "json_schema_valid", 0),
        scored_row("json-001", "primary-r16", "json", "json_schema_valid", 1),
    ]
    for row in rows:
        row["evidence_state"] = "smoke_test_verified"
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    review_path = tmp_path / "review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "capability", "winner"])
        writer.writeheader()
        writer.writerow({"case_id": "writing-001", "capability": "writing", "winner": "A"})
    private_key = tmp_path / "results" / "raw" / "review-key.json"
    private_key.parent.mkdir(parents=True)
    private_key.write_text(
        json.dumps({"assignments": {"writing-001": {"A": "primary-r16", "B": "base"}}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "summary"

    exit_code = summarize_results_main(
        [
            "--scores",
            str(scores_path),
            "--review",
            str(review_path),
            "--private-key",
            str(private_key),
            "--output",
            str(output_dir),
            "--resamples",
            "10",
        ]
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert metrics["evidence_state"] == "smoke_test_verified"
    assert metrics["decision"]["passed"] is False
    assert "## Primary metrics" in report

    (output_dir / "metrics.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metrics.csv"):
        summarize_results_main(["--verify-only", "--output", str(output_dir)])


def test_directional_pilot_verify_only_recomputes_every_source_artifact(
    tmp_path: Path,
) -> None:
    cases = pilot_cases()
    generations = [
        Generation(case.id, condition, "fixture response", "8b_pilot_measured")
        for condition in ("base", "pilot-r16")
        for case in cases
    ]
    rows = score_generation_records(cases, generations)
    packet = build_blind_pairs(generations_from_scored_rows(rows), seed=42, sample_size=21)
    reviews = [item.to_public_dict() for item in packet.items]
    for review in reviews:
        review["winner"] = "TIE"
    diagnostics = [
        {"case_id": case.id, "serious_new_failure": False}
        for case in cases
        if case.capability == "safety"
    ]
    summary = summarize_scored_records(
        rows,
        reviews,
        packet.private_key,
        evidence_state="8b_pilot_measured",
        diagnostics=diagnostics,
        review_seed=42,
    )
    manifest = valid_pilot_manifest()
    manifest["conditions"]["base"]["generation_hash"] = summary.generation_hashes["base"]
    manifest["conditions"]["adapter"]["generation_hash"] = summary.generation_hashes["adapter"]
    manifest["run_manifest"]["generation_hashes"] = dict(summary.generation_hashes)
    manifest["run_manifest"]["score_hash"] = summary.score_hash

    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text("".join(f"{canonical_json(row)}\n" for row in rows), encoding="utf-8")
    review_path = tmp_path / "review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "capability",
            "prompt",
            "A",
            "B",
            "rubric",
            "winner",
            "reviewer_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({**review, "rubric": canonical_json(review["rubric"])})
    private_key = tmp_path / "results" / "raw" / "review-key.json"
    private_key.parent.mkdir(parents=True)
    private_key.write_text(canonical_json(packet.to_private_dict()), encoding="utf-8")
    diagnostics_path = tmp_path / "diagnostics.jsonl"
    diagnostics_path.write_text(
        "".join(f"{canonical_json(row)}\n" for row in diagnostics), encoding="utf-8"
    )
    output_dir = tmp_path / "summary"
    output_dir.mkdir()
    conditions = manifest["conditions"]
    shared = {
        "evidence_state": manifest["evidence_state"],
        "evaluation_hash": manifest["evaluation_hash"],
        "gpu": manifest["run_manifest"]["gpu"],
        "peak_vram_gb": manifest["run_manifest"]["peak_vram_gb"],
        "model_revision": manifest["run_manifest"]["model_revision"],
        "package_versions": manifest["run_manifest"]["package_versions"],
    }
    (output_dir / "base-manifest.json").write_text(
        canonical_json({**shared, **conditions["base"], "runtime_seconds": 30.0}),
        encoding="utf-8",
    )
    (output_dir / "pilot-r16-manifest.json").write_text(
        canonical_json(
            {
                **shared,
                **conditions["adapter"],
                "runtime_seconds": 60.0,
                "adapter_hash": manifest["run_manifest"]["adapter_hash"],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "run-manifest.json").write_text(
        canonical_json(manifest["training_manifest"]), encoding="utf-8"
    )
    arguments = [
        "--scores",
        str(scores_path),
        "--review",
        str(review_path),
        "--private-key",
        str(private_key),
        "--diagnostics",
        str(diagnostics_path),
        "--output",
        str(output_dir),
    ]

    assert summarize_results_main(arguments) == 0
    assert summarize_results_main([*arguments, "--verify-only"]) == 0

    metrics_path = output_dir / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_payload["decision"] = {
        "status": "pass",
        "passed": True,
        "criteria": {"full_benchmark": True},
        "reasons": [],
    }
    metrics_path.write_text(canonical_json(metrics_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics.json payload"):
        summarize_results_main([*arguments, "--verify-only"])
    metrics_path.write_text(canonical_json(_summary_payload(summary)), encoding="utf-8")

    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0] = deepcopy(tampered_rows[0])
    tampered_rows[0]["scores"][0]["value"] = 1.0
    scores_path.write_text(
        "".join(f"{canonical_json(row)}\n" for row in tampered_rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="recomputed frozen-case scores"):
        summarize_results_main([*arguments, "--verify-only"])
