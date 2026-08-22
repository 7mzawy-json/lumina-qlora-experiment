import csv
import json
from pathlib import Path

import pytest

from lumina_experiment.contracts import Generation
from lumina_experiment.review import (
    build_blind_pairs,
    generations_from_scored_rows,
    is_private_raw_path,
    pair_conditions,
    write_review_packet,
)
from scripts.build_review_sheet import main as build_review_sheet_main

SCORED_CAPABILITIES = (
    "instruction",
    "reasoning",
    "knowledge",
    "summarization",
    "writing",
    "programming",
    "json",
)


def generation(case_id: str, condition: str, capability: str) -> Generation:
    version = "one" if condition == "base" else "two"
    return Generation(
        case_id=case_id,
        condition=condition,
        output=f"response {version} for {case_id}",
        evidence_state="smoke_test_verified",
        metadata={
            "capability": capability,
            "prompt": f"Prompt for {case_id}",
            "rubric": {
                "minimum": 0,
                "maximum": 2,
                "criteria": {"0": "incorrect", "1": "partial", "2": "complete"},
            },
        },
    )


def paired_generations(per_capability: int = 1) -> list[Generation]:
    records: list[Generation] = []
    for capability in SCORED_CAPABILITIES:
        for index in range(per_capability):
            case_id = f"{capability}-{index:03d}"
            records.extend(
                [
                    generation(case_id, "base", capability),
                    generation(case_id, "primary-r16", capability),
                ]
            )
    return records


def test_review_packet_conceals_condition_and_randomizes_side() -> None:
    packet = build_blind_pairs(paired_generations(), seed=42, sample_size=2)

    serialized = json.dumps(packet.to_public_dict())
    assert "condition" not in serialized
    assert "primary-r16" not in serialized
    assert {item.left_label for item in packet.items} == {"A"}
    assert {item.right_label for item in packet.items} == {"B"}
    assert set(packet.private_key) == {item.case_id for item in packet.items}


def test_review_packet_is_deterministic_for_same_seed() -> None:
    generations = paired_generations(per_capability=2)

    first = build_blind_pairs(generations, seed=7, sample_size=7)
    second = build_blind_pairs(reversed(generations), seed=7, sample_size=7)

    assert first == second


def test_default_packet_samples_six_from_each_scored_capability() -> None:
    generations = paired_generations(per_capability=8)
    generations.extend(
        [
            generation("safety-001", "base", "safety"),
            generation("safety-001", "primary-r16", "safety"),
        ]
    )

    packet = build_blind_pairs(generations)

    assert len(packet.items) == 42
    counts = {capability: 0 for capability in SCORED_CAPABILITIES}
    for item in packet.items:
        counts[item.capability] += 1
    assert counts == {capability: 6 for capability in SCORED_CAPABILITIES}


def test_default_packet_rejects_capability_with_fewer_than_six_pairs() -> None:
    generations: list[Generation] = []
    for capability in SCORED_CAPABILITIES:
        count = 5 if capability == "writing" else 7 if capability == "instruction" else 6
        for index in range(count):
            case_id = f"{capability}-{index:03d}"
            generations.extend(
                [
                    generation(case_id, "base", capability),
                    generation(case_id, "primary-r16", capability),
                ]
            )

    with pytest.raises(ValueError, match="six paired cases"):
        build_blind_pairs(generations)


def test_missing_pair_fails_closed() -> None:
    base = [generation("instruction-001", "base", "instruction")]
    adapter: list[Generation] = []

    with pytest.raises(ValueError, match="missing paired generation"):
        pair_conditions(base, adapter)


def test_duplicate_generation_fails_closed() -> None:
    base = [
        generation("instruction-001", "base", "instruction"),
        generation("instruction-001", "base", "instruction"),
    ]
    adapter = [generation("instruction-001", "primary-r16", "instruction")]

    with pytest.raises(ValueError, match="duplicate generation"):
        pair_conditions(base, adapter)


def test_pair_conditions_rejects_mislabeled_base_file() -> None:
    base = [generation("instruction-001", "candidate", "instruction")]
    adapter = [generation("instruction-001", "primary-r16", "instruction")]

    with pytest.raises(ValueError, match="base condition"):
        pair_conditions(base, adapter)


def test_review_packet_rejects_rubric_or_evidence_mismatch() -> None:
    base = generation("writing-001", "base", "writing")
    adapter = generation("writing-001", "primary-r16", "writing")
    mismatched_rubric = Generation(
        case_id=adapter.case_id,
        condition=adapter.condition,
        output=adapter.output,
        evidence_state=adapter.evidence_state,
        metadata={**adapter.metadata, "rubric": {"criteria": {"2": "different"}}},
    )
    mismatched_evidence = Generation(
        case_id=adapter.case_id,
        condition=adapter.condition,
        output=adapter.output,
        evidence_state="8b_gpu_measured",
        metadata=adapter.metadata,
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        build_blind_pairs([base, mismatched_rubric], sample_size=1)
    with pytest.raises(ValueError, match="evidence state mismatch"):
        build_blind_pairs([base, mismatched_evidence], sample_size=1)


def test_review_writer_keeps_private_key_under_ignored_raw_directory(tmp_path: Path) -> None:
    packet = build_blind_pairs(paired_generations(), seed=42, sample_size=2)
    public_path = tmp_path / "results" / "raw" / "review.csv"
    private_path = tmp_path / "results" / "raw" / "review-key.json"

    write_review_packet(packet, public_path, private_path)

    rows = list(csv.DictReader(public_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2
    assert set(rows[0]) == {
        "case_id",
        "capability",
        "prompt",
        "A",
        "B",
        "rubric",
        "winner",
        "reviewer_notes",
    }
    assert "condition" not in public_path.read_text(encoding="utf-8")
    assert json.loads(private_path.read_text(encoding="utf-8"))["assignments"]


def test_review_writer_rejects_public_private_key_path(tmp_path: Path) -> None:
    packet = build_blind_pairs(paired_generations(), seed=42, sample_size=2)

    with pytest.raises(ValueError, match="results/raw"):
        write_review_packet(
            packet,
            tmp_path / "results" / "summary" / "review.csv",
            tmp_path / "results" / "summary" / "review-key.json",
        )


def test_review_writer_rejects_trackable_sheet_with_raw_outputs(tmp_path: Path) -> None:
    packet = build_blind_pairs(paired_generations(), seed=42, sample_size=2)

    with pytest.raises(ValueError, match="review sheet.*results/raw"):
        write_review_packet(
            packet,
            tmp_path / "results" / "summary" / "review.csv",
            tmp_path / "results" / "raw" / "review-key.json",
        )


def test_review_writer_rejects_path_traversal_out_of_raw_directory(tmp_path: Path) -> None:
    packet = build_blind_pairs(paired_generations(), seed=42, sample_size=2)
    disguised_public_path = tmp_path / "results" / "raw" / ".." / "summary" / "review-key.json"

    with pytest.raises(ValueError, match="results/raw"):
        write_review_packet(
            packet,
            tmp_path / "results" / "summary" / "review.csv",
            disguised_public_path,
        )


def test_repo_nested_results_raw_directory_is_not_treated_as_ignored() -> None:
    assert is_private_raw_path(Path("nested/results/raw/review.csv")) is False
    assert is_private_raw_path(Path("results/raw/review.csv")) is True


def test_scored_rows_reconstruct_review_metadata_without_condition_leakage() -> None:
    rows = [
        {
            "case_id": "writing-001",
            "condition": condition,
            "capability": "writing",
            "prompt": "Draft a concise status update.",
            "output": f"response {version}",
            "evidence_state": "8b_gpu_measured",
            "rubric": {"minimum": 0, "maximum": 2, "criteria": {}},
            "scores": [],
        }
        for condition, version in (("base", "one"), ("primary-r16", "two"))
    ]

    generations = generations_from_scored_rows(rows)
    packet = build_blind_pairs(generations, sample_size=1)

    assert len(packet.items) == 1
    assert packet.items[0].prompt == "Draft a concise status update."
    assert "condition" not in json.dumps(packet.to_public_dict())


def test_build_review_sheet_cli_writes_public_sheet_and_private_key(tmp_path: Path) -> None:
    rows = [
        {
            "case_id": "writing-001",
            "condition": condition,
            "capability": "writing",
            "prompt": "Draft a concise status update.",
            "output": f"response {version}",
            "evidence_state": "8b_gpu_measured",
            "rubric": {"minimum": 0, "maximum": 2, "criteria": {}},
            "scores": [],
        }
        for condition, version in (("base", "one"), ("primary-r16", "two"))
    ]
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    public_path = tmp_path / "results" / "raw" / "review.csv"
    private_path = tmp_path / "results" / "raw" / "review-key.json"

    exit_code = build_review_sheet_main(
        [
            "--scores",
            str(scores_path),
            "--output",
            str(public_path),
            "--private-key",
            str(private_path),
            "--sample-size",
            "1",
        ]
    )

    assert exit_code == 0
    assert public_path.is_file()
    assert private_path.is_file()
    assert "condition" not in public_path.read_text(encoding="utf-8")
