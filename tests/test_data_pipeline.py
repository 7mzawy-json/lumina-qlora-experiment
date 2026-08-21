from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from lumina_experiment.contracts import InstructionRecord, Message
from lumina_experiment.data_pipeline import (
    DataPreparationConfig,
    extract_oasst_pairs,
    filter_records,
    prepare_dataset,
    validate_authored_records,
)
from scripts.prepare_data import main as prepare_main


@pytest.fixture
def oasst_rows() -> list[dict[str, object]]:
    return json.loads(Path("tests/fixtures/oasst_rows.json").read_text(encoding="utf-8"))


def valid_record(
    *,
    record_id: str = "valid",
    source: str = "test-source",
    license_name: str = "Apache-2.0",
    capability: str = "general",
    prompt: str = "Explain gravity.",
    answer: str = "Gravity attracts objects with mass.",
    metadata: dict[str, object] | None = None,
) -> InstructionRecord:
    return InstructionRecord(
        id=record_id,
        source=source,
        license=license_name,
        capability=capability,
        cluster_id=f"cluster-{record_id}",
        messages=(Message("user", prompt), Message("assistant", answer)),
        metadata=metadata or {"language": "en"},
    )


def test_extracts_english_root_to_highest_ranked_assistant_pair(
    oasst_rows: list[dict[str, object]],
) -> None:
    records = extract_oasst_pairs(oasst_rows)

    solar = next(record for record in records if record.cluster_id == "tree-001")
    assert solar.messages == (
        Message(role="user", content="Give two uses of solar energy."),
        Message(
            role="assistant",
            content="Solar power can generate electricity and heat water.",
        ),
    )
    assert solar.metadata["row_ids"] == ["root-solar", "assistant-solar-a"]


def test_extraction_excludes_non_english_and_deleted_replies(
    oasst_rows: list[dict[str, object]],
) -> None:
    records = extract_oasst_pairs(oasst_rows)

    assert [record.cluster_id for record in records] == ["tree-001", "tree-002", "tree-003"]
    code = next(record for record in records if record.cluster_id == "tree-003")
    assert code.messages[-1].content.startswith("def add")


def test_rejection_log_records_over_length_reason() -> None:
    accepted, rejected = filter_records(
        [valid_record(), valid_record(record_id="bad", answer="x" * 12_000)],
        max_characters=12_000,
    )

    assert [record.id for record in accepted] == ["valid"]
    assert [(item.id, item.reason) for item in rejected] == [("bad", "over_length")]


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (valid_record(record_id="language", metadata={"language": "tr"}), "non_english"),
        (
            valid_record(record_id="spam", metadata={"language": "en", "spam": True}),
            "spam_or_deleted",
        ),
        (
            valid_record(record_id="credential", answer=f"Use token {'ghp_'}{'a' * 24}"),
            "credential_pattern",
        ),
    ],
)
def test_quality_filters_record_enumerated_reason(record: InstructionRecord, reason: str) -> None:
    accepted, rejected = filter_records([record], max_characters=12_000)

    assert accepted == []
    assert [(item.id, item.source, item.reason) for item in rejected] == [
        (record.id, record.source, reason)
    ]


def test_exact_duplicate_pair_is_rejected() -> None:
    accepted, rejected = filter_records(
        [valid_record(record_id="first"), valid_record(record_id="second")],
        max_characters=12_000,
    )

    assert [record.id for record in accepted] == ["first"]
    assert [(item.id, item.reason) for item in rejected] == [("second", "duplicate_pair")]


def test_prepare_dataset_honors_exact_mixture(
    oasst_rows: list[dict[str, object]],
) -> None:
    authored = [
        valid_record(
            record_id="authored-001",
            source="lumina-demonstration-authored",
            license_name="CC-BY-4.0",
            capability="json",
            prompt=('Return JSON with exactly "sku", "quantity", and "reorder".'),
            answer='{"sku":"ITEM-001","quantity":1,"reorder":false}',
            metadata={
                "language": "en",
                "provenance": "lumina-demonstration-authored",
                "authorship": "ai_assisted",
                "review_status": "automated_checks_passed_pending_human",
                "family": "inventory",
            },
        )
    ]
    config = DataPreparationConfig.for_test(
        total_records=3,
        mixture={"general_oasst1": 1, "capability_oasst1": 1, "authored": 1},
    )

    first = prepare_dataset(oasst_rows, authored, config)
    second = prepare_dataset(list(reversed(oasst_rows)), authored, config)

    assert [record.id for record in first] == [record.id for record in second]
    assert Counter(record.source for record in first) == {
        "OpenAssistant/oasst1": 2,
        "lumina-demonstration-authored": 1,
    }


def test_authored_corpus_has_exact_transparent_composition() -> None:
    rows = [
        json.loads(line)
        for line in Path("data/authored/training.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 150
    assert len({row["id"] for row in rows}) == 150
    assert Counter(row["capability"] for row in rows) == {"json": 75, "instruction": 75}
    assert {row["source"] for row in rows} == {"lumina-demonstration-authored"}
    assert {row["license"] for row in rows} == {"CC-BY-4.0"}
    assert {row["metadata"]["language"] for row in rows} == {"en"}
    assert {row["metadata"]["authorship"] for row in rows} == {"ai_assisted"}
    assert {row["metadata"]["review_status"] for row in rows} == {
        "automated_checks_passed_pending_human"
    }

    validated = validate_authored_records(rows)
    assert len(validated) == 150


def test_fixture_only_cli_produces_stable_hashes(
    tmp_path: Path, oasst_rows: list[dict[str, object]]
) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: OpenAssistant/oasst1",
                "dataset_revision: fixture-revision",
                "language: en",
                "seed: 42",
                "total_records: 3",
                "train_records: 2",
                "validation_records: 1",
                "mixture:",
                "  general_oasst1: 1",
                "  capability_oasst1: 1",
                "  authored: 1",
                "authored_path: data/authored/training.jsonl",
                "processed_dir: data/processed",
                "max_characters: 12000",
            ]
        ),
        encoding="utf-8",
    )
    fixture_path = tmp_path / "oasst.json"
    fixture_path.write_text(json.dumps(oasst_rows), encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_rows = [
        {
            "id": "eval-general",
            "capability": "instruction",
            "prompt": "Say hello politely.",
            "scorer": "constraint",
        },
        {
            "id": "eval-solar",
            "capability": "knowledge",
            "prompt": "Give two uses of solar energy.",
            "scorer": "rubric",
        },
        {
            "id": "eval-code",
            "capability": "programming",
            "prompt": "Write a Python function that adds two integers.",
            "scorer": "python_syntax",
        },
    ]
    evaluation_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in evaluation_rows), encoding="utf-8"
    )

    output_one = tmp_path / "first"
    output_two = tmp_path / "second"
    args = [
        "--config",
        str(config_path),
        "--oasst-json",
        str(fixture_path),
        "--evaluation-cases",
        str(evaluation_path),
    ]
    assert prepare_main([*args, "--output", str(output_one)]) == 0
    assert prepare_main([*args, "--output", str(output_two)]) == 0

    first_manifest = json.loads((output_one / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((output_two / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["accepted_sha256"] == second_manifest["accepted_sha256"]
    assert first_manifest["rejections_sha256"] == second_manifest["rejections_sha256"]
    assert first_manifest["dataset_revision"] == "fixture-revision"

    train_rows = [
        json.loads(line)
        for line in (output_one / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_rows = [
        json.loads(line)
        for line in (output_one / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert (len(train_rows), len(validation_rows)) == (2, 1)
    assert {row["cluster_id"] for row in train_rows}.isdisjoint(
        {row["cluster_id"] for row in validation_rows}
    )
    assert first_manifest["train_sha256"] == second_manifest["train_sha256"]
    assert first_manifest["validation_sha256"] == second_manifest["validation_sha256"]
    assert first_manifest["contamination_hit_count"] == 2
    contamination_rows = [
        row
        for row in (output_one / "contamination-review.csv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.strip()
    ]
    assert len(contamination_rows) == 3
    assert all(",True," in row for row in contamination_rows[1:])

    accepted_rows = [
        json.loads(line)
        for line in (output_one / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected_row_ids = sorted(
        row_id
        for row in accepted_rows
        if row["source"] == "OpenAssistant/oasst1"
        for row_id in row["metadata"]["row_ids"]
    )
    assert first_manifest["oasst_row_ids"] == selected_row_ids

    rejection_rows = [
        json.loads(line)
        for line in (output_one / "rejections.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {(row["id"], row["reason"]) for row in rejection_rows} == {
        ("assistant-code-deleted", "spam_or_deleted"),
        ("assistant-turkish", "non_english"),
        ("root-turkish", "non_english"),
    }


def test_prepare_script_runs_directly_from_repository(
    tmp_path: Path, oasst_rows: list[dict[str, object]]
) -> None:
    config_path = tmp_path / "direct-data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: OpenAssistant/oasst1",
                "dataset_revision: fixture-revision",
                "language: en",
                "seed: 42",
                "total_records: 3",
                "train_records: 2",
                "validation_records: 1",
                "mixture:",
                "  general_oasst1: 1",
                "  capability_oasst1: 1",
                "  authored: 1",
                "authored_path: data/authored/training.jsonl",
                "processed_dir: data/processed",
                "max_characters: 12000",
            ]
        ),
        encoding="utf-8",
    )
    fixture_path = tmp_path / "direct-oasst.json"
    fixture_path.write_text(json.dumps(oasst_rows), encoding="utf-8")
    output_path = tmp_path / "direct-output"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--config",
            str(config_path),
            "--oasst-json",
            str(fixture_path),
            "--output",
            str(output_path),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_path / "manifest.json").is_file()
