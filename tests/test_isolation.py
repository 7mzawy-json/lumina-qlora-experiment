from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumina_experiment.contracts import EvalCase, InstructionRecord, Message
from lumina_experiment.isolation import (
    cluster_split,
    deduplicate,
    find_contamination,
    freeze_files,
)
from scripts.freeze_eval import main as freeze_main


def record(
    record_id: str,
    prompt: str,
    *,
    cluster_id: str | None = None,
    capability: str = "instruction",
) -> InstructionRecord:
    return InstructionRecord(
        id=record_id,
        source="fixture",
        license="CC-BY-4.0",
        capability=capability,
        cluster_id=cluster_id or f"cluster-{record_id}",
        messages=(
            Message(role="user", content=prompt),
            Message(role="assistant", content="A fixture response."),
        ),
        metadata={"language": "en"},
    )


def case(case_id: str, prompt: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        capability="instruction",
        prompt=prompt,
        scorer="constraint",
    )


def test_deduplicate_normalizes_unicode_line_endings_and_whitespace() -> None:
    first = record("a", "  Café\r\nchecklist  ")
    duplicate = record("b", "Cafe\u0301\nchecklist")
    unique = record("c", "A different request")

    kept, removed = deduplicate([unique, duplicate, first])

    assert [item.id for item in kept] == ["a", "c"]
    assert [(item.kept_id, item.duplicate_id) for item in removed] == [("a", "b")]


def test_cluster_never_crosses_train_validation_boundary() -> None:
    records = [
        record("a1", "Prompt A1", cluster_id="cluster-a"),
        record("a2", "Prompt A2", cluster_id="cluster-a"),
        record("b1", "Prompt B1", cluster_id="cluster-b"),
        record("b2", "Prompt B2", cluster_id="cluster-b"),
        record("c1", "Prompt C1", cluster_id="cluster-c"),
        record("c2", "Prompt C2", cluster_id="cluster-c"),
    ]

    split = cluster_split(records, validation_size=2, seed=42)

    train_clusters = {item.cluster_id for item in split.train}
    validation_clusters = {item.cluster_id for item in split.validation}
    assert train_clusters.isdisjoint(validation_clusters)
    assert len(split.validation) == 2


def test_cluster_split_prefers_capability_balance_when_size_ties() -> None:
    records = [
        record("i1", "Instruction one", cluster_id="instruction-only"),
        record("i2", "Instruction two", cluster_id="instruction-only"),
        record("j1", "JSON one", cluster_id="json-only", capability="json"),
        record("j2", "JSON two", cluster_id="json-only", capability="json"),
        record("m1", "Mixed instruction", cluster_id="mixed"),
        record("m2", "Mixed JSON", cluster_id="mixed", capability="json"),
    ]

    split = cluster_split(records, validation_size=2, seed=42)

    assert {item.cluster_id for item in split.validation} == {"mixed"}
    assert {item.capability for item in split.validation} == {"instruction", "json"}


def test_contamination_detects_close_lexical_paraphrase() -> None:
    hits = find_contamination(
        train=[record("train-1", "Summarize this article in exactly three bullet points.")],
        validation=[],
        evaluation=[case("case-1", "Summarize the article using exactly three bullet points.")],
        threshold=0.80,
    )

    assert [(hit.record_id, hit.evaluation_id, hit.source_split) for hit in hits] == [
        ("train-1", "case-1", "train")
    ]
    assert hits[0].similarity >= 0.80


def test_contamination_always_flags_exact_normalized_match() -> None:
    hits = find_contamination(
        train=[],
        validation=[record("validation-1", "Return JSON.\r\n")],
        evaluation=[case("case-1", "  Return JSON.  ")],
        threshold=1.01,
    )

    assert len(hits) == 1
    assert hits[0].exact_match is True


def test_freeze_is_stable_across_json_order_and_temporary_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_rows = [
        {"id": "case-b", "prompt": "B", "metadata": {"z": 1, "a": 2}},
        {"id": "case-a", "prompt": "A", "metadata": {"a": 2, "z": 1}},
    ]
    second_rows = [
        {"metadata": {"z": 1, "a": 2}, "prompt": "A", "id": "case-a"},
        {"prompt": "B", "id": "case-b", "metadata": {"a": 2, "z": 1}},
    ]
    for root, rows in ((first_root, first_rows), (second_root, second_rows)):
        (root / "cases.jsonl").write_text(
            "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
        )

    first_hash = freeze_files([first_root / "cases.jsonl"], first_root / "FROZEN.sha256")
    second_hash = freeze_files([second_root / "cases.jsonl"], second_root / "FROZEN.sha256")

    assert first_hash == second_hash
    assert (first_root / "FROZEN.sha256").read_text(encoding="utf-8") == f"{first_hash}\n"


def test_freeze_cli_refuses_overwrite_and_records_replacement_history(
    tmp_path: Path,
) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case_path = cases_dir / "instruction.jsonl"
    case_path.write_text('{"id":"case-1","prompt":"First"}\n', encoding="utf-8")
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text("sources: []\n", encoding="utf-8")
    output_path = tmp_path / "FROZEN.sha256"
    args = [
        "--cases",
        str(cases_dir),
        "--sources",
        str(sources_path),
        "--output",
        str(output_path),
    ]

    assert freeze_main(args) == 0
    first_hash = output_path.read_text(encoding="utf-8").strip()
    with pytest.raises(FileExistsError, match="replace-freeze"):
        freeze_main(args)

    case_path.write_text('{"id":"case-1","prompt":"Revised"}\n', encoding="utf-8")
    assert freeze_main([*args, "--replace-freeze"]) == 0
    second_hash = output_path.read_text(encoding="utf-8").strip()
    history = [
        json.loads(line)
        for line in (tmp_path / "freeze-history.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert first_hash != second_hash
    assert history[-1]["previous_hash"] == first_hash
    assert history[-1]["new_hash"] == second_hash
