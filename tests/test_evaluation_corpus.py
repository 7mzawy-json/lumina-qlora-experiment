from collections import Counter
from pathlib import Path

import yaml

from lumina_experiment.contracts import Generation
from lumina_experiment.isolation import freeze_files
from lumina_experiment.scoring import load_eval_directory, score_case

EXPECTED_COUNTS = {
    "instruction": 30,
    "reasoning": 30,
    "knowledge": 30,
    "summarization": 30,
    "writing": 30,
    "programming": 30,
    "json": 30,
    "safety": 20,
}


def test_frozen_corpus_has_exact_capability_counts() -> None:
    corpus = load_eval_directory(Path("data/eval/cases"))

    counts = Counter(case.capability for case in corpus)
    assert counts == EXPECTED_COUNTS
    assert len(corpus) == 230
    assert len({case.id for case in corpus}) == 230


def test_every_case_has_complete_scoring_fixture_and_provenance() -> None:
    corpus = load_eval_directory(Path("data/eval/cases"))

    required_keys = {
        "constraints": {"constraints"},
        "exact_match": {"answers"},
        "token_f1": {"reference"},
        "required_points": {"required_points"},
        "python_syntax": {"must_parse"},
        "json_schema": {"schema"},
        "rubric": {"rubric"},
    }
    source_counts = Counter(case.metadata["source_type"] for case in corpus)
    assert source_counts == {"authored": 180, "public_anchor": 50}
    public_capabilities = Counter(
        case.capability for case in corpus if case.metadata["source_type"] == "public_anchor"
    )
    assert public_capabilities == {
        "instruction": 10,
        "reasoning": 10,
        "knowledge": 10,
        "summarization": 10,
        "programming": 10,
    }
    for case in corpus:
        assert case.scorer in required_keys
        assert required_keys[case.scorer] <= case.expected.keys()
        assert case.metadata["review_status"].endswith("pending_human")
        if case.metadata["source_type"] == "public_anchor":
            assert case.metadata["dataset"]
            assert case.metadata["upstream_item_id"]
            assert len(case.metadata["revision"]) == 40
            assert case.metadata["license"]


def test_rubrics_are_bounded_and_observable() -> None:
    corpus = load_eval_directory(Path("data/eval/cases"))

    rubric_cases = [case for case in corpus if case.scorer == "rubric"]
    assert rubric_cases
    for case in rubric_cases:
        rubric = case.expected["rubric"]
        assert rubric["minimum"] == 0
        assert rubric["maximum"] == 2
        assert set(rubric["criteria"]) == {"0", "1", "2"}
        assert all(rubric["criteria"][level].strip() for level in ("0", "1", "2"))


def test_every_scoring_fixture_dispatches_without_execution() -> None:
    corpus = load_eval_directory(Path("data/eval/cases"))

    for case in corpus:
        generation = Generation(
            case_id=case.id,
            condition="fixture-audit",
            output="",
            evidence_state="locally_verified",
        )
        scores = score_case(case, generation)
        assert scores
        assert all(score.case_id == case.id for score in scores)


def test_source_manifest_covers_every_public_anchor() -> None:
    corpus = load_eval_directory(Path("data/eval/cases"))
    manifest = yaml.safe_load(Path("data/eval/sources.yaml").read_text(encoding="utf-8"))

    public_ids = {case.id for case in corpus if case.metadata["source_type"] == "public_anchor"}
    manifest_ids = {anchor["case_id"] for anchor in manifest["public_anchors"]}
    assert manifest["public_anchor_count"] == 50
    assert manifest_ids == public_ids


def test_committed_freeze_matches_canonical_corpus(tmp_path: Path) -> None:
    paths = sorted(Path("data/eval/cases").glob("*.jsonl"))
    paths.append(Path("data/eval/sources.yaml"))

    generated = freeze_files(paths, tmp_path / "verify.sha256")
    committed = Path("data/eval/FROZEN.sha256").read_text(encoding="utf-8").strip()
    assert generated == committed
