from collections import Counter
from pathlib import Path

import pytest

import lumina_experiment.scoring as scoring
from lumina_experiment.contracts import EvalCase, Generation
from lumina_experiment.scoring import (
    load_eval_directory,
    score_case,
    score_constraints,
    score_exact_match,
    score_json_schema,
    score_python_syntax,
    score_required_points,
    score_token_f1,
)


def test_json_schema_rejects_extra_properties() -> None:
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
        "additionalProperties": False,
    }

    score = score_json_schema('{"answer":"yes","extra":1}', schema)

    assert score.passed is False
    assert score.details["parsed"] is True
    assert score.details["errors"]


def test_json_schema_reports_parse_failure_separately() -> None:
    score = score_json_schema("not-json", {"type": "object"})

    assert score.passed is False
    assert score.details["parsed"] is False
    assert score.details["errors"]


def test_constraint_score_requires_every_constraint() -> None:
    constraints = [
        {"kind": "max_words", "value": 8},
        {"kind": "required_substring", "value": "solar"},
        {"kind": "forbidden_substring", "value": "fossil"},
    ]

    assert score_constraints("Solar energy reduces emissions.", constraints).passed is True


def test_constraint_score_supports_lists_headings_and_regex() -> None:
    output = "## Plan\n- Audit inputs\n- Record results"
    constraints = [
        {"kind": "required_heading", "value": "Plan"},
        {"kind": "exact_list_length", "value": 2},
        {"kind": "regex_shape", "value": r"(?s)^## Plan\n(?:- .+\n?){2}$"},
    ]

    result = score_constraints(output, constraints)

    assert result.passed is True
    assert len(result.details["checks"]) == 3


def test_exact_match_normalizes_outer_whitespace_and_case() -> None:
    result = score_exact_match("  PARIS\n", ["Paris", "City of Paris"])

    assert result.passed is True


def test_token_f1_normalizes_case_and_punctuation() -> None:
    result = score_token_f1("Solar, power works!", "solar power works")

    assert result.value == pytest.approx(1.0)
    assert result.passed is True


def test_required_points_rejects_explicit_contradiction() -> None:
    result = score_required_points(
        "The pilot uses QLoRA and executes generated code.",
        required_points=["uses QLoRA"],
        contradiction_phrases=["executes generated code"],
    )

    assert result.passed is False
    assert result.details["coverage"] == 1.0
    assert result.details["contradictions"] == ["executes generated code"]


def test_python_syntax_only_parses_and_never_executes(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist.txt"
    dangerous_but_valid = f'open({str(marker)!r}, "w").write("executed")'

    result = score_python_syntax(dangerous_but_valid)

    assert result.passed is True
    assert not marker.exists()
    assert score_python_syntax("def broken(:\n    pass").passed is False


def test_score_case_emits_parse_and_schema_metrics() -> None:
    case = EvalCase(
        id="json-001",
        capability="json",
        prompt="Return an answer object.",
        scorer="json_schema",
        expected={
            "schema": {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
                "additionalProperties": False,
            }
        },
    )
    generation = Generation(
        case_id="json-001",
        condition="base",
        output='{"answer":"yes"}',
        evidence_state="smoke_test_verified",
    )

    scores = score_case(case, generation)

    assert [(score.metric, score.value) for score in scores] == [
        ("json_parse_success", 1.0),
        ("json_schema_valid", 1.0),
    ]
    assert all(score.case_id == case.id for score in scores)
    assert all(score.condition == generation.condition for score in scores)


def test_score_case_rejects_mismatched_generation() -> None:
    case = EvalCase(
        id="instruction-001",
        capability="instruction",
        prompt="Reply yes.",
        scorer="exact_match",
        expected={"answers": ["yes"]},
    )
    generation = Generation(
        case_id="other",
        condition="base",
        output="yes",
        evidence_state="smoke_test_verified",
    )

    with pytest.raises(ValueError, match="case_id"):
        score_case(case, generation)


def test_load_eval_directory_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = (
        '{"id":"duplicate","capability":"instruction","prompt":"A","scorer":"exact_match"}\n'
        '{"id":"duplicate","capability":"instruction","prompt":"B","scorer":"exact_match"}\n'
    )
    (tmp_path / "cases.jsonl").write_text(rows, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate evaluation id"):
        load_eval_directory(tmp_path)


def test_balanced_pilot_selection_is_reproducible_and_covers_every_capability() -> None:
    capabilities = (
        "instruction",
        "json",
        "knowledge",
        "programming",
        "reasoning",
        "safety",
        "summarization",
        "writing",
    )
    cases = tuple(
        EvalCase(
            id=f"{capability}-{index:03d}",
            capability=capability,
            prompt=f"{capability} prompt {index}",
            scorer="rubric",
        )
        for capability in capabilities
        for index in range(1, 6)
    )
    selector = getattr(scoring, "select_balanced_cases", None)

    assert callable(selector)
    selected = selector(cases, limit=24, seed=42)
    reversed_selected = selector(tuple(reversed(cases)), limit=24, seed=42)

    assert [case.id for case in selected] == [case.id for case in reversed_selected]
    assert Counter(case.capability for case in selected) == {
        capability: 3 for capability in capabilities
    }
