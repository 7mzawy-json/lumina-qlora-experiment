from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from lumina_experiment.contracts import EvalCase, Generation, Score


def _result(
    metric: str,
    value: float,
    passed: bool | None,
    details: Mapping[str, object] | None = None,
) -> Score:
    """Build an unbound scoring result for the standalone scorer APIs."""
    return Score(
        case_id="",
        condition="",
        metric=metric,
        value=float(value),
        passed=passed,
        details=dict(details or {}),
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def score_json_schema(output: str, schema: Mapping[str, object]) -> Score:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        return _result(
            "json_schema_valid",
            0.0,
            False,
            {"parsed": False, "errors": [f"line {exc.lineno}, column {exc.colno}: {exc.msg}"]},
        )

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda error: list(error.absolute_path))
    messages = [error.message for error in errors]
    return _result(
        "json_schema_valid",
        0.0 if errors else 1.0,
        not errors,
        {"parsed": True, "errors": messages},
    )


def _list_length(output: str) -> int:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return len(parsed)
    return len(re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S", output))


def _constraint_check(output: str, constraint: Mapping[str, object]) -> tuple[bool, object]:
    kind = constraint.get("kind")
    value = constraint.get("value")
    normalized = _normalized_text(output)

    if kind == "max_words":
        actual = len(_tokens(output))
        return actual <= int(value), actual
    if kind == "required_substring":
        return _normalized_text(str(value)) in normalized, str(value)
    if kind == "forbidden_substring":
        return _normalized_text(str(value)) not in normalized, str(value)
    if kind == "required_heading":
        heading = re.escape(str(value).strip())
        matched = re.search(rf"(?im)^\s*#*\s*{heading}\s*:?[ \t]*$", output) is not None
        return matched, str(value)
    if kind == "exact_list_length":
        actual = _list_length(output)
        return actual == int(value), actual
    if kind == "regex_shape":
        matched = re.fullmatch(str(value), output.strip()) is not None
        return matched, str(value)
    raise ValueError(f"unsupported constraint kind: {kind}")


def score_constraints(output: str, constraints: Sequence[Mapping[str, object]]) -> Score:
    checks: list[dict[str, object]] = []
    for constraint in constraints:
        passed, observed = _constraint_check(output, constraint)
        checks.append(
            {
                "kind": constraint.get("kind"),
                "expected": constraint.get("value"),
                "observed": observed,
                "passed": passed,
            }
        )
    passed = bool(checks) and all(bool(check["passed"]) for check in checks)
    value = sum(bool(check["passed"]) for check in checks) / len(checks) if checks else 0.0
    return _result("constraint_adherence", value, passed, {"checks": checks})


def score_exact_match(
    output: str, answers: str | Sequence[str], *, case_sensitive: bool = False
) -> Score:
    candidates = [answers] if isinstance(answers, str) else list(answers)
    observed = " ".join(output.split())
    if not case_sensitive:
        observed = observed.casefold()
    normalized_candidates = [" ".join(answer.split()) for answer in candidates]
    if not case_sensitive:
        normalized_candidates = [answer.casefold() for answer in normalized_candidates]
    passed = observed in normalized_candidates
    return _result(
        "exact_match",
        1.0 if passed else 0.0,
        passed,
        {"accepted_answers": candidates, "case_sensitive": case_sensitive},
    )


def score_token_f1(output: str, reference: str, *, pass_threshold: float = 0.8) -> Score:
    prediction_tokens = _tokens(output)
    reference_tokens = _tokens(reference)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if not prediction_tokens and not reference_tokens:
        value = 1.0
    elif not prediction_tokens or not reference_tokens or overlap == 0:
        value = 0.0
    else:
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        value = 2 * precision * recall / (precision + recall)
    return _result(
        "token_f1",
        value,
        value >= pass_threshold,
        {
            "threshold": pass_threshold,
            "prediction_tokens": len(prediction_tokens),
            "reference_tokens": len(reference_tokens),
        },
    )


def score_required_points(
    output: str,
    required_points: Sequence[str],
    contradiction_phrases: Sequence[str] = (),
    *,
    pass_threshold: float = 1.0,
) -> Score:
    normalized = _normalized_text(output)
    matched = [point for point in required_points if _normalized_text(point) in normalized]
    contradictions = [
        phrase for phrase in contradiction_phrases if _normalized_text(phrase) in normalized
    ]
    coverage = len(matched) / len(required_points) if required_points else 1.0
    passed = coverage >= pass_threshold and not contradictions
    return _result(
        "required_points",
        coverage,
        passed,
        {
            "coverage": coverage,
            "matched_points": matched,
            "missing_points": [point for point in required_points if point not in matched],
            "contradictions": contradictions,
            "threshold": pass_threshold,
        },
    )


def _extract_python(output: str) -> str:
    match = re.fullmatch(r"\s*```(?:python|py)?\s*\n(?P<code>.*?)\n```\s*", output, re.DOTALL)
    return match.group("code") if match else output


def score_python_syntax(output: str) -> Score:
    try:
        ast.parse(_extract_python(output))
    except SyntaxError as exc:
        return _result(
            "python_syntax_valid",
            0.0,
            False,
            {"error": exc.msg, "line": exc.lineno, "executed": False},
        )
    return _result("python_syntax_valid", 1.0, True, {"executed": False})


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _as_string_sequence(value: object, name: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a string or sequence of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain only strings")
    return list(value)


def _bind(result: Score, case: EvalCase, generation: Generation) -> Score:
    return replace(result, case_id=case.id, condition=generation.condition)


def score_case(case: EvalCase, generation: Generation) -> list[Score]:
    if case.id != generation.case_id:
        raise ValueError(
            f"generation case_id {generation.case_id!r} does not match case {case.id!r}"
        )

    expected = case.expected
    if case.scorer == "json_schema":
        schema_result = score_json_schema(
            generation.output, _as_mapping(expected.get("schema"), "schema")
        )
        parse_passed = bool(schema_result.details["parsed"])
        parse_result = _result(
            "json_parse_success",
            1.0 if parse_passed else 0.0,
            parse_passed,
            {"errors": schema_result.details["errors"] if not parse_passed else []},
        )
        return [_bind(parse_result, case, generation), _bind(schema_result, case, generation)]

    if case.scorer == "constraints":
        raw_constraints = expected.get("constraints")
        if not isinstance(raw_constraints, Sequence) or isinstance(raw_constraints, (str, bytes)):
            raise ValueError("constraints must be a sequence")
        constraints = [_as_mapping(item, "constraint") for item in raw_constraints]
        result = score_constraints(generation.output, constraints)
    elif case.scorer == "exact_match":
        answers = _as_string_sequence(expected.get("answers"), "answers")
        result = score_exact_match(
            generation.output,
            answers,
            case_sensitive=bool(expected.get("case_sensitive", False)),
        )
    elif case.scorer == "token_f1":
        reference = expected.get("reference")
        if not isinstance(reference, str):
            raise ValueError("reference must be a string")
        result = score_token_f1(
            generation.output,
            reference,
            pass_threshold=float(expected.get("pass_threshold", 0.8)),
        )
    elif case.scorer == "required_points":
        points = _as_string_sequence(expected.get("required_points"), "required_points")
        contradictions = _as_string_sequence(
            expected.get("contradiction_phrases", []), "contradiction_phrases"
        )
        result = score_required_points(
            generation.output,
            points,
            contradictions,
            pass_threshold=float(expected.get("pass_threshold", 1.0)),
        )
    elif case.scorer == "python_syntax":
        parsed = score_python_syntax(generation.output)
        must_parse = bool(expected.get("must_parse", True))
        if must_parse:
            result = parsed
        else:
            passed = not bool(parsed.passed)
            result = _result(
                "python_syntax_expectation",
                1.0 if passed else 0.0,
                passed,
                {**parsed.details, "must_parse": False},
            )
    elif case.scorer == "rubric":
        result = _result(
            "human_rubric",
            0.0,
            None,
            {"requires_human_review": True, "rubric": expected.get("rubric")},
        )
    else:
        raise ValueError(f"unsupported scorer: {case.scorer}")
    return [_bind(result, case, generation)]


def _require_text(payload: Mapping[str, object], field: str, source: Path, line: int) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}:{line}: {field} must be a non-empty string")
    return value.strip()


def load_eval_directory(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for source in sorted(path.rglob("*.jsonl")):
        for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{source}:{line_number}: evaluation case must be an object")
            expected = _as_mapping(payload.get("expected", {}), "expected")
            metadata = _as_mapping(payload.get("metadata", {}), "metadata")
            case = EvalCase(
                id=_require_text(payload, "id", source, line_number),
                capability=_require_text(payload, "capability", source, line_number),
                prompt=_require_text(payload, "prompt", source, line_number),
                scorer=_require_text(payload, "scorer", source, line_number),
                expected=dict(expected),
                metadata=dict(metadata),
            )
            if case.id in seen_ids:
                raise ValueError(f"duplicate evaluation id: {case.id}")
            seen_ids.add(case.id)
            cases.append(case)
    return cases


def select_balanced_cases(cases: Sequence[EvalCase], *, limit: int, seed: int) -> list[EvalCase]:
    """Select a deterministic capability-balanced subset for a bounded pilot."""

    if limit <= 0:
        raise ValueError("balanced evaluation limit must be positive")
    if limit > len(cases):
        raise ValueError(f"balanced evaluation limit {limit} exceeds {len(cases)} cases")
    grouped: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        grouped[case.capability].append(case)
    capabilities = sorted(grouped)
    if limit < len(capabilities):
        raise ValueError("balanced evaluation limit must cover every capability")

    rng = random.Random(seed)
    per_capability, remainder = divmod(limit, len(capabilities))
    selected: list[EvalCase] = []
    for index, capability in enumerate(capabilities):
        candidates = sorted(grouped[capability], key=lambda case: case.id)
        rng.shuffle(candidates)
        count = per_capability + (1 if index < remainder else 0)
        if len(candidates) < count:
            raise ValueError(
                f"capability {capability!r} has {len(candidates)} cases; {count} required"
            )
        selected.extend(candidates[:count])
    return selected
