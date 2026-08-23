from dataclasses import replace

import pytest

from lumina_experiment.statistics import (
    ResultSummary,
    evaluate_success,
    paired_bootstrap_difference,
)


def result_summary(
    *,
    instruction_delta: float = 16.7,
    json_delta: float = 2.0,
    secondary_delta: float = 0.0,
    adapter_wins: int = 4,
    base_wins: int = 1,
    ties: int = 1,
    serious_failure: bool = False,
) -> ResultSummary:
    return ResultSummary(
        primary_deltas={
            "instruction_following": instruction_delta,
            "json_schema_validity": json_delta,
        },
        secondary_deltas={
            "reasoning": secondary_delta,
            "knowledge": 0.0,
            "summarization": 0.0,
            "programming": 0.0,
        },
        writing_adapter_wins=adapter_wins,
        writing_base_wins=base_wins,
        writing_ties=ties,
        diagnostics=tuple(
            {
                "case_id": f"safety-{index:03d}",
                "serious_new_failure": serious_failure and index == 1,
            }
            for index in range(1, 21)
        ),
        evidence_state="8b_gpu_measured",
        scored_case_ids={
            "base": tuple(f"case-{index:03d}" for index in range(210)),
            "adapter": tuple(f"case-{index:03d}" for index in range(210)),
        },
        scored_case_counts={"base": 210, "adapter": 210},
        review_capability_counts={
            "instruction": 6,
            "reasoning": 6,
            "knowledge": 6,
            "summarization": 6,
            "writing": 6,
            "programming": 6,
            "json": 6,
        },
        review_case_ids=tuple(f"review-{index:03d}" for index in range(42)),
        review_case_capabilities={
            f"review-{index:03d}": capability
            for index, capability in enumerate(
                capability
                for capability in (
                    "instruction",
                    "reasoning",
                    "knowledge",
                    "summarization",
                    "writing",
                    "programming",
                    "json",
                )
                for _ in range(6)
            )
        },
        diagnostic_case_ids=tuple(f"safety-{index:03d}" for index in range(1, 21)),
        diagnostic_generation_case_ids={
            "base": tuple(f"safety-{index:03d}" for index in range(1, 21)),
            "adapter": tuple(f"safety-{index:03d}" for index in range(1, 21)),
        },
    )


def test_bootstrap_is_reproducible() -> None:
    first = paired_bootstrap_difference([0, 1, 0], [1, 1, 1], seed=42)
    second = paired_bootstrap_difference([0, 1, 0], [1, 1, 1], seed=42)

    assert first == second
    assert first.difference_pp == pytest.approx(66.6666666667)
    assert first.resamples == 10_000


def test_bootstrap_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="paired observations"):
        paired_bootstrap_difference([0, 1], [1])


def test_success_requires_both_primary_metrics_to_improve() -> None:
    summary = result_summary(instruction_delta=16.7, json_delta=0.0)

    assert evaluate_success(summary).passed is False


def test_success_requires_one_primary_gain_of_fifteen_points() -> None:
    summary = result_summary(instruction_delta=14.9, json_delta=2.0)

    assert evaluate_success(summary).passed is False


def test_success_rejects_secondary_regression_below_five_points() -> None:
    summary = result_summary(secondary_delta=-5.1)

    assert evaluate_success(summary).passed is False


def test_writing_win_rate_is_strictly_greater_than_sixty_percent() -> None:
    summary = result_summary(adapter_wins=3, base_wins=2, ties=1)

    assert evaluate_success(summary).passed is False


def test_success_rejects_serious_new_diagnostic_failure() -> None:
    summary = result_summary(serious_failure=True)

    assert evaluate_success(summary).passed is False


def test_success_requires_explicit_boolean_diagnostic_judgments() -> None:
    summary = result_summary()
    diagnostics = list(summary.diagnostics)
    diagnostics[0] = {"case_id": diagnostics[0]["case_id"]}

    decision = evaluate_success(replace(summary, diagnostics=tuple(diagnostics)))

    assert decision.criteria["complete_safety_diagnostics"] is False


def test_success_rejects_missing_preregistered_secondary_metric() -> None:
    summary = replace(result_summary(), secondary_deltas={"reasoning": 0.0})

    decision = evaluate_success(summary)

    assert decision.passed is False
    assert decision.criteria["complete_preregistered_metrics"] is False


def test_success_rejects_incomplete_concealed_review() -> None:
    summary = replace(
        result_summary(),
        review_capability_counts={
            "instruction": 6,
            "reasoning": 6,
            "knowledge": 6,
            "summarization": 6,
            "writing": 5,
            "programming": 6,
            "json": 6,
        },
        review_case_ids=tuple(f"review-{index:03d}" for index in range(41)),
        writing_adapter_wins=4,
        writing_base_wins=1,
        writing_ties=0,
    )

    decision = evaluate_success(summary)

    assert decision.passed is False
    assert decision.criteria["complete_concealed_review"] is False


def test_success_rejects_fabricated_complete_counts_without_paired_ids() -> None:
    summary = replace(result_summary(), scored_case_ids={})

    decision = evaluate_success(summary)

    assert decision.passed is False
    assert decision.criteria["complete_scored_evaluation"] is False


def test_success_gate_passes_only_when_every_criterion_passes() -> None:
    decision = evaluate_success(result_summary())

    assert decision.passed is True
    assert all(decision.criteria.values())
    assert decision.reasons == ()
