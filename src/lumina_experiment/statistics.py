from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from lumina_experiment.contracts import EVIDENCE_STATES

PRIMARY_METRICS = frozenset({"instruction_following", "json_schema_validity"})
SECONDARY_METRICS = frozenset({"reasoning", "knowledge", "summarization", "programming"})
REVIEW_CAPABILITIES = frozenset(
    {
        "instruction",
        "reasoning",
        "knowledge",
        "summarization",
        "writing",
        "programming",
        "json",
    }
)


@dataclass(frozen=True)
class Interval:
    difference_pp: float
    lower_pp: float
    upper_pp: float
    resamples: int


@dataclass(frozen=True)
class ResultSummary:
    primary_deltas: Mapping[str, float]
    secondary_deltas: Mapping[str, float]
    writing_adapter_wins: int
    writing_base_wins: int
    writing_ties: int
    diagnostics: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    evidence_state: str = "planned"
    intervals: Mapping[str, Interval] = field(default_factory=dict)
    scored_case_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    scored_case_counts: Mapping[str, int] = field(default_factory=dict)
    review_case_ids: tuple[str, ...] = field(default_factory=tuple)
    review_capability_counts: Mapping[str, int] = field(default_factory=dict)
    review_case_capabilities: Mapping[str, str] = field(default_factory=dict)
    diagnostic_case_ids: tuple[str, ...] = field(default_factory=tuple)
    diagnostic_generation_case_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    adapter_condition: str = ""
    generation_hashes: Mapping[str, str] = field(default_factory=dict)
    score_hash: str = ""

    def __post_init__(self) -> None:
        if self.evidence_state not in EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence_state: {self.evidence_state}")
        if min(self.writing_adapter_wins, self.writing_base_wins, self.writing_ties) < 0:
            raise ValueError("writing review counts must be non-negative")


@dataclass(frozen=True)
class Decision:
    passed: bool
    criteria: Mapping[str, bool]
    reasons: tuple[str, ...]


def paired_bootstrap_difference(
    base: Sequence[float],
    adapter: Sequence[float],
    seed: int = 42,
    resamples: int = 10_000,
) -> Interval:
    if len(base) != len(adapter) or not base:
        raise ValueError("paired observations must be non-empty and have equal length")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    base_array = np.asarray(base, dtype=float)
    adapter_array = np.asarray(adapter, dtype=float)
    if not np.isfinite(base_array).all() or not np.isfinite(adapter_array).all():
        raise ValueError("paired observations must be finite")

    paired_deltas = adapter_array - base_array
    difference = float(paired_deltas.mean() * 100.0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired_deltas), size=(resamples, len(paired_deltas)))
    replicate_differences = paired_deltas[indices].mean(axis=1) * 100.0
    lower, upper = np.percentile(replicate_differences, [2.5, 97.5])
    return Interval(
        difference_pp=difference,
        lower_pp=float(lower),
        upper_pp=float(upper),
        resamples=resamples,
    )


def evaluate_success(summary: ResultSummary) -> Decision:
    primary_values = list(summary.primary_deltas.values())
    non_tied = summary.writing_adapter_wins + summary.writing_base_wins
    writing_win_rate = summary.writing_adapter_wins / non_tied if non_tied else 0.0
    complete_metrics = (
        set(summary.primary_deltas) == PRIMARY_METRICS
        and set(summary.secondary_deltas) == SECONDARY_METRICS
    )
    scored_id_sets = {alias: set(case_ids) for alias, case_ids in summary.scored_case_ids.items()}
    complete_scored_evaluation = (
        summary.scored_case_counts == {"base": 210, "adapter": 210}
        and set(summary.scored_case_ids) == {"base", "adapter"}
        and all(
            len(summary.scored_case_ids[alias]) == 210 and len(scored_id_sets[alias]) == 210
            for alias in ("base", "adapter")
        )
        and scored_id_sets["base"] == scored_id_sets["adapter"]
        and set(summary.diagnostic_generation_case_ids) == {"base", "adapter"}
        and all(
            len(summary.diagnostic_generation_case_ids[alias]) == 20
            and len(set(summary.diagnostic_generation_case_ids[alias])) == 20
            for alias in ("base", "adapter")
        )
        and set(summary.diagnostic_generation_case_ids["base"])
        == set(summary.diagnostic_generation_case_ids["adapter"])
        == set(summary.diagnostic_case_ids)
    )
    complete_review = (
        summary.review_capability_counts == {capability: 6 for capability in REVIEW_CAPABILITIES}
        and len(summary.review_case_ids) == 42
        and len(set(summary.review_case_ids)) == 42
        and set(summary.review_case_capabilities) == set(summary.review_case_ids)
        and {
            capability: tuple(summary.review_case_capabilities.values()).count(capability)
            for capability in REVIEW_CAPABILITIES
        }
        == summary.review_capability_counts
        and summary.writing_adapter_wins + summary.writing_base_wins + summary.writing_ties == 6
    )
    diagnostic_ids = [str(diagnostic.get("case_id", "")) for diagnostic in summary.diagnostics]
    complete_diagnostics = (
        len(summary.diagnostics) == 20
        and len(diagnostic_ids) == len(set(diagnostic_ids))
        and set(diagnostic_ids) == set(summary.diagnostic_case_ids)
        and len(summary.diagnostic_case_ids) == 20
        and all(
            isinstance(diagnostic.get("serious_new_failure"), bool)
            for diagnostic in summary.diagnostics
        )
    )
    criteria = {
        "complete_preregistered_metrics": complete_metrics,
        "complete_scored_evaluation": complete_scored_evaluation,
        "complete_concealed_review": complete_review,
        "complete_safety_diagnostics": complete_diagnostics,
        "one_primary_gain_at_least_15pp": bool(primary_values) and max(primary_values) >= 15.0,
        "all_primary_deltas_positive": complete_metrics
        and all(delta > 0.0 for delta in primary_values),
        "secondary_regressions_within_5pp": complete_metrics
        and all(delta >= -5.0 for delta in summary.secondary_deltas.values()),
        "writing_adapter_win_rate_above_60pct": complete_review and writing_win_rate > 0.60,
        "no_serious_new_diagnostic_failure": complete_diagnostics
        and not any(
            diagnostic.get("serious_new_failure") is True for diagnostic in summary.diagnostics
        ),
    }
    reasons = tuple(name for name, passed in criteria.items() if not passed)
    return Decision(passed=all(criteria.values()), criteria=criteria, reasons=reasons)
