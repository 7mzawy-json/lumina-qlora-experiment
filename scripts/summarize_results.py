from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumina_experiment.contracts import canonical_json  # noqa: E402
from lumina_experiment.reporting import (  # noqa: E402
    combine_condition_manifests,
    render_report,
    summarize_scored_records,
)
from lumina_experiment.statistics import (  # noqa: E402
    Interval,
    ResultSummary,
    evaluate_success,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize paired Lumina evaluation results")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument(
        "--private-key",
        type=Path,
        default=Path("results/raw/review-key.json"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected an object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(payload)
    return rows


def _load_review(path: Path) -> list[Mapping[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _private_review_key(
    path: Path,
) -> tuple[int | None, Mapping[str, Mapping[str, str]]]:
    payload = _load_json(path)
    assignments = payload.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("private review key requires assignments")
    raw_seed = payload.get("seed")
    if raw_seed is not None and (isinstance(raw_seed, bool) or not isinstance(raw_seed, int)):
        raise ValueError("private review key seed must be an integer")
    return raw_seed, assignments


def _evidence_state(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("scored rows must not be empty")
    states = {str(row.get("evidence_state")) for row in rows}
    if len(states) != 1:
        raise ValueError(f"scored rows must share one evidence state, found {sorted(states)}")
    return next(iter(states))


def _manifest(path: Path | None, output: Path, summary: ResultSummary) -> Mapping[str, object]:
    combined_path = output / "evidence-manifest.json"
    base_path = output / "base-manifest.json"
    adapter_path = output / f"{summary.adapter_condition}-manifest.json"
    if summary.evidence_state == "8b_gpu_measured":
        if not base_path.is_file() or not adapter_path.is_file():
            raise ValueError(
                "measured evidence requires base and adapter source condition manifests"
            )
        derived = combine_condition_manifests(
            _load_json(base_path), _load_json(adapter_path), summary
        )
        stored_path = path or combined_path
        if stored_path.is_file() and canonical_json(_load_json(stored_path)) != canonical_json(
            derived
        ):
            raise ValueError(
                "stored combined manifest does not match the derived evidence manifest"
            )
        return derived
    if path is not None:
        return _load_json(path)
    if combined_path.is_file():
        return _load_json(combined_path)
    return {"evidence_state": summary.evidence_state}


def _summary_payload(summary: ResultSummary) -> dict[str, object]:
    payload = asdict(summary)
    payload["decision"] = asdict(evaluate_success(summary))
    return payload


def _summary_from_payload(payload: Mapping[str, object]) -> ResultSummary:
    raw_intervals = payload.get("intervals", {})
    if not isinstance(raw_intervals, Mapping):
        raise ValueError("metrics intervals must be an object")
    intervals = {
        str(name): Interval(**values)
        for name, values in raw_intervals.items()
        if isinstance(values, Mapping)
    }
    return ResultSummary(
        primary_deltas=dict(payload.get("primary_deltas", {})),
        secondary_deltas=dict(payload.get("secondary_deltas", {})),
        writing_adapter_wins=int(payload.get("writing_adapter_wins", 0)),
        writing_base_wins=int(payload.get("writing_base_wins", 0)),
        writing_ties=int(payload.get("writing_ties", 0)),
        diagnostics=tuple(payload.get("diagnostics", ())),
        evidence_state=str(payload.get("evidence_state", "")),
        intervals=intervals,
        scored_case_ids={
            str(name): tuple(str(case_id) for case_id in case_ids)
            for name, case_ids in dict(payload.get("scored_case_ids", {})).items()
        },
        scored_case_counts={
            str(name): int(count)
            for name, count in dict(payload.get("scored_case_counts", {})).items()
        },
        review_case_ids=tuple(str(item) for item in payload.get("review_case_ids", ())),
        review_capability_counts={
            str(name): int(count)
            for name, count in dict(payload.get("review_capability_counts", {})).items()
        },
        review_case_capabilities={
            str(case_id): str(capability)
            for case_id, capability in dict(payload.get("review_case_capabilities", {})).items()
        },
        diagnostic_case_ids=tuple(str(item) for item in payload.get("diagnostic_case_ids", ())),
        diagnostic_generation_case_ids={
            str(name): tuple(str(case_id) for case_id in case_ids)
            for name, case_ids in dict(payload.get("diagnostic_generation_case_ids", {})).items()
        },
        adapter_condition=str(payload.get("adapter_condition", "")),
        generation_hashes={
            str(name): str(value)
            for name, value in dict(payload.get("generation_hashes", {})).items()
        },
        score_hash=str(payload.get("score_hash", "")),
    )


def _metrics_csv_text(summary: ResultSummary) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=["category", "metric", "delta_pp", "lower_pp", "upper_pp"],
        lineterminator="\n",
    )
    writer.writeheader()
    for category, metrics in (
        ("primary", summary.primary_deltas),
        ("secondary", summary.secondary_deltas),
    ):
        for metric, delta in sorted(metrics.items()):
            interval = summary.intervals[metric]
            writer.writerow(
                {
                    "category": category,
                    "metric": metric,
                    "delta_pp": delta,
                    "lower_pp": interval.lower_pp,
                    "upper_pp": interval.upper_pp,
                }
            )
    return handle.getvalue()


def _write_metrics_csv(summary: ResultSummary, path: Path) -> None:
    path.write_text(_metrics_csv_text(summary), encoding="utf-8", newline="\n")


def _verify_existing(args: argparse.Namespace) -> int:
    metrics_path = args.output / "metrics.json"
    report_path = args.output / "report.md"
    summary = _summary_from_payload(_load_json(metrics_path))
    if summary.evidence_state == "8b_gpu_measured":
        missing = [
            name
            for name, value in (
                ("--scores", args.scores),
                ("--review", args.review),
                ("--diagnostics", args.diagnostics),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "measured --verify-only requires source artifacts: " + ", ".join(missing)
            )
        rows = _load_jsonl(args.scores)
        review_seed, private_key = _private_review_key(args.private_key)
        recomputed = summarize_scored_records(
            rows,
            _load_review(args.review),
            private_key,
            evidence_state=_evidence_state(rows),
            diagnostics=_load_jsonl(args.diagnostics),
            seed=args.seed,
            resamples=args.resamples,
            review_seed=review_seed,
        )
        if recomputed != summary:
            raise ValueError("existing metrics do not match recomputed source artifacts")
    manifest = _manifest(args.manifest, args.output, summary)
    rendered = render_report(summary, manifest)
    if report_path.read_text(encoding="utf-8") != rendered:
        raise ValueError("existing report does not match validated metrics and manifest")
    metrics_csv_path = args.output / "metrics.csv"
    if metrics_csv_path.read_text(encoding="utf-8") != _metrics_csv_text(summary):
        raise ValueError("existing metrics.csv does not match validated metrics")
    print(f"verified result summary in {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        return _verify_existing(args)
    if args.scores is None or args.review is None:
        raise ValueError("--scores and --review are required unless --verify-only is used")

    rows = _load_jsonl(args.scores)
    reviews = _load_review(args.review)
    diagnostics = _load_jsonl(args.diagnostics) if args.diagnostics else []
    review_seed, private_key = _private_review_key(args.private_key)
    summary = summarize_scored_records(
        rows,
        reviews,
        private_key,
        evidence_state=_evidence_state(rows),
        diagnostics=diagnostics,
        seed=args.seed,
        resamples=args.resamples,
        review_seed=review_seed,
    )
    manifest = _manifest(args.manifest, args.output, summary)
    report = render_report(summary, manifest)

    args.output.mkdir(parents=True, exist_ok=True)
    if summary.evidence_state == "8b_gpu_measured":
        (args.output / "evidence-manifest.json").write_text(
            f"{canonical_json(manifest)}\n",
            encoding="utf-8",
            newline="\n",
        )
    (args.output / "metrics.json").write_text(
        f"{canonical_json(_summary_payload(summary))}\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_metrics_csv(summary, args.output / "metrics.csv")
    (args.output / "report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"wrote validated summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
