from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumina_experiment.contracts import EvalCase, InstructionRecord, canonical_json  # noqa: E402
from lumina_experiment.data_pipeline import (  # noqa: E402
    DataPreparationConfig,
    audit_oasst_rows,
    extract_oasst_pairs,
    filter_records,
    load_data_config,
    prepare_dataset,
)
from lumina_experiment.isolation import (  # noqa: E402
    ContaminationHit,
    cluster_split,
    deduplicate,
    find_contamination,
)


def _load_json(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected a JSON array of objects in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _resolve_revision(config: DataPreparationConfig, config_path: Path) -> str:
    if config.dataset_revision:
        return config.dataset_revision
    if not config.revision_file:
        raise ValueError("dataset_revision or revision_file is required")
    revision_path = Path(config.revision_file)
    if not revision_path.is_absolute():
        project_candidate = Path.cwd() / revision_path
        config_candidate = config_path.parent / revision_path
        revision_path = project_candidate if project_candidate.exists() else config_candidate
    revisions = yaml.safe_load(revision_path.read_text(encoding="utf-8")) or {}
    if not isinstance(revisions, dict):
        raise ValueError("revision file must contain a YAML mapping")
    revision = revisions.get(config.dataset_id)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"no immutable revision pinned for {config.dataset_id}")
    return revision.strip()


def _download_oasst(config: DataPreparationConfig, revision: str) -> list[dict[str, object]]:
    from datasets import load_dataset

    dataset = load_dataset(config.dataset_id, revision=revision, split="train")
    return [dict(row) for row in dataset]


def _jsonl_text(rows: Sequence[dict[str, object]]) -> str:
    return "".join(f"{canonical_json(row)}\n" for row in rows)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_evaluation_cases(path: Path) -> list[EvalCase]:
    paths = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    rows = [row for case_path in paths for row in _load_jsonl(case_path)]
    return [
        EvalCase(
            id=str(row["id"]),
            capability=str(row["capability"]),
            prompt=str(row["prompt"]),
            scorer=str(row["scorer"]),
            expected=row.get("expected", {}),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def _contamination_csv(hits: Sequence[ContaminationHit]) -> str:
    output = io.StringIO(newline="")
    fieldnames = [
        "record_id",
        "evaluation_id",
        "source_split",
        "similarity",
        "exact_match",
        "record_prompt",
        "evaluation_prompt",
        "adjudication",
        "notes",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for hit in hits:
        writer.writerow(
            {
                "record_id": hit.record_id,
                "evaluation_id": hit.evaluation_id,
                "source_split": hit.source_split,
                "similarity": f"{hit.similarity:.6f}",
                "exact_match": hit.exact_match,
                "record_prompt": hit.record_prompt,
                "evaluation_prompt": hit.evaluation_prompt,
                "adjudication": "",
                "notes": "",
            }
        )
    return output.getvalue()


def _write_outputs(
    output: Path,
    accepted: list[InstructionRecord],
    train: list[InstructionRecord],
    validation: list[InstructionRecord],
    rejections: list[dict[str, object]],
    revision: str,
    contamination_text: str | None,
    contamination_hit_count: int | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    accepted_rows = [record.to_dict() for record in accepted]
    accepted_text = _jsonl_text(accepted_rows)
    train_text = _jsonl_text([record.to_dict() for record in train])
    validation_text = _jsonl_text([record.to_dict() for record in validation])
    rejections_text = _jsonl_text(rejections)
    (output / "accepted.jsonl").write_text(accepted_text, encoding="utf-8", newline="\n")
    (output / "train.jsonl").write_text(train_text, encoding="utf-8", newline="\n")
    (output / "validation.jsonl").write_text(validation_text, encoding="utf-8", newline="\n")
    (output / "rejections.jsonl").write_text(rejections_text, encoding="utf-8", newline="\n")
    if contamination_text is not None:
        (output / "contamination-review.csv").write_text(
            contamination_text, encoding="utf-8", newline="\n"
        )

    manifest = {
        "dataset_revision": revision,
        "accepted_count": len(accepted),
        "train_count": len(train),
        "validation_count": len(validation),
        "rejection_count": len(rejections),
        "source_counts": dict(sorted(Counter(record.source for record in accepted).items())),
        "capability_counts": dict(
            sorted(Counter(record.capability for record in accepted).items())
        ),
        "accepted_sha256": _sha256(accepted_text),
        "train_sha256": _sha256(train_text),
        "validation_sha256": _sha256(validation_text),
        "rejections_sha256": _sha256(rejections_text),
        "contamination_status": "screened" if contamination_text is not None else "not_run",
        "contamination_hit_count": contamination_hit_count,
        "contamination_review_sha256": (
            _sha256(contamination_text) if contamination_text is not None else None
        ),
        "oasst_row_ids": sorted(
            row_id
            for record in accepted
            if record.source == "OpenAssistant/oasst1"
            for row_id in record.metadata["row_ids"]
        ),
    }
    (output / "manifest.json").write_text(
        f"{canonical_json(manifest)}\n", encoding="utf-8", newline="\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic Lumina instruction data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oasst-json",
        type=Path,
        help="Offline JSON fixture; omit to download the pinned Hugging Face revision",
    )
    parser.add_argument(
        "--evaluation-cases",
        type=Path,
        default=Path("data/eval/cases"),
        help="Evaluation JSONL file/directory; screening is marked not_run when absent",
    )
    parser.add_argument("--contamination-threshold", type=float, default=0.92)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_data_config(args.config)
    revision = _resolve_revision(config, args.config)
    oasst_rows = (
        _load_json(args.oasst_json) if args.oasst_json else _download_oasst(config, revision)
    )
    authored_rows = _load_jsonl(Path(config.authored_path))

    raw_rejections = audit_oasst_rows(oasst_rows)
    extracted = extract_oasst_pairs(oasst_rows)
    _, oasst_rejections = filter_records(extracted, config.max_characters)
    accepted = prepare_dataset(oasst_rows, authored_rows, config)
    unique_records, duplicates = deduplicate(accepted)
    if duplicates:
        duplicate_ids = ", ".join(item.duplicate_id for item in duplicates[:5])
        raise ValueError(
            "deduplication reduced the configured dataset count; "
            f"replace duplicate candidates before continuing: {duplicate_ids}"
        )
    split = cluster_split(
        unique_records,
        validation_size=config.validation_records,
        seed=config.seed,
    )
    if (
        len(split.train) != config.train_records
        or len(split.validation) != config.validation_records
    ):
        raise ValueError(
            "cluster-aware split could not satisfy exact configured counts: "
            f"train={len(split.train)}, validation={len(split.validation)}"
        )
    contamination_text: str | None = None
    contamination_hit_count: int | None = None
    if args.evaluation_cases.exists():
        evaluation_cases = _load_evaluation_cases(args.evaluation_cases)
        hits = find_contamination(
            split.train,
            split.validation,
            evaluation_cases,
            threshold=args.contamination_threshold,
        )
        contamination_text = _contamination_csv(hits)
        contamination_hit_count = len(hits)
    _write_outputs(
        args.output,
        unique_records,
        list(split.train),
        list(split.validation),
        [
            rejection.to_dict()
            for rejection in sorted(
                [*raw_rejections, *oasst_rejections],
                key=lambda item: (item.id, item.reason),
            )
        ],
        revision,
        contamination_text,
        contamination_hit_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
