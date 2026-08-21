from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumina_experiment.contracts import InstructionRecord, canonical_json  # noqa: E402
from lumina_experiment.data_pipeline import (  # noqa: E402
    DataPreparationConfig,
    audit_oasst_rows,
    extract_oasst_pairs,
    filter_records,
    load_data_config,
    prepare_dataset,
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


def _write_outputs(
    output: Path,
    accepted: list[InstructionRecord],
    rejections: list[dict[str, object]],
    revision: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    accepted_rows = [record.to_dict() for record in accepted]
    accepted_text = _jsonl_text(accepted_rows)
    rejections_text = _jsonl_text(rejections)
    (output / "accepted.jsonl").write_text(accepted_text, encoding="utf-8", newline="\n")
    (output / "rejections.jsonl").write_text(rejections_text, encoding="utf-8", newline="\n")

    manifest = {
        "dataset_revision": revision,
        "accepted_count": len(accepted),
        "rejection_count": len(rejections),
        "source_counts": dict(sorted(Counter(record.source for record in accepted).items())),
        "capability_counts": dict(
            sorted(Counter(record.capability for record in accepted).items())
        ),
        "accepted_sha256": _sha256(accepted_text),
        "rejections_sha256": _sha256(rejections_text),
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
    _write_outputs(
        args.output,
        accepted,
        [
            rejection.to_dict()
            for rejection in sorted(
                [*raw_rejections, *oasst_rejections],
                key=lambda item: (item.id, item.reason),
            )
        ],
        revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
