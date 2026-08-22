from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumina_experiment.contracts import Generation, canonical_json  # noqa: E402
from lumina_experiment.reporting import score_generation_records  # noqa: E402
from lumina_experiment.review import is_private_raw_path, pair_conditions  # noqa: E402
from lumina_experiment.scoring import load_eval_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score paired Lumina generations")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("data/eval/cases"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _generation_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "generations.jsonl"
    if candidate.is_file():
        return candidate
    raise ValueError(f"generation input must be JSONL or contain generations.jsonl: {path}")


def _load_generations(path: Path) -> list[Generation]:
    generations: list[Generation] = []
    source = _generation_file(path)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{source}:{line_number}: generation must be an object")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{source}:{line_number}: metadata must be an object")
        generations.append(
            Generation(
                case_id=str(payload.get("case_id", "")),
                condition=str(payload.get("condition", "")),
                output=payload.get("output", ""),
                evidence_state=str(payload.get("evidence_state", "")),
                metadata=dict(metadata),
            )
        )
    return generations


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not is_private_raw_path(args.output):
        raise ValueError("scored rows contain raw generations and must be stored under results/raw")
    cases = load_eval_directory(args.cases)
    base = _load_generations(args.base)
    adapter = _load_generations(args.adapter)
    pair_conditions(base, adapter)
    rows = score_generation_records(cases, [*base, *adapter])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(rows)} scored generation records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
