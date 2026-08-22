from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumina_experiment.review import (  # noqa: E402
    build_blind_pairs,
    generations_from_scored_rows,
    write_review_packet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a concealed Lumina A/B review sheet")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--private-key",
        type=Path,
        default=Path("results/raw/review-key.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=42)
    return parser


def _load_rows(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number}: scored row must be an object")
        rows.append(payload)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_rows(args.scores)
    packet = build_blind_pairs(
        generations_from_scored_rows(rows),
        seed=args.seed,
        sample_size=args.sample_size,
    )
    write_review_packet(packet, args.output, args.private_key)
    print(
        f"wrote {len(packet.items)} concealed pairs to {args.output}; "
        f"private key stored at {args.private_key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
