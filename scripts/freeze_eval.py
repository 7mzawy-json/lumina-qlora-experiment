from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumina_experiment.contracts import canonical_json  # noqa: E402
from lumina_experiment.isolation import freeze_files  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze Lumina evaluation assets")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-freeze", action="store_true")
    return parser


def _case_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"case path does not exist: {path}")
    cases = sorted(path.rglob("*.jsonl"))
    if not cases:
        raise ValueError(f"no JSONL case files found under {path}")
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    previous_hash: str | None = None
    if args.output.exists():
        if not args.replace_freeze:
            raise FileExistsError(
                f"{args.output} already exists; pass --replace-freeze to replace it"
            )
        previous_hash = args.output.read_text(encoding="utf-8").strip()

    paths = [*_case_paths(args.cases), args.sources]
    digest = freeze_files(paths, args.output)
    if previous_hash is not None:
        history_path = args.output.parent / "freeze-history.jsonl"
        history_record = {
            "previous_hash": previous_hash,
            "new_hash": digest,
            "replaced_at_utc": datetime.now(UTC).isoformat(),
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as history:
            history.write(f"{canonical_json(history_record)}\n")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
