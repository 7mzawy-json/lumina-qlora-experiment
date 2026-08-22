"""Fail closed when a tracked file contains a concrete credential-shaped value."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

SCANNER = "lumina-tracked-secret-scan"
VERSION = 1
TOKEN_PATTERNS = {
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
}
ASSIGNMENT_PATTERN = re.compile(r"^\s*(HF_TOKEN|WANDB_API_KEY)\s*=\s*(.*?)\s*$")


def intended_paths(root: Path) -> list[Path]:
    """Return tracked plus non-ignored untracked files for the intended commit set."""

    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        check=True,
    )
    paths = {root / Path(item) for item in completed.stdout.decode("utf-8").split("\0") if item}
    return sorted(paths, key=lambda path: path.as_posix())


def _placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'")
    return not normalized or normalized.startswith(("${", "<", "REPLACE", "YOUR_"))


def scan_paths(root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    """Return deterministic findings for concrete values in the supplied tracked files."""

    findings: list[dict[str, object]] = []
    resolved_root = root.resolve()
    for path in sorted(
        (candidate.resolve() for candidate in paths), key=lambda item: item.as_posix()
    ):
        try:
            relative = path.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ValueError(f"tracked file escaped scan root: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"tracked file is missing: {relative}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            for kind, pattern in TOKEN_PATTERNS.items():
                if pattern.search(line):
                    findings.append({"kind": kind, "line": line_number, "path": relative})
            assignment = ASSIGNMENT_PATTERN.match(line)
            if assignment and not _placeholder(assignment.group(2)):
                kind = (
                    "assigned_hf_token"
                    if assignment.group(1) == "HF_TOKEN"
                    else "assigned_wandb_key"
                )
                findings.append({"kind": kind, "line": line_number, "path": relative})
    return sorted(
        findings, key=lambda item: (str(item["path"]), int(item["line"]), str(item["kind"]))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = intended_paths(root)
    evidence = {
        "findings": scan_paths(root, paths),
        "scanner": SCANNER,
        "scanned_files": len(paths),
        "version": VERSION,
    }
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 1 if evidence["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
