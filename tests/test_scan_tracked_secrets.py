from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.scan_tracked_secrets import intended_paths, main, scan_paths


def test_scanner_detects_assigned_token_but_not_documented_pattern(tmp_path: Path) -> None:
    """Only concrete credentials—not scanner documentation—are a finding."""

    documented = tmp_path / "docs.md"
    documented.write_text("Use the pattern hf_[A-Za-z0-9]{20,}.\n", encoding="utf-8")
    assigned = tmp_path / "settings.env"
    hf_token = "hf_" + "abcdefghijklmnopqrstuvwx"
    wandb_key = "w" * 24
    assigned.write_text(f"HF_TOKEN={hf_token}\nWANDB_API_KEY={wandb_key}\n", encoding="utf-8")

    findings = scan_paths(tmp_path, [documented, assigned])

    assert findings == [
        {"kind": "assigned_hf_token", "line": 1, "path": "settings.env"},
        {"kind": "hf_token", "line": 1, "path": "settings.env"},
        {"kind": "assigned_wandb_key", "line": 2, "path": "settings.env"},
    ]


def test_scanner_main_emits_canonical_failure_evidence(tmp_path: Path, monkeypatch, capsys) -> None:
    """A finding produces deterministic JSON and a nonzero exit code."""

    source = tmp_path / "tracked.py"
    github_token = "ghp_" + "abcdefghijklmnopqrstuvwx"
    source.write_text(f"token = '{github_token}'\n", encoding="utf-8")
    monkeypatch.setattr("scripts.scan_tracked_secrets.intended_paths", lambda _: [source])

    assert main(["--root", str(tmp_path)]) == 1

    evidence = json.loads(capsys.readouterr().out)
    assert evidence == {
        "findings": [{"kind": "github_token", "line": 1, "path": "tracked.py"}],
        "scanner": "lumina-tracked-secret-scan",
        "scanned_files": 1,
        "version": 1,
    }


def test_intended_paths_include_untracked_and_exclude_ignored_files(tmp_path: Path) -> None:
    """The pre-commit candidate set includes new code but excludes generated/secret files."""

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
    untracked = tmp_path / "new_script.py"
    untracked.write_text("value = 2\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.env\n", encoding="utf-8")
    ignored = tmp_path / "ignored.env"
    ignored.write_text("HF_TOKEN=placeholder\n", encoding="utf-8")

    names = {path.name for path in intended_paths(tmp_path)}
    assert {"new_script.py", "tracked.py"}.issubset(names)
    assert "ignored.env" not in names


def test_scanner_fixture_source_is_clean() -> None:
    """The test fixture itself does not embed a complete credential-shaped token."""

    root = Path.cwd()
    assert scan_paths(root, [Path(__file__)]) == []
