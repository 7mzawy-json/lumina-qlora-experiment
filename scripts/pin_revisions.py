"""Resolve and fail-close immutable Hugging Face revisions for the experiment."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import httpx
import yaml

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_KEYS = ("model_id", "dataset_id")
REPOSITORY_CONFIGS = (
    Path("configs/primary-r16.yaml"),
    Path("configs/smoke.yaml"),
    Path("configs/data.yaml"),
)
SOURCE_MANIFEST = Path("data/eval/sources.yaml")


def _valid_sha(value: object) -> str:
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError("Hugging Face did not return a 40-character immutable commit SHA")
    return value


def _load_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return loaded


def resolve_hub_revision(
    repository: str, revision: str | None = None, repository_type: str = "models"
) -> str:
    """Resolve a public Hub repository (or verify a supplied commit) via Hub APIs."""

    suffix = f"/revision/{revision}" if revision else ""
    if repository_type not in {"models", "datasets"}:
        raise ValueError("repository_type must be models or datasets")
    url = f"https://huggingface.co/api/{repository_type}/{repository}{suffix}"
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected Hugging Face API payload for {repository}")
    resolved = _valid_sha(payload.get("sha"))
    if revision and resolved != revision:
        raise ValueError(f"Hugging Face revision verification mismatch for {repository}")
    return resolved


def resolve_current_hub_revision(repository: str, repository_type: str) -> str:
    """Resolve a current revision through the stable typed pinning interface."""

    return resolve_hub_revision(repository, repository_type=repository_type)


def _configured_repositories(paths: Iterable[Path]) -> dict[str, str]:
    repositories: dict[str, str] = {}
    for path in paths:
        config = _load_mapping(path)
        for key in REPOSITORY_KEYS:
            value = config.get(key)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} must be a non-empty repository ID in {path}")
                repositories[value] = "models" if key == "model_id" else "datasets"
    return repositories


def _evaluation_repositories(path: Path) -> dict[str, str]:
    manifest = _load_mapping(path)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"datasets must be a list in {path}")
    anchors: dict[str, str] = {}
    for entry in datasets:
        if not isinstance(entry, Mapping):
            raise ValueError(f"every evaluation dataset must be a mapping in {path}")
        repository = entry.get("name")
        revision = entry.get("revision")
        if not isinstance(repository, str) or not repository.strip():
            raise ValueError(f"evaluation dataset name is invalid in {path}")
        anchors[repository] = _valid_sha(revision)
    return anchors


def _write_revisions(path: Path, revisions: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(dict(sorted(revisions.items())), allow_unicode=True, sort_keys=True)
    path.write_text(rendered, encoding="utf-8", newline="\n")


def pin_revisions(
    *,
    output: Path,
    config_paths: Iterable[Path],
    source_manifest: Path,
    resolve_revision: Callable[[str, str], str],
    verify_revision: Callable[[str, str], str] | None = None,
    replace: bool = False,
) -> dict[str, str]:
    """Return and persist a complete immutable mapping without silent drift."""

    evaluation_anchors = _evaluation_repositories(source_manifest)
    desired = {
        repository: _valid_sha(resolve_revision(repository, repository_type))
        for repository, repository_type in _configured_repositories(config_paths).items()
    }
    for repository, revision in evaluation_anchors.items():
        if verify_revision is not None:
            verified = _valid_sha(verify_revision(repository, revision))
            if verified != revision:
                raise ValueError(f"Hugging Face revision verification mismatch for {repository}")
        desired[repository] = revision

    existing = _load_mapping(output) if output.exists() else {}
    pins = {str(repository): _valid_sha(revision) for repository, revision in existing.items()}
    conflicts = {
        repository: (pins[repository], revision)
        for repository, revision in desired.items()
        if repository in pins and pins[repository] != revision
    }
    if conflicts and not replace:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"existing pin differs for {names}; rerun with --replace to update it")
    merged = {**pins, **desired}
    if merged != pins or not output.exists():
        _write_revisions(output, merged)
    return dict(sorted(merged.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--config", action="append", type=Path, dest="configs")
    parser.add_argument("--sources", type=Path, default=SOURCE_MANIFEST)
    args = parser.parse_args(argv)
    config_paths = args.configs or list(REPOSITORY_CONFIGS)
    pins = pin_revisions(
        output=args.output,
        config_paths=config_paths,
        source_manifest=args.sources,
        resolve_revision=resolve_current_hub_revision,
        verify_revision=lambda repository, revision: resolve_hub_revision(
            repository, revision, "datasets"
        ),
        replace=args.replace,
    )
    print(yaml.safe_dump(pins, allow_unicode=True, sort_keys=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
