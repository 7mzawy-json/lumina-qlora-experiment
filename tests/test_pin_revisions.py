from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pin_revisions import pin_revisions, resolve_hub_revision


def revision_inputs(tmp_path: Path) -> tuple[list[Path], Path, Path, dict[str, str]]:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    primary = config_dir / "primary.yaml"
    primary.write_text("model_id: Qwen/Qwen3-8B-Base\n", encoding="utf-8")
    smoke = config_dir / "smoke.yaml"
    smoke.write_text("model_id: Qwen/Qwen3-0.6B-Base\n", encoding="utf-8")
    data = config_dir / "data.yaml"
    data.write_text("dataset_id: OpenAssistant/oasst1\n", encoding="utf-8")
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        "datasets:\n  - name: google/IFEval\n    revision: " + "a" * 40 + "\n",
        encoding="utf-8",
    )
    resolved = {
        "Qwen/Qwen3-8B-Base": "b" * 40,
        "Qwen/Qwen3-0.6B-Base": "c" * 40,
        "OpenAssistant/oasst1": "d" * 40,
    }
    return [primary, smoke, data], sources, config_dir / "revisions.yaml", resolved


def resolver(values: dict[str, str]):
    def resolve_revision(repository: str, repository_type: str) -> str:
        assert repository_type in {"models", "datasets"}
        return values[repository]

    return resolve_revision


def test_hub_resolver_uses_model_api_for_a_configured_non_qwen_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future configured model must not be looked up through the datasets API."""

    observed_urls: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"sha": "a" * 40}

    def fake_get(url: str, **_: object) -> Response:
        observed_urls.append(url)
        return Response()

    monkeypatch.setattr("scripts.pin_revisions.httpx.get", fake_get)

    assert resolve_hub_revision("Acme/Instruction-Model") == "a" * 40
    assert observed_urls == ["https://huggingface.co/api/models/Acme/Instruction-Model"]


def test_pin_revisions_preserves_frozen_evaluation_anchor_and_refuses_drift(
    tmp_path: Path,
) -> None:
    """A changed remote SHA must not silently move a previously frozen anchor."""

    configs, sources, output, resolved = revision_inputs(tmp_path)

    pin_revisions(
        output=output,
        config_paths=configs,
        source_manifest=sources,
        resolve_revision=resolver(resolved),
    )

    rendered = output.read_text(encoding="utf-8")
    assert "google/IFEval: " + "a" * 40 in rendered
    assert "Qwen/Qwen3-8B-Base: " + "b" * 40 in rendered

    resolved["Qwen/Qwen3-8B-Base"] = "f" * 40
    with pytest.raises(ValueError, match="--replace"):
        pin_revisions(
            output=output,
            config_paths=configs,
            source_manifest=sources,
            resolve_revision=resolver(resolved),
        )


def test_pin_revisions_identical_rerun_does_not_rewrite_output(tmp_path: Path) -> None:
    """A stable complete mapping remains byte-for-byte and timestamp stable."""

    configs, sources, output, resolved = revision_inputs(tmp_path)
    pin_revisions(
        output=output,
        config_paths=configs,
        source_manifest=sources,
        resolve_revision=resolver(resolved),
    )
    before = (output.read_bytes(), output.stat().st_mtime_ns)

    pin_revisions(
        output=output,
        config_paths=configs,
        source_manifest=sources,
        resolve_revision=resolver(resolved),
    )

    assert (output.read_bytes(), output.stat().st_mtime_ns) == before


def test_pin_revisions_replace_updates_a_changed_current_pin(tmp_path: Path) -> None:
    """An explicitly requested replacement records the newly resolved immutable SHA."""

    configs, sources, output, resolved = revision_inputs(tmp_path)
    pin_revisions(
        output=output,
        config_paths=configs,
        source_manifest=sources,
        resolve_revision=resolver(resolved),
    )
    resolved["Qwen/Qwen3-8B-Base"] = "e" * 40

    updated = pin_revisions(
        output=output,
        config_paths=configs,
        source_manifest=sources,
        resolve_revision=resolver(resolved),
        replace=True,
    )

    assert updated["Qwen/Qwen3-8B-Base"] == "e" * 40


def test_pin_revisions_does_not_mask_a_resolver_type_error(tmp_path: Path) -> None:
    """A resolver failure must propagate instead of being misread as an old signature."""

    configs, sources, output, _ = revision_inputs(tmp_path)
    calls = 0

    def broken_resolver(repository: str, repository_type: str) -> str:
        nonlocal calls
        assert repository and repository_type
        calls += 1
        raise TypeError("Hub client decoding error")

    with pytest.raises(TypeError, match="Hub client decoding error"):
        pin_revisions(
            output=output,
            config_paths=configs,
            source_manifest=sources,
            resolve_revision=broken_resolver,
        )
    assert calls == 1
