import re
from pathlib import Path

import nbformat

from scripts.build_notebook import build_notebook

NOTEBOOK_PATH = Path("notebooks/lumina_qlora_demo.ipynb")


def test_notebook_has_ordered_evidence_gates() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    headings = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]

    assert headings == [
        "# Lumina QLoRA Demonstrative Experiment",
        "## 1. Runtime and dependency verification",
        "## 2. Repository and frozen-data verification",
        "## 3. Small-checkpoint smoke test",
        "## 4. Untouched 8B baseline",
        "## 5. Primary rank-16 QLoRA training",
        "## 6. Adapted-model generation",
        "## 7. Export and local verification",
        "## 8. Optional rank-8 ablation",
    ]


def test_notebook_keeps_github_token_in_headers_and_out_of_artifacts() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert 'userdata.get("GITHUB_TOKEN")' in source
    assert '"Authorization": f"Bearer {github_token}"' in source
    assert "del github_token" in source
    assert "RUN_8B = False" in source
    assert "generate_condition(" in source
    assert "train_adapter(" in source
    assert not re.search(r"https?://[^\s/@:]+:[^\s/@]+@", source)
    assert not re.search(r"(?:ghp_|github_pat_|hf_)[A-Za-z0-9]{20,}", source)


def test_notebook_generator_is_deterministic() -> None:
    assert nbformat.writes(build_notebook(), version=4) == nbformat.writes(
        build_notebook(), version=4
    )


def test_notebook_verifies_freezes_and_preserves_colab_torch() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    assert "verify_frozen_evaluation" in source
    assert "verify_frozen_dataset_split" in source
    assert '"--no-deps"' in source
    assert "torch_before" in source
    assert "torch_after" in source


def test_notebook_resolves_each_model_once_and_passes_immutable_revisions() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    assert "smoke_model_revision = resolve_hub_revision(smoke_config.model_id)" in source
    assert "model_revision=smoke_model_revision" in source
    assert "model_revision=primary_model_revision" in source
