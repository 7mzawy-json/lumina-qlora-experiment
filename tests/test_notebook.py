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


def test_notebook_requires_all_colab_secrets_and_wires_hf_token_only_in_memory() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    for secret_name in ("GITHUB_REPOSITORY", "GITHUB_COMMIT", "GITHUB_TOKEN", "HF_TOKEN"):
        assert f'userdata.get("{secret_name}")' in source
    assert 'hf_token = userdata.get("HF_TOKEN")' in source
    assert "not hf_token or not hf_token.strip()" in source
    assert 'os.environ["HF_TOKEN"] = hf_token' in source
    assert "del hf_token" in source
    assert "HUGGINGFACE_HUB_TOKEN" not in source
    assert (
        "hf_token"
        not in source[
            source.index("del hf_token") + len("del hf_token") : source.index(
                "RUN_ABLATION = False"
            )
        ]
    )


def test_notebook_uses_only_requested_verified_immutable_commit_for_archive_download() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    assert 're.fullmatch(r"[0-9a-fA-F]{40}", requested_commit)' in source
    assert 'requested_commit = userdata.get("GITHUB_COMMIT")' in source
    assert (
        'f"https://api.github.com/repos/{github_repository}/commits/{requested_commit}"' in source
    )
    assert 'verified_commit = commit_response.json()["sha"]' in source
    assert "verified_commit != requested_commit" in source
    assert 'f"https://api.github.com/repos/{github_repository}/zipball/{verified_commit}"' in source
    assert "/commits/HEAD" not in source
    assert "/zipball/{repository_commit}" not in source


def test_notebook_clears_hf_token_environment_before_export_archive_is_constructed() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    cleanup = 'os.environ.pop("HF_TOKEN", None)'
    archive = 'archive_path = Path("/content/lumina-results.zip")'
    assert cleanup in source
    assert source.index(cleanup) < source.index(archive)
    ablation = "RUN_ABLATION = False"
    assert "hf_token" not in source[source.index(archive) : source.index(ablation)]


def test_notebook_rehydrates_hf_token_only_in_enabled_ablation_and_finally_clears_it() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    ablation = source[source.index("RUN_ABLATION = False") :]
    assert "RUN_ABLATION = False" in ablation
    assert 'if RUN_ABLATION:\n    ablation_hf_token = userdata.get("HF_TOKEN")' in ablation
    assert "if not ablation_hf_token or not ablation_hf_token.strip():" in ablation
    assert 'os.environ["HF_TOKEN"] = ablation_hf_token' in ablation
    assert "del ablation_hf_token" in ablation
    assert "try:" in ablation
    assert "ablation_run = train_adapter(ablation_config, primary_datasets)" in ablation
    assert 'finally:\n        os.environ.pop("HF_TOKEN", None)' in ablation
    assert ablation.index("del ablation_hf_token") < ablation.index("try:")
    assert ablation.index("try:") < ablation.index("finally:")


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


def test_notebook_imports_the_verified_processed_archive_before_smoke_work() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    assert "lumina-processed.zip" in source
    required_processed = (
        'required_processed = {"accepted.jsonl", "train.jsonl", '
        '"validation.jsonl", "manifest.json"}'
    )
    assert required_processed in source
    assert "shutil.copytree(processed_staging, processed_destination)" in source
    assert source.index("processed_destination") < source.index("smoke_datasets =")


def test_notebook_preserves_drive_archive_and_deletes_only_disposable_copy() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    persistent_archive = (
        'persistent_processed_archive = Path("/content/drive/MyDrive/lumina-processed.zip")'
    )
    assert persistent_archive in source
    assert 'processed_archive_copy = Path("/content/lumina-processed-upload.zip")' in source
    assert "shutil.copy2(persistent_processed_archive, processed_archive_copy)" in source
    assert "processed_archive_copy.unlink(missing_ok=True)" in source
    assert "persistent_processed_archive.unlink" not in source


def test_notebook_resolves_each_model_once_and_passes_immutable_revisions() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    assert "smoke_model_revision = resolve_hub_revision(smoke_config.model_id)" in source
    assert "model_revision=smoke_model_revision" in source
    assert "model_revision=primary_model_revision" in source
