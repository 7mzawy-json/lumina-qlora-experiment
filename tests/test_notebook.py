import re
import subprocess
import sys
from pathlib import Path

import nbformat

from scripts.build_notebook import build_notebook

NOTEBOOK_PATH = Path("notebooks/lumina_qlora_demo.ipynb")
GPU_REQUIREMENTS_PATH = Path("requirements-gpu.txt")


def test_notebook_has_ordered_evidence_gates() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    headings = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]

    assert headings == [
        "# Lumina QLoRA Demonstrative Experiment",
        "## 1. Runtime and dependency verification",
        "## 2. Repository and frozen-data verification",
        "## 3. Small-checkpoint smoke test",
        "## 4. Balanced 8B pilot baseline",
        "## 5. Bounded rank-16 QLoRA pilot training",
        "## 6. Pilot adapted-model generation",
        "## 7. Export and local verification",
        "## 8. Optional rank-8 ablation",
    ]


def test_notebook_downloads_public_repository_without_github_credentials() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert 'userdata.get("GITHUB_TOKEN")' not in source
    assert '"Authorization"' not in source
    assert "github_token" not in source
    assert "RUN_8B_PILOT = False" in source
    assert "generate_condition(" in source
    assert "train_adapter(" in source
    assert not re.search(r"https?://[^\s/@:]+:[^\s/@]+@", source)
    assert not re.search(r"(?:ghp_|github_pat_|hf_)[A-Za-z0-9]{20,}", source)


def test_notebook_requires_only_public_repo_coordinates_and_hf_token() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    for secret_name in ("GITHUB_REPOSITORY", "GITHUB_COMMIT", "HF_TOKEN"):
        assert f'userdata.get("{secret_name}")' in source
    assert 'userdata.get("GITHUB_TOKEN")' not in source
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
    assert "ablation_run = train_adapter(" in ablation
    assert "ablation_config, pilot_datasets, model_revision=pilot_model_revision" in ablation
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


def test_gpu_requirements_include_python_313_compatibility_pins_under_no_deps_install() -> None:
    requirements = GPU_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    assert "jedi==0.20.0" in requirements
    assert "pyarrow==25.0.0" in requirements
    assert (
        '[sys.executable, "-m", "pip", "install", "--no-deps", "-r", "requirements-gpu.txt"]'
        in source
    )


def test_notebook_records_every_direct_gpu_requirement_version_in_manifests() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    expected_packages = (
        "accelerate",
        "bitsandbytes",
        "datasets",
        "jedi",
        "peft",
        "pyarrow",
        "transformers",
        "trl",
        "torch",
    )
    package_versions_source = source[
        source.index("package_versions = {") : source.index('print({"gpu"')
    ]
    for package in expected_packages:
        assert f'"{package}"' in package_versions_source


def test_runtime_cell_exposes_checkout_src_to_current_python_process(tmp_path: Path) -> None:
    package_directory = tmp_path / "src" / "lumina_experiment"
    package_directory.mkdir(parents=True)
    (package_directory / "__init__.py").write_text(
        'BOOTSTRAP_MARKER = "current-process-import-works"\n', encoding="utf-8"
    )

    runtime_source = build_notebook().cells[2].source
    pip_check = 'subprocess.run([sys.executable, "-m", "pip", "check"], check=True)\n'
    torch_check = '\ntorch_after = {"version": torch.__version__, "cuda": torch.version.cuda}'
    bootstrap_source = runtime_source.split(pip_check, maxsplit=1)[1].split(
        torch_check, maxsplit=1
    )[0]
    probe = f"""import sys
from pathlib import Path

workspace = Path({str(tmp_path)!r})
{bootstrap_source}
from lumina_experiment import BOOTSTRAP_MARKER
assert BOOTSTRAP_MARKER == "current-process-import-works"
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


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
    assert "pilot_model_revision = resolve_hub_revision(pilot_config.model_id)" in source
    assert "model_revision=smoke_model_revision" in source
    assert "model_revision=pilot_model_revision" in source


def test_notebook_runs_balanced_pilot_and_exports_its_training_manifest() -> None:
    source = "\n".join(cell.source for cell in build_notebook().cells if cell.cell_type == "code")

    assert 'Path("configs/pilot-r16.yaml")' in source
    assert "select_balanced_cases(" in source
    assert "limit=pilot_config.eval_case_limit" in source
    assert "pilot_run = train_adapter(" in source
    assert "canonical_json(pilot_run)" in source
    assert 'Path("results/raw/pilot-r16/run-manifest.json")' in source
    assert "adapted_started = time.monotonic()" in source
    assert "runtime_seconds=time.monotonic() - adapted_started" in source
    assert 'runtime_seconds=float(pilot_run.artifacts["runtime_seconds"])' not in source
