"""Build the tracked Colab notebook without embedding credentials or outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "lumina_qlora_demo.ipynb"

HEADINGS = (
    "# Lumina QLoRA Demonstrative Experiment",
    "## 1. Runtime and dependency verification",
    "## 2. Repository and frozen-data verification",
    "## 3. Small-checkpoint smoke test",
    "## 4. Untouched 8B baseline",
    "## 5. Primary rank-16 QLoRA training",
    "## 6. Adapted-model generation",
    "## 7. Export and local verification",
    "## 8. Optional rank-8 ablation",
)


def _code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source)


def build_notebook() -> nbformat.NotebookNode:
    """Return a deterministic, output-free Colab notebook using package interfaces."""

    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(HEADINGS[0]),
        nbformat.v4.new_markdown_cell(HEADINGS[1]),
        _code(
            """import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from google.colab import files, userdata
import torch

torch_before = {"version": torch.__version__, "cuda": torch.version.cuda}

github_repository = os.environ.get("GITHUB_REPOSITORY") or userdata.get("GITHUB_REPOSITORY")
github_token = userdata.get("GITHUB_TOKEN")
if not github_repository or not github_token:
    raise RuntimeError("Set GITHUB_REPOSITORY and GITHUB_TOKEN in Colab Secrets before continuing.")

headers = {"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github+json"}
commit_response = requests.get(
    f"https://api.github.com/repos/{github_repository}/commits/HEAD", headers=headers, timeout=30
)
commit_response.raise_for_status()
repository_commit = commit_response.json()["sha"]
if (
    not isinstance(repository_commit, str)
    or len(repository_commit) != 40
    or any(character not in "0123456789abcdef" for character in repository_commit.casefold())
):
    raise RuntimeError("GitHub did not return an immutable repository commit")
archive_response = requests.get(
    f"https://api.github.com/repos/{github_repository}/zipball/{repository_commit}",
    headers=headers,
    timeout=120,
)
archive_response.raise_for_status()

workspace = Path("/content/lumina")
shutil.rmtree(workspace, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
    roots = set()
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError("refusing an unsafe repository archive member")
        roots.add(member_path.parts[0])
    if len(roots) != 1:
        raise RuntimeError("repository archive must contain exactly one root directory")
    archive.extractall(workspace.parent)
extracted = workspace.parent / roots.pop()
extracted.rename(workspace)

del github_token, headers, archive_response, commit_response
os.chdir(workspace)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "-r", "requirements-gpu.txt"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

torch_after = {"version": torch.__version__, "cuda": torch.version.cuda}
if torch_after != torch_before:
    raise RuntimeError("dependency installation changed the Colab CUDA PyTorch build")

from importlib.metadata import version

from lumina_experiment.gpu import probe_gpu

gpu = probe_gpu()
package_versions = {
    package: version(package)
    for package in ("torch", "transformers", "trl", "peft", "bitsandbytes")
}
print({"gpu": gpu.to_dict(), "python": sys.version.split()[0], "packages": package_versions})"""
        ),
        nbformat.v4.new_markdown_cell(HEADINGS[2]),
        _code(
            """USE_DRIVE_PROCESSED_ARCHIVE = False
if USE_DRIVE_PROCESSED_ARCHIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    persistent_processed_archive = Path("/content/drive/MyDrive/lumina-processed.zip")
    if not persistent_processed_archive.is_file():
        raise FileNotFoundError(
            f"processed-data archive is missing: {persistent_processed_archive}"
        )
    processed_archive_copy = Path("/content/lumina-processed-upload.zip")
    shutil.copy2(persistent_processed_archive, processed_archive_copy)
else:
    uploaded = files.upload()
    source_processed_archive = workspace / "lumina-processed.zip"
    if source_processed_archive.name not in uploaded:
        raise RuntimeError("Upload the exact local lumina-processed.zip package.")
    if not source_processed_archive.is_file():
        raise FileNotFoundError(f"processed-data archive is missing: {source_processed_archive}")
    processed_archive_copy = Path("/content/lumina-processed-upload.zip")
    shutil.copy2(source_processed_archive, processed_archive_copy)
    source_processed_archive.unlink(missing_ok=True)
processed_staging = Path("/content/lumina-processed-staging")
shutil.rmtree(processed_staging, ignore_errors=True)
with zipfile.ZipFile(processed_archive_copy) as archive:
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError("refusing an unsafe processed-data archive member")
    archive.extractall(processed_staging)
required_processed = {"accepted.jsonl", "train.jsonl", "validation.jsonl", "manifest.json"}
if not required_processed.issubset({item.name for item in processed_staging.iterdir()}):
    raise RuntimeError("processed-data archive is missing required frozen artifacts")
processed_destination = workspace / "data" / "processed"
if processed_destination.exists():
    raise RuntimeError("refusing to replace an existing processed-data directory")
shutil.copytree(processed_staging, processed_destination)
processed_archive_copy.unlink(missing_ok=True)"""
        ),
        _code(
            """from lumina_experiment.config import load_experiment_config
from lumina_experiment.gpu import (
    resolve_hub_revision,
    verify_frozen_dataset_split,
    verify_frozen_evaluation,
)

frozen_hash = verify_frozen_evaluation()
dataset_hashes = verify_frozen_dataset_split(Path("data/processed"))
primary_config = load_experiment_config(Path("configs/primary-r16.yaml"))
smoke_config = load_experiment_config(Path("configs/smoke.yaml"))
primary_model_revision = resolve_hub_revision(primary_config.model_id)
smoke_model_revision = resolve_hub_revision(smoke_config.model_id)
print(
    {
        "repository_commit": repository_commit,
        "evaluation_hash": frozen_hash,
        "dataset_hashes": dataset_hashes,
        "model_revision": primary_model_revision,
    }
)"""
        ),
        nbformat.v4.new_markdown_cell(HEADINGS[3]),
        _code(
            """from lumina_experiment.gpu import (
    build_condition_manifest,
    export_condition_manifest,
    export_generations,
    generate_condition,
    load_frozen_dataset_split,
    train_adapter,
)
from lumina_experiment.scoring import load_eval_directory

smoke_datasets = load_frozen_dataset_split(Path("data/processed"))
smoke_cases = load_eval_directory(Path("data/eval/cases"))[: smoke_config.eval_case_limit]
smoke_run = train_adapter(smoke_config, smoke_datasets, model_revision=smoke_model_revision)
smoke_generations = generate_condition(
    smoke_config,
    smoke_cases,
    Path(smoke_run.selected_checkpoint),
    model_revision=smoke_model_revision,
)
export_generations(smoke_generations, Path("results/raw/smoke/generations.jsonl"))
smoke_condition = build_condition_manifest(
    smoke_config,
    smoke_generations,
    gpu=gpu,
    runtime_seconds=float(smoke_run.artifacts["runtime_seconds"]),
    peak_vram_gb=float(smoke_run.artifacts["peak_vram_gb"]),
    model_revision=smoke_model_revision,
    package_versions=package_versions,
    adapter_hash=smoke_run.artifacts["adapter_sha256"] or None,
)
export_condition_manifest(smoke_condition, Path("results/raw/smoke/manifest.json"))"""
        ),
        nbformat.v4.new_markdown_cell(HEADINGS[4]),
        _code(
            """RUN_8B = False
if not RUN_8B:
    gate_message = (
        "Change RUN_8B = False to RUN_8B = True only after the smoke artifacts "
        "pass local verification."
    )
    raise RuntimeError(
        gate_message
    )

all_cases = load_eval_directory(Path("data/eval/cases"))
import time
import torch

torch.cuda.reset_peak_memory_stats()
base_started = time.monotonic()
base_generations = generate_condition(
    primary_config, all_cases, model_revision=primary_model_revision
)
export_generations(base_generations, Path("results/raw/base/generations.jsonl"))
base_condition = build_condition_manifest(
    primary_config,
    base_generations,
    gpu=gpu,
    runtime_seconds=time.monotonic() - base_started,
    peak_vram_gb=torch.cuda.max_memory_allocated() / (1024**3),
    model_revision=primary_model_revision,
    package_versions=package_versions,
)
export_condition_manifest(base_condition, Path("results/raw/base/manifest.json"))"""
        ),
        nbformat.v4.new_markdown_cell(HEADINGS[5]),
        _code(
            """primary_datasets = load_frozen_dataset_split(Path("data/processed"))
primary_run = train_adapter(
    primary_config, primary_datasets, model_revision=primary_model_revision
)
if primary_run.selected_checkpoint is None or not primary_run.validation_selection:
    raise RuntimeError("training did not produce validation-only checkpoint selection evidence")"""
        ),
        nbformat.v4.new_markdown_cell(HEADINGS[6]),
        _code(
            """adapted_generations = generate_condition(
    primary_config,
    all_cases,
    Path(primary_run.selected_checkpoint),
    model_revision=primary_model_revision,
)
if [item.case_id for item in base_generations] != [item.case_id for item in adapted_generations]:
    raise RuntimeError("base and adapter generation case order differs")
export_generations(adapted_generations, Path("results/raw/primary-r16/generations.jsonl"))
primary_condition = build_condition_manifest(
    primary_config,
    adapted_generations,
    gpu=gpu,
    runtime_seconds=float(primary_run.artifacts["runtime_seconds"]),
    peak_vram_gb=float(primary_run.artifacts["peak_vram_gb"]),
    model_revision=primary_model_revision,
    package_versions=package_versions,
    adapter_hash=primary_run.artifacts["adapter_sha256"] or None,
)
export_condition_manifest(primary_condition, Path("results/raw/primary-r16/manifest.json"))"""
        ),
        nbformat.v4.new_markdown_cell(HEADINGS[7]),
        _code(
            """archive_path = Path("/content/lumina-results.zip")
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for candidate in sorted(Path("results").rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix in {".bin", ".pt", ".pth", ".safetensors"}:
            continue
        archive.write(candidate, candidate.as_posix())
files.download(str(archive_path))"""
        ),
        nbformat.v4.new_markdown_cell(HEADINGS[8]),
        _code(
            """RUN_ABLATION = False
if RUN_ABLATION:
    ablation_config = load_experiment_config(Path("configs/ablation-r8.yaml"))
    ablation_run = train_adapter(ablation_config, primary_datasets)
    print({"selected_checkpoint": ablation_run.selected_checkpoint})"""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"lumina-{index:02d}"
    return notebook


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), NOTEBOOK_PATH, version=4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
