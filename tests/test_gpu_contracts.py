import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumina_experiment.config import load_experiment_config
from lumina_experiment.contracts import Generation, GpuProbe
from lumina_experiment.gpu import (
    _validation_records,
    build_condition_manifest,
    build_generation_kwargs,
    probe_gpu,
    resolve_runtime_config,
    select_validation_checkpoint,
    verify_frozen_dataset_split,
    verify_frozen_evaluation,
)
from lumina_experiment.isolation import freeze_files


def primary_config():
    return load_experiment_config(Path("configs/primary-r16.yaml"))


def gpu(name: str, bf16: bool, vram_gb: float) -> GpuProbe:
    return GpuProbe(
        name=name,
        vram_gb=vram_gb,
        bf16_supported=bf16,
        torch_version="2.7.0",
        cuda_version="12.6",
    )


def test_t4_fallback_is_recorded_not_silent() -> None:
    adjusted = resolve_runtime_config(
        primary_config(), gpu(name="Tesla T4", bf16=False, vram_gb=15.0)
    )

    assert adjusted.compute_dtype == "float16"
    assert adjusted.max_sequence_length == 1024
    assert adjusted.deviations == (
        "BF16 unsupported; used FP16",
        "VRAM below 20 GB; reduced max sequence length from 2048 to 1024",
    )


def test_probe_gpu_does_not_treat_emulated_t4_bf16_as_native(monkeypatch) -> None:
    class T4Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_properties(_device_index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=15 * 1024**3, major=7, minor=5)

        @staticmethod
        def get_device_name(_device_index: int) -> str:
            return "Tesla T4"

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu128",
        cuda=T4Cuda(),
        version=SimpleNamespace(cuda="12.8"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    detected = probe_gpu()

    assert detected.bf16_supported is False


def test_generation_config_is_deterministic() -> None:
    generation = build_generation_kwargs(primary_config())

    assert generation == {"do_sample": False, "temperature": 0.0, "max_new_tokens": 512}


def test_checkpoint_selection_uses_only_validation_metrics() -> None:
    checkpoint, evidence = select_validation_checkpoint(
        {
            "checkpoint-epoch-1": {
                "validation_loss": 0.42,
                "validation_instruction_following": 0.81,
            },
            "checkpoint-epoch-2": {
                "validation_loss": 0.38,
                "validation_instruction_following": 0.80,
            },
        }
    )

    assert checkpoint == "checkpoint-epoch-2"
    assert evidence == {
        "validation_loss": 0.38,
        "validation_instruction_following": 0.80,
    }


def test_checkpoint_selection_rejects_evaluation_metric_names() -> None:
    with pytest.raises(ValueError, match="validation"):
        select_validation_checkpoint({"checkpoint-epoch-1": {"eval_loss": 0.42}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_checkpoint_selection_rejects_non_finite_validation_metrics(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        select_validation_checkpoint({"checkpoint-epoch-1": {"validation_loss": value}})


def test_validation_history_is_paired_with_its_actual_checkpoint(tmp_path: Path) -> None:
    records = _validation_records(
        [
            {"step": 10, "eval_loss": 0.42},
            {"step": 20, "eval_loss": 0.38, "eval_accuracy": 0.8},
        ],
        tmp_path,
    )

    assert records == {
        str(tmp_path / "checkpoint-10"): {"validation_loss": 0.42},
        str(tmp_path / "checkpoint-20"): {"validation_loss": 0.38, "validation_accuracy": 0.8},
    }


def test_frozen_evaluation_is_recomputed_and_verified(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case_path = cases_dir / "instruction.jsonl"
    case_path.write_text('{"id":"instruction-001","prompt":"Return 4."}\n', encoding="utf-8")
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text("sources: []\n", encoding="utf-8")
    freeze_path = tmp_path / "FROZEN.sha256"
    expected = freeze_files([case_path, sources_path], freeze_path)

    assert verify_frozen_evaluation(cases_dir, sources_path, freeze_path) == expected
    case_path.write_text('{"id":"instruction-001","prompt":"Return 5."}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="freeze"):
        verify_frozen_evaluation(cases_dir, sources_path, freeze_path)


def test_frozen_dataset_artifacts_are_recomputed_and_verified(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"id":"train-001"}\n', encoding="utf-8")
    validation.write_text('{"id":"validation-001"}\n', encoding="utf-8")
    manifest = {
        "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_frozen_dataset_split(tmp_path) == manifest
    validation.write_text('{"id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        verify_frozen_dataset_split(tmp_path)


def test_condition_manifest_rejects_unchecked_caller_model_revision() -> None:
    generation = Generation(
        case_id="instruction-001",
        condition="base",
        output="4",
        evidence_state="smoke_test_verified",
        metadata={"model_revision": "a" * 40},
    )

    with pytest.raises(ValueError, match="generation revision"):
        build_condition_manifest(
            primary_config(),
            [generation],
            gpu=gpu("Tesla T4", False, 15.0),
            runtime_seconds=1.0,
            peak_vram_gb=1.0,
            model_revision="b" * 40,
            package_versions={"torch": "2.7.0"},
        )


@pytest.mark.parametrize("invalid_metadata", [{}, {"model_revision": 123}])
def test_condition_manifest_requires_a_valid_revision_on_every_generation(
    invalid_metadata: dict[str, object],
) -> None:
    generations = [
        Generation(
            case_id="instruction-001",
            condition="base",
            output="4",
            evidence_state="smoke_test_verified",
            metadata={"model_revision": "a" * 40},
        ),
        Generation(
            case_id="instruction-002",
            condition="base",
            output="5",
            evidence_state="smoke_test_verified",
            metadata=invalid_metadata,
        ),
    ]

    with pytest.raises(ValueError, match="generation revision"):
        build_condition_manifest(
            primary_config(),
            generations,
            gpu=gpu("Tesla T4", False, 15.0),
            runtime_seconds=1.0,
            peak_vram_gb=1.0,
            package_versions={"torch": "2.7.0"},
        )
