import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import lumina_experiment.gpu as gpu_runtime
from lumina_experiment.config import load_experiment_config
from lumina_experiment.contracts import (
    DatasetSplit,
    Generation,
    GpuProbe,
    InstructionRecord,
    Message,
)
from lumina_experiment.gpu import (
    _assert_assistant_generation_mask,
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
from lumina_experiment.reporting import generation_artifact_hash


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


def instruction_record(
    prefix: str,
    index: int,
    *,
    source: str = "test-source",
) -> InstructionRecord:
    return InstructionRecord(
        id=f"{prefix}-{index:03d}",
        source=source,
        license="Apache-2.0",
        capability="general",
        cluster_id=f"{prefix}-cluster-{index:03d}",
        messages=(Message("user", f"Prompt {index}"), Message("assistant", f"Answer {index}")),
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


def test_fp16_amp_restores_trainable_parameters_to_fp32_before_training(
    monkeypatch,
) -> None:
    class FakeData:
        def __init__(self, dtype: str) -> None:
            self.dtype = dtype

        def to(self, dtype: str):
            return FakeData(dtype)

    class FakeParameter:
        def __init__(self, *, requires_grad: bool, dtype: str) -> None:
            self.requires_grad = requires_grad
            self.data = FakeData(dtype)

    trainable = FakeParameter(requires_grad=True, dtype="bfloat16")
    frozen = FakeParameter(requires_grad=False, dtype="bfloat16")

    class FakeModel:
        @staticmethod
        def parameters():
            return (trainable, frozen)

    class FakeTrainer:
        model = FakeModel()
        observed_dtypes = None

        def train(self) -> None:
            self.observed_dtypes = (trainable.data.dtype, frozen.data.dtype)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    trainer = FakeTrainer()
    train_with_runtime_precision = getattr(gpu_runtime, "_train_with_runtime_precision", None)

    assert callable(train_with_runtime_precision)
    train_with_runtime_precision(
        trainer,
        gpu_runtime.RuntimeConfig(
            compute_dtype="float16",
            max_sequence_length=1024,
            deviations=("BF16 unsupported; used FP16",),
        ),
    )

    assert trainer.observed_dtypes == ("float32", "bfloat16")


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


def test_smoke_record_limit_bounds_both_training_splits() -> None:
    datasets = DatasetSplit(
        train=tuple(instruction_record("train", index) for index in range(60)),
        validation=tuple(instruction_record("validation", index) for index in range(60)),
    )
    limiter = getattr(gpu_runtime, "_limit_dataset_split", None)

    assert callable(limiter), "GPU training must consume ExperimentConfig.record_limit"

    limited = limiter(load_experiment_config(Path("configs/smoke.yaml")), datasets)

    assert [record.id for record in limited.train] == [f"train-{index:03d}" for index in range(50)]
    assert [record.id for record in limited.validation] == [
        f"validation-{index:03d}" for index in range(50)
    ]


def test_pilot_source_mixture_is_bounded_and_independent_of_input_order() -> None:
    specific_source = "lumina-demonstration-authored"
    general_source = "OpenAssistant/oasst1"
    train = tuple(
        instruction_record("train-specific", index, source=specific_source) for index in range(100)
    ) + tuple(
        instruction_record("train-general", index, source=general_source) for index in range(300)
    )
    validation = tuple(
        instruction_record("validation-specific", index, source=specific_source)
        for index in range(100)
    ) + tuple(
        instruction_record("validation-general", index, source=general_source)
        for index in range(300)
    )
    config = load_experiment_config(Path("configs/pilot-r16.yaml"))
    limiter = getattr(gpu_runtime, "_limit_dataset_split", None)

    assert callable(limiter)
    selected = limiter(config, DatasetSplit(train=train, validation=validation))
    reversed_selected = limiter(
        config,
        DatasetSplit(train=tuple(reversed(train)), validation=tuple(reversed(validation))),
    )

    assert len(selected.train) == 256
    assert len(selected.validation) == 40
    assert Counter(record.source for record in selected.train) == {
        specific_source: 77,
        general_source: 179,
    }
    assert Counter(record.source for record in selected.validation) == {
        specific_source: 12,
        general_source: 28,
    }
    assert [record.id for record in selected.train] == [
        record.id for record in reversed_selected.train
    ]
    assert [record.id for record in selected.validation] == [
        record.id for record in reversed_selected.validation
    ]

    unexpected_train = train + (
        instruction_record("train-unexpected", 0, source="unexpected/source"),
    )
    with pytest.raises(ValueError, match="unexpected mixture source"):
        limiter(config, DatasetSplit(train=unexpected_train, validation=validation))


def test_assistant_mask_preflight_uses_trl_training_template(monkeypatch) -> None:
    class QwenTokenizer:
        chat_template = "raw-qwen3-template"

        def apply_chat_template(
            self,
            _messages,
            *,
            tokenize,
            return_dict,
            return_assistant_tokens_mask,
            chat_template=None,
        ):
            assert tokenize is True
            assert return_dict is True
            assert return_assistant_tokens_mask is True
            if chat_template == "trl-qwen3-training-template":
                return {"assistant_masks": [0, 1, 1]}
            return {"assistant_masks": [0, 0, 0]}

    chat_template_utils = ModuleType("trl.chat_template_utils")
    chat_template_utils.get_training_chat_template = lambda _tokenizer: (
        "trl-qwen3-training-template"
    )
    chat_template_utils.qwen3_training_chat_template = "unused-qwen3-fallback-template"
    trl = ModuleType("trl")
    trl.chat_template_utils = chat_template_utils
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.chat_template_utils", chat_template_utils)

    tokenizer = QwenTokenizer()

    _assert_assistant_generation_mask(
        tokenizer,
        instruction_record("mask", 0),
        model_id="Other/Compatible-Model",
        model_revision="a" * 40,
    )

    assert tokenizer.chat_template == "trl-qwen3-training-template"


def test_assistant_mask_preflight_handles_pinned_legacy_qwen3_template(monkeypatch) -> None:
    class LegacyQwenTokenizer:
        name_or_path = "Qwen/Qwen3-0.6B-Base"
        chat_template = "legacy-qwen3-template"

        def apply_chat_template(
            self,
            _messages,
            *,
            tokenize,
            return_dict,
            return_assistant_tokens_mask,
            chat_template=None,
        ):
            assert tokenize is True
            assert return_dict is True
            assert return_assistant_tokens_mask is True
            if chat_template == "trl-qwen3-fallback-template":
                return {"assistant_masks": [0, 1, 1]}
            return {"assistant_masks": [0, 0, 0]}

    def reject_legacy_template(_tokenizer):
        raise ValueError(
            "The chat template is not training-compatible (missing prefix-preservation or "
            "`{% generation %}` markers) and patching is not supported for this template. "
            "Please manually modify the chat template for training."
        )

    chat_template_utils = ModuleType("trl.chat_template_utils")
    chat_template_utils.get_training_chat_template = reject_legacy_template
    chat_template_utils.qwen3_training_chat_template = "trl-qwen3-fallback-template"
    trl = ModuleType("trl")
    trl.chat_template_utils = chat_template_utils
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.chat_template_utils", chat_template_utils)
    model_id = "Qwen/Test-Legacy-Base"
    model_revision = "a" * 40
    monkeypatch.setattr(
        gpu_runtime,
        "_QWEN3_LEGACY_TEMPLATE_PROVENANCE",
        {(model_id, model_revision): hashlib.sha256(b"legacy-qwen3-template").hexdigest()},
    )

    tokenizer = LegacyQwenTokenizer()

    _assert_assistant_generation_mask(
        tokenizer,
        instruction_record("mask", 0),
        model_id=model_id,
        model_revision=model_revision,
    )

    assert tokenizer.chat_template == "trl-qwen3-fallback-template"


def test_assistant_mask_preflight_rejects_unsupported_legacy_template(monkeypatch) -> None:
    class UnsupportedTokenizer:
        name_or_path = "Other/Unsupported-Base"
        chat_template = "unsupported-template"

        def apply_chat_template(self, *_args, **_kwargs):
            return {"assistant_masks": [0, 1]}

    def reject_legacy_template(_tokenizer):
        raise ValueError("patching is not supported for this template")

    chat_template_utils = ModuleType("trl.chat_template_utils")
    chat_template_utils.get_training_chat_template = reject_legacy_template
    chat_template_utils.qwen3_training_chat_template = "trl-qwen3-fallback-template"
    trl = ModuleType("trl")
    trl.chat_template_utils = chat_template_utils
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.chat_template_utils", chat_template_utils)
    tokenizer = UnsupportedTokenizer()

    with pytest.raises(ValueError, match="patching is not supported"):
        _assert_assistant_generation_mask(
            tokenizer,
            instruction_record("mask", 0),
            model_id="Other/Unsupported-Base",
            model_revision="b" * 40,
        )

    assert tokenizer.chat_template == "unsupported-template"


def test_assistant_mask_preflight_rejects_changed_allowlisted_qwen3_template(
    monkeypatch,
) -> None:
    class ChangedQwenTokenizer:
        chat_template = "changed-or-corrupted-template"

        def apply_chat_template(self, *_args, **_kwargs):
            return {"assistant_masks": [0, 1]}

    def reject_legacy_template(_tokenizer):
        raise ValueError(
            "The chat template is not training-compatible (missing prefix-preservation or "
            "`{% generation %}` markers) and patching is not supported for this template. "
            "Please manually modify the chat template for training."
        )

    chat_template_utils = ModuleType("trl.chat_template_utils")
    chat_template_utils.get_training_chat_template = reject_legacy_template
    chat_template_utils.qwen3_training_chat_template = "trl-qwen3-fallback-template"
    trl = ModuleType("trl")
    trl.chat_template_utils = chat_template_utils
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.chat_template_utils", chat_template_utils)
    tokenizer = ChangedQwenTokenizer()

    with pytest.raises(ValueError, match="patching is not supported"):
        _assert_assistant_generation_mask(
            tokenizer,
            instruction_record("mask", 0),
            model_id="Qwen/Qwen3-0.6B-Base",
            model_revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        )

    assert tokenizer.chat_template == "changed-or-corrupted-template"


def test_assistant_mask_preflight_rejects_unexpected_helper_error(monkeypatch) -> None:
    class LegacyQwenTokenizer:
        chat_template = "verified-legacy-template"

        def apply_chat_template(self, *_args, **_kwargs):
            return {"assistant_masks": [0, 1]}

    def reject_for_another_reason(_tokenizer):
        raise ValueError("a different TRL validation failed")

    chat_template_utils = ModuleType("trl.chat_template_utils")
    chat_template_utils.get_training_chat_template = reject_for_another_reason
    chat_template_utils.qwen3_training_chat_template = "trl-qwen3-fallback-template"
    trl = ModuleType("trl")
    trl.chat_template_utils = chat_template_utils
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.chat_template_utils", chat_template_utils)
    model_id = "Qwen/Test-Legacy-Base"
    model_revision = "c" * 40
    monkeypatch.setattr(
        gpu_runtime,
        "_QWEN3_LEGACY_TEMPLATE_PROVENANCE",
        {(model_id, model_revision): hashlib.sha256(b"verified-legacy-template").hexdigest()},
    )
    tokenizer = LegacyQwenTokenizer()

    with pytest.raises(ValueError, match="different TRL validation"):
        _assert_assistant_generation_mask(
            tokenizer,
            instruction_record("mask", 0),
            model_id=model_id,
            model_revision=model_revision,
        )

    assert tokenizer.chat_template == "verified-legacy-template"


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


def test_condition_manifest_generation_hash_matches_scored_artifact_contract() -> None:
    generation = Generation(
        case_id="instruction-001",
        condition="base",
        output="4",
        evidence_state="smoke_test_verified",
        metadata={"model_revision": "a" * 40},
    )

    manifest = build_condition_manifest(
        primary_config(),
        [generation],
        gpu=gpu("Tesla T4", False, 15.0),
        runtime_seconds=1.0,
        peak_vram_gb=1.0,
        model_revision="a" * 40,
        package_versions={"torch": "2.7.0"},
    )

    assert manifest.generation_hash == generation_artifact_hash([generation])


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
