"""GPU-only QLoRA runtime helpers with lightweight import-time dependencies."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from lumina_experiment.config import ExperimentConfig
from lumina_experiment.contracts import (
    ConditionManifest,
    DatasetSplit,
    EvalCase,
    Generation,
    GpuProbe,
    InstructionRecord,
    RunManifest,
    canonical_json,
)
from lumina_experiment.isolation import freeze_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_EVALUATION_FILE = REPO_ROOT / "data" / "eval" / "FROZEN.sha256"
FROZEN_EVALUATION_CASES = REPO_ROOT / "data" / "eval" / "cases"
FROZEN_EVALUATION_SOURCES = REPO_ROOT / "data" / "eval" / "sources.yaml"
SYSTEM_PROMPT = "You are Lumina, a precise and helpful assistant."
_QWEN3_LEGACY_TEMPLATE_PROVENANCE = {
    (
        "Qwen/Qwen3-0.6B-Base",
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    ): "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
    (
        "Qwen/Qwen3-8B-Base",
        "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    ): "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
}
_TRL_UNSUPPORTED_TEMPLATE_ERROR = (
    "The chat template is not training-compatible (missing prefix-preservation or "
    "`{% generation %}` markers) and patching is not supported for this template. "
    "Please manually modify the chat template for training."
)


@dataclass(frozen=True)
class RuntimeConfig:
    """The effective GPU-dependent configuration and every recorded deviation."""

    compute_dtype: str
    max_sequence_length: int
    deviations: tuple[str, ...]


def resolve_runtime_config(config: ExperimentConfig, gpu: GpuProbe) -> RuntimeConfig:
    """Apply pre-registered low-memory fallbacks and make each one auditable."""

    compute_dtype = config.compute_dtype
    max_sequence_length = config.max_sequence_length
    deviations: list[str] = []
    if compute_dtype == "bfloat16" and not gpu.bf16_supported:
        compute_dtype = "float16"
        deviations.append("BF16 unsupported; used FP16")
    if gpu.vram_gb < 20.0 and max_sequence_length > 1024:
        max_sequence_length = 1024
        deviations.append(
            "VRAM below 20 GB; reduced max sequence length from "
            f"{config.max_sequence_length} to 1024"
        )
    return RuntimeConfig(
        compute_dtype=compute_dtype,
        max_sequence_length=max_sequence_length,
        deviations=tuple(deviations),
    )


def build_generation_kwargs(config: ExperimentConfig) -> dict[str, object]:
    """Return the paired-condition deterministic decoding settings."""

    return {
        "do_sample": False,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
    }


def probe_gpu() -> GpuProbe:
    """Probe the active CUDA device without importing torch during ordinary tests."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA CUDA GPU is required for the QLoRA runtime")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    native_bf16_supported = (properties.major, properties.minor) >= (8, 0) and bool(
        torch.cuda.is_bf16_supported()
    )
    return GpuProbe(
        name=torch.cuda.get_device_name(device_index),
        vram_gb=properties.total_memory / (1024**3),
        bf16_supported=native_bf16_supported,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def resolve_hub_revision(repo_id: str) -> str:
    """Resolve a Hub model identifier to its immutable 40-character commit SHA."""

    from huggingface_hub import HfApi

    revision = HfApi().model_info(repo_id).sha
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        raise ValueError(f"Hub did not return an immutable 40-character revision for {repo_id}")
    return revision.casefold()


def select_validation_checkpoint(
    validation_checkpoints: Mapping[str, Mapping[str, float]],
) -> tuple[str, dict[str, float]]:
    """Select solely by validation metrics; evaluation assets are not an input."""

    if not validation_checkpoints:
        raise ValueError("at least one validation checkpoint is required")
    normalized: list[tuple[str, dict[str, float]]] = []
    for checkpoint, metrics in validation_checkpoints.items():
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("checkpoint paths must be non-empty strings")
        if not isinstance(metrics, Mapping) or "validation_loss" not in metrics:
            raise ValueError("checkpoint selection requires validation_loss only from validation")
        selected: dict[str, float] = {}
        for name, value in metrics.items():
            if not isinstance(name, str) or not name.startswith("validation_"):
                raise ValueError("checkpoint selection accepts only validation metrics")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("validation metrics must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("validation metrics must be finite")
            selected[name] = float(value)
        normalized.append((checkpoint, selected))
    return min(normalized, key=lambda item: (item[1]["validation_loss"], item[0]))


def verify_frozen_evaluation(
    cases_dir: Path = FROZEN_EVALUATION_CASES,
    sources_path: Path = FROZEN_EVALUATION_SOURCES,
    freeze_path: Path = FROZEN_EVALUATION_FILE,
) -> str:
    """Recompute and compare the frozen evaluation digest before GPU work begins."""

    case_paths = sorted(cases_dir.rglob("*.jsonl")) if cases_dir.is_dir() else []
    if not case_paths or not sources_path.is_file() or not freeze_path.is_file():
        raise ValueError("frozen evaluation assets are incomplete")
    expected = freeze_path.read_text(encoding="utf-8").strip().casefold()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("the frozen evaluation hash must be a SHA-256 digest")
    actual = freeze_digest([*case_paths, sources_path])
    if actual != expected:
        raise ValueError("frozen evaluation assets do not match the recorded freeze digest")
    return actual


def verify_frozen_dataset_split(directory: Path) -> dict[str, str]:
    """Recompute prepared train/validation artifact hashes and fail closed on drift."""

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen dataset manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen dataset manifest must be a JSON object")
    verified: dict[str, str] = {}
    for split in ("train", "validation"):
        path = directory / f"{split}.jsonl"
        field = f"{split}_sha256"
        expected = payload.get(field)
        if not path.is_file() or not isinstance(expected, str):
            raise ValueError(f"frozen {split} artifact is incomplete")
        actual = _sha256(path.read_bytes())
        if actual != expected.casefold():
            raise ValueError(f"frozen {split} artifact does not match its manifest hash")
        verified[field] = actual
    return verified


def load_frozen_dataset_split(directory: Path) -> DatasetSplit:
    """Load the immutable train/validation JSONL files prepared by the local pipeline."""

    verify_frozen_dataset_split(directory)

    def load_records(path: Path) -> tuple[InstructionRecord, ...]:
        if not path.is_file():
            raise FileNotFoundError(f"frozen dataset split is missing: {path}")
        records: list[InstructionRecord] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(InstructionRecord.from_dict(payload))
        if not records:
            raise ValueError(f"frozen dataset split is empty: {path}")
        return tuple(records)

    return DatasetSplit(
        train=load_records(directory / "train.jsonl"),
        validation=load_records(directory / "validation.jsonl"),
    )


def _limit_dataset_split(config: ExperimentConfig, datasets: DatasetSplit) -> DatasetSplit:
    """Apply a deterministic, configuration-bound record limit for smoke runs."""

    if config.record_limit is None:
        return datasets
    return DatasetSplit(
        train=datasets.train[: config.record_limit],
        validation=datasets.validation[: config.record_limit],
    )


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _config_hash(config: ExperimentConfig) -> str:
    return _sha256(canonical_json(asdict(config)))


def _dataset_hash(datasets: DatasetSplit) -> str:
    payload = {
        "train": [record.to_dict() for record in datasets.train],
        "validation": [record.to_dict() for record in datasets.validation],
    }
    return _sha256(canonical_json(payload))


def _frozen_evaluation_hash() -> str:
    return verify_frozen_evaluation()


def _immutable_revision(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("model revision must be an immutable 40-character SHA")
    return value.casefold()


def _resolved_revision(config: ExperimentConfig, model_revision: str | None) -> str:
    return (
        resolve_hub_revision(config.model_id)
        if model_revision is None
        else _immutable_revision(model_revision)
    )


def _package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "trl", "peft", "bitsandbytes")
    }


def _runtime_evidence_state(config: ExperimentConfig) -> str:
    return "smoke_test_verified" if config.name == "smoke" else "8b_gpu_measured"


def _dtype(name: str):
    import torch

    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _train_with_runtime_precision(trainer, runtime: RuntimeConfig) -> None:
    import torch

    if runtime.compute_dtype == "float16":
        for parameter in trainer.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.float32)
    trainer.train()


def _load_model_and_tokenizer(config: ExperimentConfig, revision: str, runtime: RuntimeConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_dtype(runtime.compute_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=revision,
        quantization_config=quantization,
        device_map="auto",
    )
    return model, tokenizer


def _assert_assistant_generation_mask(
    tokenizer,
    record: InstructionRecord,
    *,
    model_id: str,
    model_revision: str,
) -> None:
    from trl.chat_template_utils import (
        get_training_chat_template,
        qwen3_training_chat_template,
    )

    try:
        training_chat_template = get_training_chat_template(tokenizer)
    except ValueError as error:
        raw_template = getattr(tokenizer, "chat_template", None)
        expected_hash = _QWEN3_LEGACY_TEMPLATE_PROVENANCE.get((model_id, model_revision))
        if (
            str(error) != _TRL_UNSUPPORTED_TEMPLATE_ERROR
            or not isinstance(raw_template, str)
            or expected_hash is None
            or _sha256(raw_template) != expected_hash
        ):
            raise
        training_chat_template = qwen3_training_chat_template
    if training_chat_template is not None:
        tokenizer.chat_template = training_chat_template
    rendered = tokenizer.apply_chat_template(
        [{"role": message.role, "content": message.content} for message in record.messages],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        chat_template=training_chat_template,
    )
    masks = rendered.get("assistant_masks") or rendered.get("assistant_tokens_mask")
    if masks is None or not any(masks):
        raise ValueError("the tokenizer chat template must expose an assistant-generation mask")


def _validation_records(
    log_history: Sequence[object], checkpoint_root: Path
) -> dict[str, dict[str, float]]:
    validation_entries: dict[str, dict[str, float]] = {}
    for entry in log_history:
        if not isinstance(entry, Mapping) or "eval_loss" not in entry:
            continue
        step = entry.get("step")
        if isinstance(step, bool) or not isinstance(step, (int, float)) or step < 0:
            raise ValueError("validation history must identify its checkpoint step")
        if int(step) != step:
            raise ValueError("validation history checkpoint step must be integral")
        translated = {
            "validation_loss": float(entry["eval_loss"]),
            **{
                f"validation_{name.removeprefix('eval_')}": float(value)
                for name, value in entry.items()
                if isinstance(name, str)
                and name.startswith("eval_")
                and name != "eval_loss"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            },
        }
        checkpoint = str(checkpoint_root / f"checkpoint-{int(step)}")
        validation_entries[checkpoint] = translated
    if not validation_entries:
        raise ValueError("training completed without validation loss history")
    return validation_entries


def train_adapter(
    config: ExperimentConfig,
    datasets: DatasetSplit,
    *,
    model_revision: str | None = None,
) -> RunManifest:
    """Train a QLoRA adapter and record validation-only checkpoint selection evidence."""

    resolved_revision = _resolved_revision(config, model_revision)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    gpu = probe_gpu()
    runtime = resolve_runtime_config(config, gpu)
    training_datasets = _limit_dataset_split(config, datasets)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = _load_model_and_tokenizer(config, resolved_revision, runtime)
    _assert_assistant_generation_mask(
        tokenizer,
        training_datasets.train[0],
        model_id=config.model_id,
        model_revision=resolved_revision,
    )
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    output_dir = Path(config.output_dir or f"artifacts/checkpoints/{config.name}")
    arguments = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.max_epochs,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        per_device_train_batch_size=config.micro_batch_size,
        per_device_eval_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.scheduler,
        warmup_ratio=config.warmup_ratio,
        max_length=runtime.max_sequence_length,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=runtime.compute_dtype == "bfloat16",
        fp16=runtime.compute_dtype == "float16",
        seed=config.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )
    train_records = Dataset.from_list([record.to_dict() for record in training_datasets.train])
    validation_records = Dataset.from_list(
        [record.to_dict() for record in training_datasets.validation]
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=train_records,
        eval_dataset=validation_records,
        processing_class=tokenizer,
        peft_config=lora,
    )
    started = time.monotonic()
    _train_with_runtime_precision(trainer, runtime)
    final_adapter = output_dir / "final-adapter"
    trainer.save_model(str(final_adapter))
    selected_checkpoint, validation_selection = select_validation_checkpoint(
        _validation_records(trainer.state.log_history, output_dir)
    )
    if not Path(selected_checkpoint).is_dir():
        raise FileNotFoundError(f"selected validation checkpoint is missing: {selected_checkpoint}")
    adapter_file = Path(selected_checkpoint) / "adapter_model.safetensors"
    adapter_hash = _sha256(adapter_file.read_bytes()) if adapter_file.is_file() else ""
    return RunManifest(
        run_id=f"{config.name}-{int(time.time())}",
        evidence_state=_runtime_evidence_state(config),
        model_id=config.model_id,
        model_revision=resolved_revision,
        config_hash=_config_hash(config),
        dataset_hash=_dataset_hash(training_datasets),
        evaluation_hash=_frozen_evaluation_hash(),
        selected_checkpoint=selected_checkpoint,
        validation_selection=validation_selection,
        artifacts={
            "final_adapter": str(final_adapter),
            "adapter_sha256": adapter_hash,
            "runtime_seconds": f"{time.monotonic() - started:.6f}",
            "peak_vram_gb": f"{torch.cuda.max_memory_allocated() / (1024**3):.6f}",
            "runtime_deviations": canonical_json(runtime.deviations),
        },
    )


def generate_condition(
    config: ExperimentConfig,
    cases: Sequence[EvalCase],
    adapter_path: Path | None = None,
    *,
    model_revision: str | None = None,
) -> list[Generation]:
    """Generate deterministic base or adapter responses for exactly the supplied cases."""

    import torch

    if not cases:
        raise ValueError("at least one evaluation case is required")
    resolved_revision = _resolved_revision(config, model_revision)
    runtime = resolve_runtime_config(config, probe_gpu())
    model, tokenizer = _load_model_and_tokenizer(config, resolved_revision, runtime)
    condition = "base"
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        condition = config.name
    model.eval()
    evidence_state = _runtime_evidence_state(config)
    generations: list[Generation] = []
    for case in cases:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": case.prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            tokens = model.generate(**encoded, **build_generation_kwargs(config))
        response_tokens = tokens[0][encoded["input_ids"].shape[1] :]
        generations.append(
            Generation(
                case_id=case.id,
                condition=condition,
                output=tokenizer.decode(response_tokens, skip_special_tokens=True),
                evidence_state=evidence_state,
                metadata={
                    "model_revision": resolved_revision,
                    "runtime_deviations": runtime.deviations,
                },
            )
        )
    return generations


def build_condition_manifest(
    config: ExperimentConfig,
    generations: Sequence[Generation],
    *,
    gpu: GpuProbe,
    runtime_seconds: float,
    peak_vram_gb: float,
    model_revision: str | None = None,
    package_versions: Mapping[str, str],
    adapter_hash: str | None = None,
) -> ConditionManifest:
    """Build the shared Task 5 typed condition evidence without running inference."""

    if not generations:
        raise ValueError("condition manifest requires at least one generation")
    conditions = {generation.condition for generation in generations}
    states = {generation.evidence_state for generation in generations}
    if len(conditions) != 1 or len(states) != 1:
        raise ValueError(
            "condition manifest generations must share one condition and evidence state"
        )
    raw_generation_revisions = [
        generation.metadata.get("model_revision") for generation in generations
    ]
    if not all(isinstance(revision, str) for revision in raw_generation_revisions):
        raise ValueError("condition manifest requires one immutable generation revision")
    generation_revisions = {_immutable_revision(revision) for revision in raw_generation_revisions}
    if len(generation_revisions) != 1:
        raise ValueError("condition manifest requires one immutable generation revision")
    actual_revision = generation_revisions.pop()
    if model_revision is not None and _immutable_revision(model_revision) != actual_revision:
        raise ValueError(
            "condition manifest caller revision does not match the generation revision"
        )
    generation_rows = [asdict(generation) for generation in generations]
    inference_payload = {
        "model_id": config.model_id,
        "model_revision": actual_revision,
        "system_prompt": SYSTEM_PROMPT,
        "generation": build_generation_kwargs(config),
    }
    return ConditionManifest(
        evidence_state=states.pop(),
        condition=conditions.pop(),
        evaluation_hash=_frozen_evaluation_hash(),
        case_ids=tuple(generation.case_id for generation in generations),
        inference_config_hash=_sha256(canonical_json(inference_payload)),
        generation_hash=_sha256(canonical_json(generation_rows)),
        gpu=gpu.to_dict(),
        runtime_seconds=runtime_seconds,
        peak_vram_gb=peak_vram_gb,
        model_revision=actual_revision,
        package_versions=dict(package_versions),
        adapter_hash=adapter_hash,
    )


def export_condition_manifest(manifest: ConditionManifest, path: Path) -> None:
    """Write one canonical, portable condition manifest for Task 5 reporting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8", newline="\n")


def export_generations(generations: Sequence[Generation], path: Path) -> None:
    """Write raw condition results in the JSONL shape consumed by local scoring."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"{canonical_json(generation)}\n" for generation in generations)
    path.write_text(lines, encoding="utf-8", newline="\n")
