from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
ALLOWED_CAPABILITIES = frozenset(
    {
        "general",
        "instruction",
        "reasoning",
        "knowledge",
        "summarization",
        "writing",
        "programming",
        "json",
        "safety",
    }
)
EVIDENCE_STATES = frozenset(
    {"planned", "locally_verified", "smoke_test_verified", "8b_gpu_measured"}
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ALLOWED_ROLES:
            raise ValueError(f"unsupported message role: {self.role}")
        object.__setattr__(self, "content", _require_text(self.content, "message content"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Message:
        return cls(
            role=_require_text(payload.get("role"), "message role"),
            content=_require_text(payload.get("content"), "message content"),
        )


@dataclass(frozen=True)
class InstructionRecord:
    id: str
    source: str
    license: str
    capability: str
    cluster_id: str
    messages: tuple[Message, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("id", "source", "license", "cluster_id"):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name)
            )
        if self.capability not in ALLOWED_CAPABILITIES:
            raise ValueError(f"unsupported capability: {self.capability}")
        if not self.messages:
            raise ValueError("messages must not be empty")

        conversational = self.messages
        if self.messages[0].role == "system":
            conversational = self.messages[1:]
        if not conversational or conversational[0].role != "user":
            raise ValueError(
                "conversation must begin with a user message after an optional system message"
            )
        expected = "user"
        for message in conversational:
            if message.role != expected:
                raise ValueError("user and assistant roles must alternate")
            expected = "assistant" if expected == "user" else "user"
        if conversational[-1].role != "assistant":
            raise ValueError("final message must have assistant role")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InstructionRecord:
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
            raise ValueError("messages must be a sequence")
        messages = tuple(
            Message.from_dict(message) for message in raw_messages if isinstance(message, Mapping)
        )
        if len(messages) != len(raw_messages):
            raise ValueError("each message must be an object")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            id=_require_text(payload.get("id"), "id"),
            source=_require_text(payload.get("source"), "source"),
            license=_require_text(payload.get("license"), "license"),
            capability=_require_text(payload.get("capability"), "capability"),
            cluster_id=_require_text(payload.get("cluster_id"), "cluster_id"),
            messages=messages,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvalCase:
    id: str
    capability: str
    prompt: str
    scorer: str
    expected: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capability not in ALLOWED_CAPABILITIES - {"general"}:
            raise ValueError(f"unsupported evaluation capability: {self.capability}")


@dataclass(frozen=True)
class Score:
    case_id: str
    condition: str
    metric: str
    value: float
    passed: bool | None
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Generation:
    case_id: str
    condition: str
    output: str
    evidence_state: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("case_id", "condition"):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name)
            )
        if self.evidence_state not in EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence_state: {self.evidence_state}")
        if not isinstance(self.output, str):
            raise ValueError("output must be a string")

    @property
    def identity(self) -> tuple[str, str]:
        return (self.condition, self.case_id)


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[InstructionRecord, ...]
    validation: tuple[InstructionRecord, ...]


@dataclass(frozen=True)
class GpuProbe:
    name: str
    vram_gb: float
    bf16_supported: bool
    torch_version: str
    cuda_version: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "gpu.name"))
        object.__setattr__(self, "vram_gb", _require_positive_float(self.vram_gb, "gpu.vram_gb"))
        if not isinstance(self.bf16_supported, bool):
            raise ValueError("gpu.bf16_supported must be boolean")
        object.__setattr__(
            self,
            "torch_version",
            _require_text(self.torch_version, "gpu.torch_version"),
        )
        if self.cuda_version is not None:
            object.__setattr__(
                self,
                "cuda_version",
                _require_text(self.cuda_version, "gpu.cuda_version"),
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GpuProbe:
        bf16_supported = payload.get("bf16_supported")
        if not isinstance(bf16_supported, bool):
            raise ValueError("gpu.bf16_supported must be boolean")
        return cls(
            name=_require_text(payload.get("name"), "gpu.name"),
            vram_gb=_require_positive_float(payload.get("vram_gb"), "gpu.vram_gb"),
            bf16_supported=bf16_supported,
            torch_version=_require_text(payload.get("torch_version"), "gpu.torch_version"),
            cuda_version=(
                _require_text(payload["cuda_version"], "gpu.cuda_version")
                if payload.get("cuda_version") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_hex_digest(value: object, field_name: str, length: int) -> str:
    text = _require_text(value, field_name)
    if len(text) != length or any(character not in "0123456789abcdefABCDEF" for character in text):
        raise ValueError(f"{field_name} must be a {length}-character hexadecimal digest")
    return text.casefold()


def _require_positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be finite and positive") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return number


@dataclass(frozen=True)
class ConditionManifest:
    """Portable evidence emitted for one baseline or adapter generation condition."""

    evidence_state: str
    condition: str
    evaluation_hash: str
    case_ids: tuple[str, ...]
    inference_config_hash: str
    generation_hash: str
    gpu: Mapping[str, object]
    runtime_seconds: float
    peak_vram_gb: float
    model_revision: str
    package_versions: Mapping[str, str]
    adapter_hash: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_state not in EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence_state: {self.evidence_state}")
        object.__setattr__(self, "condition", _require_text(self.condition, "condition"))
        for field_name in (
            "evaluation_hash",
            "inference_config_hash",
            "generation_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_hex_digest(getattr(self, field_name), field_name, 64),
            )
        object.__setattr__(
            self,
            "model_revision",
            _require_hex_digest(self.model_revision, "model_revision", 40),
        )
        if not self.case_ids or len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("case_ids must be non-empty and unique")
        if not isinstance(self.gpu, Mapping):
            raise ValueError("gpu must be an object")
        object.__setattr__(self, "gpu", GpuProbe.from_dict(self.gpu).to_dict())
        for field_name in ("runtime_seconds", "peak_vram_gb"):
            object.__setattr__(
                self,
                field_name,
                _require_positive_float(getattr(self, field_name), field_name),
            )
        if not isinstance(self.package_versions, Mapping) or not self.package_versions:
            raise ValueError("package_versions must not be empty")
        if not all(
            isinstance(name, str) and name.strip() and isinstance(version, str) and version.strip()
            for name, version in self.package_versions.items()
        ):
            raise ValueError("package_versions must contain non-empty strings")
        if self.adapter_hash is not None:
            object.__setattr__(
                self,
                "adapter_hash",
                _require_hex_digest(self.adapter_hash, "adapter_hash", 64),
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConditionManifest:
        raw_case_ids = payload.get("case_ids")
        if not isinstance(raw_case_ids, Sequence) or isinstance(raw_case_ids, (str, bytes)):
            raise ValueError("case_ids must be a sequence")
        gpu = payload.get("gpu")
        packages = payload.get("package_versions")
        if not isinstance(gpu, Mapping):
            raise ValueError("gpu must be an object")
        if not isinstance(packages, Mapping):
            raise ValueError("package_versions must be an object")
        if not all(
            isinstance(name, str) and isinstance(version, str) for name, version in packages.items()
        ):
            raise ValueError("package_versions must contain string names and versions")
        adapter_hash = payload.get("adapter_hash")
        if adapter_hash is not None and not isinstance(adapter_hash, str):
            raise ValueError("adapter_hash must be a string")
        return cls(
            evidence_state=_require_text(payload.get("evidence_state"), "evidence_state"),
            condition=_require_text(payload.get("condition"), "condition"),
            evaluation_hash=_require_text(payload.get("evaluation_hash"), "evaluation_hash"),
            case_ids=tuple(_require_text(case_id, "case_id") for case_id in raw_case_ids),
            inference_config_hash=_require_text(
                payload.get("inference_config_hash"), "inference_config_hash"
            ),
            generation_hash=_require_text(payload.get("generation_hash"), "generation_hash"),
            gpu=dict(gpu),
            runtime_seconds=_require_positive_float(
                payload.get("runtime_seconds"), "runtime_seconds"
            ),
            peak_vram_gb=_require_positive_float(payload.get("peak_vram_gb"), "peak_vram_gb"),
            model_revision=_require_text(payload.get("model_revision"), "model_revision"),
            package_versions=dict(packages),
            adapter_hash=adapter_hash,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    evidence_state: str
    model_id: str
    model_revision: str
    config_hash: str
    dataset_hash: str
    evaluation_hash: str
    selected_checkpoint: str | None = None
    validation_selection: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "model_id",
            "model_revision",
            "config_hash",
            "dataset_hash",
            "evaluation_hash",
        ):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name)
            )
        if self.evidence_state not in EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence_state: {self.evidence_state}")
        if self.selected_checkpoint is not None:
            object.__setattr__(
                self,
                "selected_checkpoint",
                _require_text(self.selected_checkpoint, "selected_checkpoint"),
            )
        if not isinstance(self.validation_selection, Mapping):
            raise ValueError("validation_selection must be an object")
        if any(
            not isinstance(name, str)
            or not name.startswith("validation_")
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for name, value in self.validation_selection.items()
        ):
            raise ValueError("validation_selection must contain finite validation metrics")
        object.__setattr__(self, "validation_selection", dict(self.validation_selection))


def canonical_json(value: object) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


JsonObject = Mapping[str, Any]
