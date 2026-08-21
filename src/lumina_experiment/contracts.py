from __future__ import annotations

import json
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


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    evidence_state: str
    model_id: str
    model_revision: str
    config_hash: str
    dataset_hash: str
    evaluation_hash: str
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


def canonical_json(value: object) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


JsonObject = Mapping[str, Any]
