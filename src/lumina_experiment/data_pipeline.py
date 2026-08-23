from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lumina_experiment.contracts import InstructionRecord, Message

OASST_SOURCE = "OpenAssistant/oasst1"
OASST_LICENSE = "Apache-2.0"
AUTHORED_SOURCE = "lumina-demonstration-authored"
AUTHORED_LICENSE = "CC-BY-4.0"

REJECTION_REASONS = frozenset(
    {
        "empty_content",
        "invalid_role_order",
        "non_english",
        "spam_or_deleted",
        "over_length",
        "credential_pattern",
        "duplicate_pair",
    }
)

_CREDENTIAL_PATTERN = re.compile(
    r"(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|"
    r"sk-[A-Za-z0-9]{20,})"
)

_CAPABILITY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("programming", ("python", "function", "code", "debug", "algorithm")),
    ("json", ("json", "schema", "structured output")),
    ("summarization", ("summarize", "summary", "condense")),
    ("writing", ("rewrite", "edit", "draft", "email", "paragraph")),
    ("reasoning", ("reason", "calculate", "solve", "logic")),
    ("knowledge", ("explain", "what is", "uses of", "history", "science")),
    ("instruction", ("exactly", "must include", "do not", "format")),
)


@dataclass(frozen=True)
class Rejection:
    id: str
    source: str
    reason: str

    def __post_init__(self) -> None:
        if self.reason not in REJECTION_REASONS:
            raise ValueError(f"unsupported rejection reason: {self.reason}")

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "source": self.source, "reason": self.reason}


@dataclass(frozen=True)
class DataPreparationConfig:
    dataset_id: str
    language: str
    seed: int
    total_records: int
    train_records: int
    validation_records: int
    mixture: Mapping[str, int]
    authored_path: str
    processed_dir: str
    max_characters: int = 12_000
    revision_file: str | None = None
    dataset_revision: str | None = None

    def __post_init__(self) -> None:
        required_mixture = {"general_oasst1", "capability_oasst1", "authored"}
        if set(self.mixture) != required_mixture:
            raise ValueError(f"mixture keys must be exactly {sorted(required_mixture)}")
        if any(value < 0 for value in self.mixture.values()):
            raise ValueError("mixture counts must not be negative")
        if sum(self.mixture.values()) != self.total_records:
            raise ValueError("mixture counts must equal total_records")
        if self.train_records + self.validation_records != self.total_records:
            raise ValueError("train and validation counts must equal total_records")
        if self.seed < 0 or self.max_characters <= 0:
            raise ValueError("seed and max_characters must be valid non-negative values")
        if self.language != "en":
            raise ValueError("this Lumina pilot supports English data only")

    @classmethod
    def for_test(cls, *, total_records: int, mixture: Mapping[str, int]) -> DataPreparationConfig:
        return cls(
            dataset_id=OASST_SOURCE,
            dataset_revision="fixture-revision",
            language="en",
            seed=42,
            total_records=total_records,
            train_records=total_records,
            validation_records=0,
            mixture=dict(mixture),
            authored_path="data/authored/training.jsonl",
            processed_dir="data/processed",
        )


@dataclass(frozen=True)
class PreparationResult:
    accepted: tuple[InstructionRecord, ...]
    rejected: tuple[Rejection, ...] = field(default_factory=tuple)


def load_data_config(path: Path) -> DataPreparationConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("data configuration must be a YAML mapping")
    try:
        return DataPreparationConfig(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid data configuration fields: {exc}") from exc


def classify_capability(prompt: str) -> str:
    normalized = prompt.casefold()
    for capability, keywords in _CAPABILITY_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return capability
    return "general"


def _is_flagged(row: Mapping[str, object]) -> bool:
    return bool(row.get("deleted") or row.get("spam") or row.get("is_spam"))


def _row_text(row: Mapping[str, object]) -> str:
    value = row.get("text")
    return value.strip() if isinstance(value, str) else ""


def _rank_key(row: Mapping[str, object]) -> tuple[float, str]:
    rank = row.get("rank")
    numeric_rank = float(rank) if isinstance(rank, (int, float)) else float("inf")
    return numeric_rank, str(row.get("message_id", ""))


def extract_oasst_pairs(
    rows: Iterable[Mapping[str, object]],
) -> list[InstructionRecord]:
    materialized = [dict(row) for row in rows]
    children: dict[str, list[dict[str, object]]] = {}
    for row in materialized:
        parent_id = row.get("parent_id")
        if isinstance(parent_id, str):
            children.setdefault(parent_id, []).append(row)

    roots = sorted(
        (
            row
            for row in materialized
            if row.get("parent_id") is None
            and row.get("role") in {"prompter", "user"}
            and row.get("lang") == "en"
            and not _is_flagged(row)
            and _row_text(row)
        ),
        key=lambda row: (str(row.get("message_tree_id", "")), str(row.get("message_id", ""))),
    )

    records: list[InstructionRecord] = []
    for root in roots:
        root_id = str(root.get("message_id", ""))
        candidates = [
            child
            for child in children.get(root_id, [])
            if child.get("role") == "assistant"
            and child.get("lang") == "en"
            and not _is_flagged(child)
            and _row_text(child)
        ]
        if not candidates:
            continue
        reply = min(candidates, key=_rank_key)
        reply_id = str(reply.get("message_id", ""))
        tree_id = str(root.get("message_tree_id") or reply.get("message_tree_id") or root_id)
        prompt = _row_text(root)
        records.append(
            InstructionRecord(
                id=f"oasst:{root_id}:{reply_id}",
                source=OASST_SOURCE,
                license=OASST_LICENSE,
                capability=classify_capability(prompt),
                cluster_id=tree_id,
                messages=(
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=_row_text(reply)),
                ),
                metadata={
                    "language": "en",
                    "message_tree_id": tree_id,
                    "row_ids": [root_id, reply_id],
                    "selected_reply_rank": reply.get("rank"),
                },
            )
        )
    return records


def audit_oasst_rows(rows: Iterable[Mapping[str, object]]) -> list[Rejection]:
    rejected: list[Rejection] = []
    for row in sorted(rows, key=lambda item: str(item.get("message_id", ""))):
        row_id = str(row.get("message_id", ""))
        if _is_flagged(row):
            reason = "spam_or_deleted"
        elif row.get("lang") != "en":
            reason = "non_english"
        elif not _row_text(row):
            reason = "empty_content"
        else:
            continue
        rejected.append(Rejection(id=row_id, source=OASST_SOURCE, reason=reason))
    return rejected


def _pair_key(record: InstructionRecord) -> tuple[str, ...]:
    return tuple(" ".join(message.content.casefold().split()) for message in record.messages)


def _rejection_reason(
    record: InstructionRecord, *, max_characters: int, duplicate: bool
) -> str | None:
    if any(not message.content.strip() for message in record.messages):
        return "empty_content"
    if record.metadata.get("language") != "en":
        return "non_english"
    if record.metadata.get("spam") or record.metadata.get("deleted"):
        return "spam_or_deleted"
    if sum(len(message.content) for message in record.messages) > max_characters:
        return "over_length"
    if any(_CREDENTIAL_PATTERN.search(message.content) for message in record.messages):
        return "credential_pattern"
    if duplicate:
        return "duplicate_pair"
    return None


def filter_records(
    records: Iterable[InstructionRecord], max_characters: int
) -> tuple[list[InstructionRecord], list[Rejection]]:
    accepted: list[InstructionRecord] = []
    rejected: list[Rejection] = []
    seen_pairs: set[tuple[str, ...]] = set()

    for record in records:
        pair_key = _pair_key(record)
        reason = _rejection_reason(
            record, max_characters=max_characters, duplicate=pair_key in seen_pairs
        )
        if reason is not None:
            rejected.append(Rejection(id=record.id, source=record.source, reason=reason))
            continue
        seen_pairs.add(pair_key)
        accepted.append(record)
    return accepted, rejected


def _normalize_authored(
    authored_rows: Iterable[InstructionRecord | Mapping[str, object]],
) -> list[InstructionRecord]:
    return validate_authored_records(authored_rows)


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text) if part.strip()])


def _validate_json_answer(record: InstructionRecord, family: str) -> None:
    expected_keys = {
        "inventory": {"sku", "quantity", "reorder"},
        "meeting_action": {"meeting_id", "owner", "due_date", "completed"},
        "support_ticket": {"ticket_id", "priority", "tags"},
        "project_risk": {"risk", "likelihood", "mitigation"},
        "travel_plan": {"city", "days", "activities"},
    }
    if family not in expected_keys:
        raise ValueError(f"authored record {record.id} has unknown JSON family")
    try:
        payload = json.loads(record.messages[-1].content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"authored record {record.id} has invalid JSON output") from exc
    if not isinstance(payload, dict) or set(payload) != expected_keys[family]:
        raise ValueError(f"authored record {record.id} violates exact JSON keys")
    if family in {"support_ticket", "travel_plan"}:
        array_key = "tags" if family == "support_ticket" else "activities"
        if not isinstance(payload[array_key], list) or len(payload[array_key]) != 2:
            raise ValueError(f"authored record {record.id} violates array length")


def _validate_instruction_answer(record: InstructionRecord, family: str) -> None:
    answer = record.messages[-1].content
    lines = answer.splitlines()
    if family == "exact_bullets":
        if len(lines) != 3 or not all(
            line.startswith(prefix)
            for line, prefix in zip(lines, ("- Check:", "- Act:", "- Review:"), strict=True)
        ):
            raise ValueError(f"authored record {record.id} violates bullet constraints")
        return
    if family == "bounded_summary":
        words = re.findall(r"\b[\w'-]+\b", answer)
        if (
            _sentence_count(answer) != 2
            or len(words) > 35
            or "measurable progress" not in answer
            or re.search(r"\bvery\b", answer, flags=re.IGNORECASE)
        ):
            raise ValueError(f"authored record {record.id} violates summary constraints")
        return
    if family == "fixed_headings":
        if (
            len(lines) != 6
            or lines[0::2] != ["Situation", "Action", "Result"]
            or any(_sentence_count(line) != 1 for line in lines[1::2])
        ):
            raise ValueError(f"authored record {record.id} violates heading constraints")
        return
    if family == "four_line_email":
        if (
            len(lines) != 4
            or not lines[0].startswith("Subject:")
            or not lines[1].startswith("Hello")
            or "next step" not in lines[2]
            or lines[3] != "Regards, Sam"
            or "!" in answer
        ):
            raise ValueError(f"authored record {record.id} violates email constraints")
        return
    if family == "numbered_steps":
        if (
            len(lines) != 4
            or any(not line.startswith(f"{index}.") for index, line in enumerate(lines, 1))
            or any(len(re.findall(r"\b[\w'-]+\b", line.partition(".")[2])) > 8 for line in lines)
            or ";" in answer
        ):
            raise ValueError(f"authored record {record.id} violates numbered-step constraints")
        return
    raise ValueError(f"authored record {record.id} has unknown instruction family")


def validate_authored_records(
    authored_rows: Iterable[InstructionRecord | Mapping[str, object]],
) -> list[InstructionRecord]:
    normalized = [
        row if isinstance(row, InstructionRecord) else InstructionRecord.from_dict(row)
        for row in authored_rows
    ]
    if len({record.id for record in normalized}) != len(normalized):
        raise ValueError("authored record IDs must be unique")
    if len({record.cluster_id for record in normalized}) != len(normalized):
        raise ValueError("authored cluster IDs must be unique")
    for record in normalized:
        if record.source != AUTHORED_SOURCE or record.license != AUTHORED_LICENSE:
            raise ValueError(f"authored record {record.id} has invalid source or license")
        if record.metadata.get("authorship") != "ai_assisted":
            raise ValueError(f"authored record {record.id} lacks transparent authorship")
        if record.metadata.get("review_status") not in {
            "automated_checks_passed_pending_human",
            "human_approved",
        }:
            raise ValueError(f"authored record {record.id} has invalid review_status")
        if record.metadata.get("language") != "en":
            raise ValueError(f"authored record {record.id} must declare English language")
        family = record.metadata.get("family")
        if not isinstance(family, str):
            raise ValueError(f"authored record {record.id} lacks a task family")
        if record.capability == "json":
            _validate_json_answer(record, family)
        elif record.capability == "instruction":
            _validate_instruction_answer(record, family)
        else:
            raise ValueError(f"authored record {record.id} has invalid capability")
    return normalized


def _select(records: list[InstructionRecord], count: int, rng: random.Random, label: str):
    if len(records) < count:
        raise ValueError(f"insufficient {label} records: need {count}, found {len(records)}")
    ordered = sorted(records, key=lambda record: record.id)
    rng.shuffle(ordered)
    return ordered[:count]


def prepare_dataset(
    oasst_rows: Iterable[Mapping[str, object]],
    authored_rows: Iterable[InstructionRecord | Mapping[str, object]],
    config: DataPreparationConfig,
) -> list[InstructionRecord]:
    extracted = extract_oasst_pairs(oasst_rows)
    oasst, _ = filter_records(extracted, config.max_characters)
    authored, _ = filter_records(_normalize_authored(authored_rows), config.max_characters)

    general = [record for record in oasst if record.capability == "general"]
    capability = [record for record in oasst if record.capability != "general"]
    rng = random.Random(config.seed)
    selected = [
        *_select(general, config.mixture["general_oasst1"], rng, "general OASST1"),
        *_select(
            capability,
            config.mixture["capability_oasst1"],
            rng,
            "capability-targeted OASST1",
        ),
        *_select(authored, config.mixture["authored"], rng, "authored"),
    ]
    rng.shuffle(selected)
    if len(selected) != config.total_records:
        expected = config.total_records
        found = len(selected)
        raise ValueError(f"accepted record count mismatch: expected {expected}, found {found}")
    return selected
