from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lumina_experiment.contracts import Generation, canonical_json

REVIEW_CAPABILITIES = (
    "instruction",
    "reasoning",
    "knowledge",
    "summarization",
    "writing",
    "programming",
    "json",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_RAW_ROOT = (REPO_ROOT / "results" / "raw").resolve(strict=False)


@dataclass(frozen=True)
class GenerationPair:
    case_id: str
    base: Generation
    adapter: Generation


@dataclass(frozen=True)
class ReviewItem:
    case_id: str
    capability: str
    prompt: str
    rubric: Mapping[str, object]
    left_label: str
    left_output: str
    right_label: str
    right_output: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "capability": self.capability,
            "prompt": self.prompt,
            "A": self.left_output,
            "B": self.right_output,
            "rubric": dict(self.rubric),
            "winner": "",
            "reviewer_notes": "",
        }


@dataclass(frozen=True)
class ReviewPacket:
    items: tuple[ReviewItem, ...]
    private_key: Mapping[str, Mapping[str, str]]
    seed: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "items": [item.to_public_dict() for item in self.items],
        }

    def to_private_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "assignments": {
                case_id: dict(assignments)
                for case_id, assignments in sorted(self.private_key.items())
            },
        }


def _index_generations(generations: Iterable[Generation], label: str) -> dict[str, Generation]:
    indexed: dict[str, Generation] = {}
    for generation in generations:
        if generation.case_id in indexed:
            raise ValueError(f"duplicate generation for {label}: {generation.case_id}")
        indexed[generation.case_id] = generation
    return indexed


def pair_conditions(
    base_generations: Iterable[Generation], adapter_generations: Iterable[Generation]
) -> tuple[GenerationPair, ...]:
    base = _index_generations(base_generations, "base")
    adapter = _index_generations(adapter_generations, "adapter")
    if any(generation.condition != "base" for generation in base.values()):
        raise ValueError("base condition file contains a non-base condition label")
    if set(base) != set(adapter):
        missing_from_adapter = sorted(set(base) - set(adapter))
        missing_from_base = sorted(set(adapter) - set(base))
        raise ValueError(
            "missing paired generation; "
            f"adapter_missing={missing_from_adapter}, base_missing={missing_from_base}"
        )
    adapter_conditions = {generation.condition for generation in adapter.values()}
    if len(adapter_conditions) != 1 or "base" in adapter_conditions:
        raise ValueError("adapter condition file must contain one non-base condition label")
    return tuple(
        GenerationPair(case_id=case_id, base=base[case_id], adapter=adapter[case_id])
        for case_id in sorted(base)
    )


def _metadata_text(metadata: Mapping[str, object], field: str, case_id: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{case_id} metadata requires non-empty {field}")
    return value.strip()


def _metadata_rubric(metadata: Mapping[str, object], case_id: str) -> Mapping[str, object]:
    rubric = metadata.get("rubric")
    if not isinstance(rubric, Mapping):
        raise ValueError(f"{case_id} metadata requires rubric")
    return rubric


def _balanced_sample(
    pairs: Sequence[GenerationPair], sample_size: int, rng: random.Random
) -> list[GenerationPair]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    grouped: dict[str, list[GenerationPair]] = defaultdict(list)
    for pair in pairs:
        capability = _metadata_text(pair.base.metadata, "capability", pair.case_id)
        if capability in REVIEW_CAPABILITIES:
            grouped[capability].append(pair)
    if sample_size == 42 and any(
        len(grouped[capability]) < 6 for capability in REVIEW_CAPABILITIES
    ):
        raise ValueError("default review requires at least six paired cases per scored capability")
    if sample_size > sum(len(group) for group in grouped.values()):
        raise ValueError("sample_size exceeds available paired review cases")

    for capability in grouped:
        grouped[capability].sort(key=lambda pair: pair.case_id)
        rng.shuffle(grouped[capability])

    selected: list[GenerationPair] = []
    while len(selected) < sample_size:
        progressed = False
        for capability in REVIEW_CAPABILITIES:
            if grouped[capability] and len(selected) < sample_size:
                selected.append(grouped[capability].pop())
                progressed = True
        if not progressed:
            break
    return selected


def build_blind_pairs(
    generations: Iterable[Generation], seed: int = 42, sample_size: int = 42
) -> ReviewPacket:
    ordered = sorted(generations, key=lambda item: (item.condition, item.case_id))
    by_condition: dict[str, list[Generation]] = defaultdict(list)
    for generation in ordered:
        by_condition[generation.condition].append(generation)
    if "base" not in by_condition or len(by_condition) != 2:
        raise ValueError("blind review requires exactly base and one adapter condition")
    adapter_condition = next(condition for condition in by_condition if condition != "base")
    pairs = pair_conditions(by_condition["base"], by_condition[adapter_condition])

    rng = random.Random(seed)
    selected = _balanced_sample(pairs, sample_size, rng)
    items: list[ReviewItem] = []
    private_key: dict[str, dict[str, str]] = {}
    for pair in selected:
        base_metadata = pair.base.metadata
        adapter_metadata = pair.adapter.metadata
        capability = _metadata_text(base_metadata, "capability", pair.case_id)
        prompt = _metadata_text(base_metadata, "prompt", pair.case_id)
        rubric = _metadata_rubric(base_metadata, pair.case_id)
        adapter_rubric = _metadata_rubric(adapter_metadata, pair.case_id)
        if (
            capability != adapter_metadata.get("capability")
            or prompt != adapter_metadata.get("prompt")
            or canonical_json(rubric) != canonical_json(adapter_rubric)
        ):
            raise ValueError(f"paired metadata mismatch for {pair.case_id}")
        if pair.base.evidence_state != pair.adapter.evidence_state:
            raise ValueError(f"paired evidence state mismatch for {pair.case_id}")

        base_on_left = bool(rng.getrandbits(1))
        left = pair.base if base_on_left else pair.adapter
        right = pair.adapter if base_on_left else pair.base
        items.append(
            ReviewItem(
                case_id=pair.case_id,
                capability=capability,
                prompt=prompt,
                rubric=dict(rubric),
                left_label="A",
                left_output=left.output,
                right_label="B",
                right_output=right.output,
            )
        )
        private_key[pair.case_id] = {"A": left.condition, "B": right.condition}

    return ReviewPacket(items=tuple(items), private_key=private_key, seed=seed)


def generations_from_scored_rows(rows: Iterable[Mapping[str, object]]) -> list[Generation]:
    generations: list[Generation] = []
    for index, row in enumerate(rows, 1):
        required = ("case_id", "condition", "capability", "prompt", "output", "evidence_state")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"scored row {index} missing fields: {missing}")
        rubric = row.get("rubric", {})
        if not isinstance(rubric, Mapping):
            raise ValueError(f"scored row {index} rubric must be an object")
        generations.append(
            Generation(
                case_id=str(row["case_id"]),
                condition=str(row["condition"]),
                output=str(row["output"]),
                evidence_state=str(row["evidence_state"]),
                metadata={
                    "capability": str(row["capability"]),
                    "prompt": str(row["prompt"]),
                    "rubric": dict(rubric),
                },
            )
        )
    return generations


def is_private_raw_path(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        parts = [part.casefold() for part in resolved.parts]
        return any(
            parts[index : index + 2] == ["results", "raw"] for index in range(len(parts) - 1)
        )
    try:
        resolved.relative_to(REPO_RAW_ROOT)
    except ValueError:
        return False
    return True


def write_review_packet(packet: ReviewPacket, public_path: Path, private_key_path: Path) -> None:
    if not is_private_raw_path(public_path):
        raise ValueError("review sheet contains raw outputs and must be stored under results/raw")
    if not is_private_raw_path(private_key_path):
        raise ValueError("private review key must be stored under results/raw")
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "capability",
        "prompt",
        "A",
        "B",
        "rubric",
        "winner",
        "reviewer_notes",
    ]
    with public_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in packet.items:
            row = item.to_public_dict()
            row["rubric"] = canonical_json(row["rubric"])
            writer.writerow(row)
    private_key_path.write_text(
        f"{canonical_json(packet.to_private_dict())}\n",
        encoding="utf-8",
        newline="\n",
    )
