from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit

from lumina_experiment.contracts import (
    DatasetSplit,
    EvalCase,
    InstructionRecord,
    canonical_json,
)


@dataclass(frozen=True)
class Duplicate:
    kept_id: str
    duplicate_id: str
    reason: str = "normalized_exact_match"


@dataclass(frozen=True)
class ContaminationHit:
    record_id: str
    evaluation_id: str
    source_split: str
    similarity: float
    exact_match: bool
    record_prompt: str
    evaluation_prompt: str


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def _record_key(record: InstructionRecord) -> tuple[tuple[str, str], ...]:
    return tuple((message.role, normalize_text(message.content)) for message in record.messages)


def deduplicate(
    records: Iterable[InstructionRecord],
) -> tuple[list[InstructionRecord], list[Duplicate]]:
    kept: list[InstructionRecord] = []
    removed: list[Duplicate] = []
    seen: dict[tuple[tuple[str, str], ...], str] = {}
    for record in sorted(records, key=lambda item: item.id):
        key = _record_key(record)
        kept_id = seen.get(key)
        if kept_id is not None:
            removed.append(Duplicate(kept_id=kept_id, duplicate_id=record.id))
            continue
        seen[key] = record.id
        kept.append(record)
    return kept, removed


def _capability_divergence(
    validation: Sequence[InstructionRecord], population: Sequence[InstructionRecord]
) -> float:
    population_counts = Counter(record.capability for record in population)
    validation_counts = Counter(record.capability for record in validation)
    return sum(
        abs(
            validation_counts[capability] / len(validation)
            - population_counts[capability] / len(population)
        )
        for capability in population_counts
    )


def cluster_split(
    records: Iterable[InstructionRecord], validation_size: int = 300, seed: int = 42
) -> DatasetSplit:
    ordered = sorted(records, key=lambda item: item.id)
    if validation_size < 0 or validation_size >= len(ordered):
        raise ValueError("validation_size must be non-negative and smaller than the dataset")
    if validation_size == 0:
        return DatasetSplit(train=tuple(ordered), validation=())

    groups = [record.cluster_id for record in ordered]
    unique_groups = set(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two clusters are required for a split")

    target_fraction = validation_size / len(ordered)
    candidate_count = min(512, max(64, len(unique_groups) * 16))
    splitter = GroupShuffleSplit(
        n_splits=candidate_count,
        test_size=target_fraction,
        random_state=seed,
    )
    candidates: list[
        tuple[
            tuple[int, float, tuple[str, ...]],
            tuple[InstructionRecord, ...],
            tuple[InstructionRecord, ...],
        ]
    ] = []
    for train_indices, validation_indices in splitter.split(ordered, groups=groups):
        train = tuple(ordered[index] for index in train_indices)
        validation = tuple(ordered[index] for index in validation_indices)
        score = (
            abs(len(validation) - validation_size),
            _capability_divergence(validation, ordered),
            tuple(record.id for record in validation),
        )
        candidates.append((score, train, validation))

    _, train, validation = min(candidates, key=lambda candidate: candidate[0])
    train_clusters = {record.cluster_id for record in train}
    validation_clusters = {record.cluster_id for record in validation}
    if not train_clusters.isdisjoint(validation_clusters):
        raise ValueError("cluster crossed the train/validation boundary")
    return DatasetSplit(
        train=tuple(sorted(train, key=lambda item: item.id)),
        validation=tuple(sorted(validation, key=lambda item: item.id)),
    )


def _prompt(record: InstructionRecord) -> str:
    return "\n".join(
        normalize_text(message.content) for message in record.messages if message.role == "user"
    )


def find_contamination(
    train: Iterable[InstructionRecord],
    validation: Iterable[InstructionRecord],
    evaluation: Iterable[EvalCase],
    threshold: float = 0.92,
) -> list[ContaminationHit]:
    sources = [
        *(("train", record) for record in train),
        *(("validation", record) for record in validation),
    ]
    evaluation_cases = list(evaluation)
    if not sources or not evaluation_cases:
        return []

    source_prompts = [_prompt(record) for _, record in sources]
    evaluation_prompts = [normalize_text(item.prompt) for item in evaluation_cases]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=True,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform([*source_prompts, *evaluation_prompts])
    source_matrix = matrix[: len(sources)]
    evaluation_matrix = matrix[len(sources) :]
    similarities = cosine_similarity(source_matrix, evaluation_matrix)

    hits: list[ContaminationHit] = []
    for source_index, (source_split, record) in enumerate(sources):
        source_prompt = source_prompts[source_index]
        for evaluation_index, evaluation_case in enumerate(evaluation_cases):
            evaluation_prompt = evaluation_prompts[evaluation_index]
            exact_match = source_prompt == evaluation_prompt
            similarity = float(similarities[source_index, evaluation_index])
            if not exact_match and similarity < threshold:
                continue
            hits.append(
                ContaminationHit(
                    record_id=record.id,
                    evaluation_id=evaluation_case.id,
                    source_split=source_split,
                    similarity=similarity,
                    exact_match=exact_match,
                    record_prompt=source_prompt,
                    evaluation_prompt=evaluation_prompt,
                )
            )
    return sorted(
        hits,
        key=lambda hit: (hit.evaluation_id, hit.source_split, hit.record_id),
    )


def _canonical_file(path: Path) -> tuple[str, int]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, Mapping) and isinstance(row.get("id"), str) for row in rows):
            raise ValueError(f"every JSONL record must have a string id: {path}")
        ordered = sorted(rows, key=lambda row: row["id"])
        return "".join(f"{canonical_json(row)}\n" for row in ordered), len(ordered)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        count = len(payload) if isinstance(payload, list) else 1
        return f"{canonical_json(payload)}\n", count
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        count = len(payload) if isinstance(payload, list) else 1
        return f"{canonical_json(payload)}\n", count
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return normalized, len(normalized.splitlines())


def freeze_files(paths: Sequence[Path], destination: Path) -> str:
    if not paths:
        raise ValueError("at least one file is required for freezing")
    resolved = [path.resolve() for path in paths]
    if any(not path.is_file() for path in resolved):
        raise ValueError("every freeze input must be an existing file")
    common_root = Path(os.path.commonpath([str(path.parent) for path in resolved]))

    entries: list[dict[str, object]] = []
    for path in resolved:
        canonical_content, record_count = _canonical_file(path)
        entries.append(
            {
                "path": path.relative_to(common_root).as_posix(),
                "record_count": record_count,
                "sha256": hashlib.sha256(canonical_content.encode("utf-8")).hexdigest(),
            }
        )
    manifest = {"files": sorted(entries, key=lambda entry: str(entry["path"]))}
    digest = hashlib.sha256(f"{canonical_json(manifest)}\n".encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"{digest}\n", encoding="utf-8", newline="\n")
    return digest
