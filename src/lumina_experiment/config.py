from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from lumina_experiment.contracts import EVIDENCE_STATES


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_id: str
    revision_file: str
    seed: int
    quantization: str
    double_quantization: bool
    compute_dtype: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    learning_rate: float
    scheduler: str
    warmup_ratio: float
    max_epochs: int
    max_sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    temperature: float
    max_new_tokens: int
    max_steps: int | None = None
    record_limit: int | None = None
    validation_record_limit: int | None = None
    eval_case_limit: int | None = None
    output_dir: str | None = None
    mixture_source: str | None = None
    mixture_other_source: str | None = None
    mixture_fraction: float | None = None
    evidence_state: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.model_id.strip():
            raise ValueError("name and model_id must be non-empty")
        if self.quantization != "nf4":
            raise ValueError("quantization must be nf4")
        if self.compute_dtype not in {"bfloat16", "float16"}:
            raise ValueError("compute_dtype must be bfloat16 or float16")
        if self.target_modules != "all-linear":
            raise ValueError("target_modules must be all-linear for this pilot")
        if self.temperature != 0.0:
            raise ValueError("temperature must be 0 for paired deterministic evaluation")
        positive_fields = {
            "seed": self.seed,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "max_sequence_length": self.max_sequence_length,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_new_tokens": self.max_new_tokens,
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "max_steps",
            "record_limit",
            "validation_record_limit",
            "eval_case_limit",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when provided")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        mixture_fields = (
            self.mixture_source,
            self.mixture_other_source,
            self.mixture_fraction,
        )
        if any(value is not None for value in mixture_fields) and not all(
            value is not None for value in mixture_fields
        ):
            raise ValueError(
                "mixture_source, mixture_other_source, and mixture_fraction "
                "must be provided together"
            )
        if self.mixture_source is not None:
            if not self.mixture_source.strip():
                raise ValueError("mixture_source must be non-empty")
            if not self.mixture_other_source or not self.mixture_other_source.strip():
                raise ValueError("mixture_other_source must be non-empty")
            if self.mixture_source == self.mixture_other_source:
                raise ValueError("mixture sources must be different")
            if self.record_limit is None:
                raise ValueError("mixture sampling requires record_limit")
            if not 0 < self.mixture_fraction < 1:
                raise ValueError("mixture_fraction must be in (0, 1)")
        if self.evidence_state is not None and self.evidence_state not in EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence_state: {self.evidence_state}")

    def without(self, *field_names: str) -> dict[str, object]:
        excluded = set(field_names)
        return {key: value for key, value in asdict(self).items() if key not in excluded}


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a YAML mapping")
    try:
        return ExperimentConfig(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid experiment configuration fields: {exc}") from exc
