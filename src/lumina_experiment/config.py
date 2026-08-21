from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


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
    eval_case_limit: int | None = None
    output_dir: str | None = None

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
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")

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
