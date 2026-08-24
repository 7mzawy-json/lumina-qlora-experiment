from pathlib import Path

import pytest

from lumina_experiment.config import load_experiment_config


def test_primary_config_matches_preregistered_adapter() -> None:
    config = load_experiment_config(Path("configs/primary-r16.yaml"))

    assert config.model_id == "Qwen/Qwen3-8B-Base"
    assert (config.lora_rank, config.lora_alpha) == (16, 32)
    assert config.target_modules == "all-linear"
    assert config.max_epochs == 2
    assert config.temperature == 0.0


def test_pilot_config_is_time_bounded_and_declares_directional_evidence() -> None:
    config = load_experiment_config(Path("configs/pilot-r16.yaml"))

    assert config.model_id == "Qwen/Qwen3-8B-Base"
    assert (config.lora_rank, config.lora_alpha) == (16, 32)
    assert config.max_steps == 16
    assert config.record_limit == 256
    assert config.validation_record_limit == 40
    assert config.eval_case_limit == 24
    assert config.max_sequence_length == 512
    assert config.max_new_tokens == 192
    assert config.mixture_source == "lumina-demonstration-authored"
    assert config.mixture_other_source == "OpenAssistant/oasst1"
    assert config.mixture_fraction == pytest.approx(0.30)
    assert config.evidence_state == "8b_pilot_measured"


def test_lr_diagnostic_changes_only_learning_rate_and_run_identity() -> None:
    pilot = load_experiment_config(Path("configs/pilot-r16.yaml"))
    diagnostic = load_experiment_config(Path("configs/pilot-r16-lr1e4.yaml"))

    assert diagnostic.learning_rate == pytest.approx(0.0001)
    assert diagnostic.name == "pilot-r16-lr1e4"
    assert diagnostic.output_dir == "artifacts/checkpoints/pilot-r16-lr1e4"
    assert pilot.without("name", "learning_rate", "output_dir") == diagnostic.without(
        "name", "learning_rate", "output_dir"
    )


def test_ablation_changes_only_identity_rank_and_alpha() -> None:
    primary = load_experiment_config(Path("configs/primary-r16.yaml"))
    ablation = load_experiment_config(Path("configs/ablation-r8.yaml"))

    assert primary.without("name", "lora_rank", "lora_alpha") == ablation.without(
        "name", "lora_rank", "lora_alpha"
    )


def test_config_rejects_sampling_for_paired_evaluation(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
name: invalid
model_id: Qwen/Qwen3-8B-Base
revision_file: configs/revisions.yaml
seed: 42
quantization: nf4
double_quantization: true
compute_dtype: bfloat16
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: all-linear
learning_rate: 0.0002
scheduler: cosine
warmup_ratio: 0.03
max_epochs: 2
max_sequence_length: 2048
micro_batch_size: 1
gradient_accumulation_steps: 16
temperature: 0.7
max_new_tokens: 512
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="temperature"):
        load_experiment_config(config_path)
