import json

import pytest

from lumina_experiment.contracts import (
    ConditionManifest,
    EvalCase,
    Generation,
    InstructionRecord,
    Message,
    RunManifest,
    canonical_json,
)


def test_condition_manifest_has_typed_task_6_to_task_8_evidence_fields() -> None:
    manifest = ConditionManifest.from_dict(
        {
            "evidence_state": "8b_gpu_measured",
            "condition": "base",
            "evaluation_hash": "a" * 64,
            "case_ids": ["instruction-001"],
            "inference_config_hash": "b" * 64,
            "generation_hash": "c" * 64,
            "gpu": {
                "name": "NVIDIA T4",
                "vram_gb": 15.0,
                "bf16_supported": False,
                "torch_version": "2.0",
                "cuda_version": "12.1",
            },
            "runtime_seconds": 10.0,
            "peak_vram_gb": 12.0,
            "model_revision": "d" * 40,
            "package_versions": {"torch": "2.0"},
        }
    )

    assert manifest.condition == "base"
    assert manifest.case_ids == ("instruction-001",)


def test_condition_manifest_rejects_non_string_package_versions() -> None:
    payload = {
        "evidence_state": "8b_gpu_measured",
        "condition": "base",
        "evaluation_hash": "a" * 64,
        "case_ids": ["instruction-001"],
        "inference_config_hash": "b" * 64,
        "generation_hash": "c" * 64,
        "gpu": {
            "name": "NVIDIA T4",
            "vram_gb": 15.0,
            "bf16_supported": False,
            "torch_version": "2.0",
            "cuda_version": "12.1",
        },
        "runtime_seconds": 10.0,
        "peak_vram_gb": 12.0,
        "model_revision": "d" * 40,
        "package_versions": {"torch": 2},
    }

    with pytest.raises(ValueError, match="package_versions"):
        ConditionManifest.from_dict(payload)


def test_condition_manifest_rejects_non_string_adapter_hash() -> None:
    payload = {
        "evidence_state": "8b_gpu_measured",
        "condition": "primary-r16",
        "evaluation_hash": "a" * 64,
        "case_ids": ["instruction-001"],
        "inference_config_hash": "b" * 64,
        "generation_hash": "c" * 64,
        "gpu": {
            "name": "NVIDIA T4",
            "vram_gb": 15.0,
            "bf16_supported": False,
            "torch_version": "2.0",
            "cuda_version": "12.1",
        },
        "runtime_seconds": 10.0,
        "peak_vram_gb": 12.0,
        "model_revision": "d" * 40,
        "package_versions": {"torch": "2.0"},
        "adapter_hash": 1,
    }

    with pytest.raises(ValueError, match="adapter_hash"):
        ConditionManifest.from_dict(payload)


def valid_record_payload() -> dict[str, object]:
    return {
        "id": "oasst:1",
        "source": "OpenAssistant/oasst1",
        "license": "Apache-2.0",
        "capability": "general",
        "cluster_id": "tree-1",
        "messages": [
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "Gravity attracts masses."},
        ],
    }


def test_instruction_record_rejects_non_alternating_roles() -> None:
    payload = valid_record_payload()
    payload["messages"] = [
        {"role": "user", "content": "Explain gravity."},
        {"role": "user", "content": "Briefly."},
    ]

    with pytest.raises(ValueError, match="alternate"):
        InstructionRecord.from_dict(payload)


def test_instruction_record_requires_final_assistant_message() -> None:
    payload = valid_record_payload()
    payload["messages"] = [{"role": "user", "content": "Explain gravity."}]

    with pytest.raises(ValueError, match="final message"):
        InstructionRecord.from_dict(payload)


def test_instruction_record_normalizes_messages_to_immutable_values() -> None:
    record = InstructionRecord.from_dict(valid_record_payload())

    assert record.messages == (
        Message(role="user", content="Explain gravity."),
        Message(role="assistant", content="Gravity attracts masses."),
    )


def test_generation_identity_is_condition_and_case() -> None:
    generation = Generation(
        case_id="json-001",
        condition="base",
        output="{}",
        evidence_state="8b_gpu_measured",
    )

    assert generation.identity == ("base", "json-001")


def test_generation_rejects_unknown_evidence_state() -> None:
    with pytest.raises(ValueError, match="evidence_state"):
        Generation(
            case_id="json-001",
            condition="base",
            output="{}",
            evidence_state="measured",
        )


def test_canonical_json_is_stable_and_compact() -> None:
    serialized = canonical_json({"z": 1, "a": ["é", 2]})

    assert serialized == '{"a":["é",2],"z":1}'
    assert json.loads(serialized) == {"a": ["é", 2], "z": 1}


def test_eval_case_rejects_unsupported_capability() -> None:
    with pytest.raises(ValueError, match="capability"):
        EvalCase(
            id="unsupported-001",
            capability="translation",
            prompt="Translate this sentence.",
            scorer="rubric",
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "run_id",
        "model_id",
        "model_revision",
        "config_hash",
        "dataset_hash",
        "evaluation_hash",
    ],
)
def test_run_manifest_requires_identity_revision_and_hashes(field_name: str) -> None:
    fields = {
        "run_id": "run-001",
        "evidence_state": "smoke_test_verified",
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "model_revision": "0123456789abcdef",
        "config_hash": "config-hash",
        "dataset_hash": "dataset-hash",
        "evaluation_hash": "evaluation-hash",
    }
    fields[field_name] = ""

    with pytest.raises(ValueError, match=field_name):
        RunManifest(**fields)
