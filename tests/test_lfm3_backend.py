import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from apple_harness_adapter.backends.lfm3 import LFM3Policy, parse_completion  # noqa: E402
from apple_harness_adapter.backends.lfm3_model import Model, ModelArgs  # noqa: E402


def test_implicit_thinking_prefix_and_typed_xml_tool_call():
    tools = [
        {
            "function": {
                "name": "bash",
                "parameters": {
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    }
                },
            }
        }
    ]
    call = (
        "<tool_call>\n<function=bash>\n<parameter=command>\nline 1\nline 2\n"
        "</parameter>\n<parameter=timeout>\n2\n</parameter>\n</function>\n</tool_call>"
    )
    result = parse_completion(f"Plan.</think>{call}<tool_call>truncated", tools, thinking=True)
    assert result["reasoning_content"] == "Plan."
    assert result["content"] is None
    assert [json.loads(call["function"]["arguments"]) for call in result["tool_calls"]] == [
        {"command": "line 1\nline 2", "timeout": 2}
    ]
    assert "tool_calls" not in parse_completion(call, [], thinking=True)
    assert "tool_calls" not in parse_completion(call, [])
    assert parse_completion(call, [], include_undeclared=True)["tool_calls"]


def test_lfm3_cached_forward_and_lora_targets():
    args = ModelArgs(
        model_type="lfm3",
        vocab_size=20,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=3,
        layer_types=["conv", "full_attention", "sliding_attention"],
        sliding_window=4,
        rope_parameters={
            "full_attention": {"rope_theta": 10000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
    )
    model = Model(args)
    inputs = mx.array([[1, 3, 4, 1, 0, 9, 8, 7]])
    cache = model.make_cache()
    cached = mx.concatenate(
        [model(inputs[:, index : index + 1], cache=cache) for index in range(inputs.shape[1])],
        axis=1,
    )
    assert mx.allclose(model(inputs), cached, atol=2e-5, rtol=2e-5).item()
    policy = LFM3Policy.__new__(LFM3Policy)
    policy.model = model
    assert {path.rsplit(".", 1)[-1] for path in policy.configure_model()} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "in_proj",
        "out_proj",
        "w1",
        "w2",
        "w3",
    }
