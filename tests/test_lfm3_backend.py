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


@pytest.mark.parametrize("interrupted", [False, True])
def test_collect_only_sampler_uses_existing_prefill_callback_and_counts_reasoning(
    monkeypatch, interrupted
):
    from apple_harness_adapter.backends import base
    from apple_harness_adapter.endpoint import chat_completion
    from apple_harness_adapter.telemetry import ModelRequest

    policy = LFM3Policy.__new__(LFM3Policy)
    policy.model_name = "fixture"
    policy.model = object()
    policy.prompt_ids = lambda *_args: [1, 4]

    class Tokenizer:
        eos_token_ids = (6,)

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return [{"<think>": 4, "</think>": 5}[text]]

        def decode(self, tokens, *, skip_special_tokens):
            assert not skip_special_tokens
            return "".join({0: "Plan.", 4: "<think>", 5: "</think>"}[token] for token in tokens)

    policy.tokenizer = Tokenizer()

    def generate(
        prompt, model, *, max_tokens, sampler, stream, prefill_step_size, prompt_progress_callback
    ):
        assert prompt.tolist() == [1, 4]
        assert model is policy.model and max_tokens == 3
        prompt_progress_callback(0, 2)
        prompt_progress_callback(2, 2)
        for token in (0, 5, 6):
            yield token, mx.zeros((7,))
            if interrupted:
                raise RuntimeError("fixture sampling failure")

    monkeypatch.setattr(base, "generate_step", generate)
    request = ModelRequest(relay_id=0, endpoint_id=0)

    def complete():
        return chat_completion(
            policy,
            {"messages": [], "temperature": 0.8},
            max_context=5,
            max_output=3,
            effort="low",
            telemetry=request,
        )

    if interrupted:
        with pytest.raises(RuntimeError, match="fixture sampling failure"):
            complete()
        assert request.observed_sampled_tokens == 1
        assert request.sampled_tokens is request.reasoning_tokens is None
        assert request.decode_seconds >= 0
        return

    payload = complete()
    assert payload["choices"][0]["token_ids"] == [0, 5, 6]
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert request.sampled_tokens == 3 and request.reasoning_tokens == 1
    assert request.prefill_processed_tokens == 2 and request.prefill_reused_tokens == 0
    assert 0 <= request.prefill_to_first_token_seconds <= request.inference_seconds
    assert request.decode_seconds == pytest.approx(
        request.inference_seconds - request.prefill_to_first_token_seconds
    )
