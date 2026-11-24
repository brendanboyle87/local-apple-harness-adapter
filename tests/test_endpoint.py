import json
from types import SimpleNamespace

from apple_harness_adapter.endpoint import chat_completion, parse_completion
from apple_harness_adapter.trace import Turn

TOOLS = [{"type": "function", "function": {"name": "write"}}]
THINK = "<|message_model|><|content_thinking|>Use write.<|end_message|>"
CALL = '<|message_model|>write<|content_invoke_tool_json|>{"name":"write","args":{"path":"a"}}'
EOS = "<|content_model_end_sampling|>"


def test_inkling_message_extraction_handles_complete_and_truncated_calls():
    complete = parse_completion(THINK + CALL + "<|end_message|>" + EOS, TOOLS)
    assert complete["reasoning_content"] == "Use write."
    assert complete["content"] is None
    assert json.loads(complete["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
    assert "tool_calls" not in parse_completion(THINK + CALL, TOOLS)
    assert "tool_calls" not in parse_completion(CALL + "<|end_message|>", [])
    undeclared = parse_completion(CALL + "<|end_message|>", [], include_undeclared=True)
    assert undeclared["tool_calls"][0]["function"]["name"] == "write"


def test_chat_completion_pairs_sampler_tokens_and_logprobs():
    def prompt_ids(messages, tools, effort):
        assert messages[0]["reasoning_content"] == "previous thought"
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a"}
        assert tools == TOOLS and effort == "none"
        return [11, 22]

    def sample(ids, limit, temp):
        assert ids == [11, 22] and limit == 4 and temp == 0.8
        return Turn(tuple(ids), (30, 31), (-0.1, -0.2), temp)

    policy = SimpleNamespace(
        prompt_ids=prompt_ids,
        sample=sample,
        tokenizer=SimpleNamespace(eos_token_ids={31}),
        decode=lambda _: "<|message_model|><|content_text|>Done<|end_message|>" + EOS,
    )
    request = {
        "messages": [
            {
                "role": "assistant",
                "reasoning": "previous thought",
                "tool_calls": [{"function": {"name": "write", "arguments": '{"path":"a"}'}}],
            }
        ],
        "tools": TOOLS,
        "temperature": 0.8,
        "max_tokens": 8,
    }
    response = chat_completion(policy, request, max_context=6, max_output=128, effort="none")
    assert response["prompt_token_ids"] == [11, 22]
    choice = response["choices"][0]
    assert choice["token_ids"] == [30, 31]
    assert [(row["token"], row["logprob"]) for row in choice["logprobs"]["content"]] == [
        ("token_id:30", -0.1),
        ("token_id:31", -0.2),
    ]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Done"
