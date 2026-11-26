import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from apple_harness_adapter.backends.qwen3 import Qwen3Policy, parse_completion  # noqa: E402


def test_qwen_tool_calls_require_complete_frames_and_keep_json_content():
    value = "literal <tool_call> and </tool_call> text"
    call = json.dumps({"name": "write", "arguments": {"content": value}})
    other = json.dumps({"name": "undeclared", "arguments": {}})
    text = (
        "<think>Plan.</think>Writing."
        f"<tool_call>\n{call}\n</tool_call>"
        f"<tool_call>{other}</tool_call>"
        f"<tool_call>{call}"
    )
    tools = [{"function": {"name": "write"}}]
    message = parse_completion(text, tools)
    assert message["reasoning_content"] == "Plan."
    assert message["content"] == "Writing."
    (result,) = message["tool_calls"]
    assert result["function"]["name"] == "write"
    assert json.loads(result["function"]["arguments"]) == {"content": value}
    reviewed = parse_completion(text, tools, include_undeclared=True)
    assert [call["function"]["name"] for call in reviewed["tool_calls"]] == ["write", "undeclared"]


def test_qwen_prompt_renders_tool_history_and_thinking_switch():
    policy = Qwen3Policy.__new__(Qwen3Policy)
    history = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "OK"}]},
    ]

    def apply_template(messages, *, tools, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        assert messages[0] == dict(history[0], content="")
        assert messages[1] == dict(history[1], content="OK")
        return "rendered"

    def encode(text, *, add_special_tokens):
        assert text == "rendered" and not add_special_tokens
        return [7, 8]

    policy.tokenizer = SimpleNamespace(apply_chat_template=apply_template, encode=encode)
    assert policy.prompt_ids(history, None, "none") == [7, 8]
    assert history[0]["content"] is None


def test_qwen_rejects_adapter_from_another_checkpoint(tmp_path):
    policy = Qwen3Policy.__new__(Qwen3Policy)
    policy.adapter_metadata = {"backend": "qwen3", "model": "example", "revision": "right"}
    metadata_path = tmp_path / "adapter_metadata.json"
    metadata_path.write_text(json.dumps(dict(policy.adapter_metadata, revision="wrong")))
    with pytest.raises(ValueError, match="does not match the selected backend and model"):
        policy.load_adapter(tmp_path)
