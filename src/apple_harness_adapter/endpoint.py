"""Inkling chat-completions endpoint with exact sampled IDs and log probabilities."""

import json
import re
from uuid import uuid4

MODEL_NAME = "inkling-2-mini"
FRAME = re.compile(r"<\|message_model\|>([^<]*?)<\|content_(text|thinking|invoke_tool_json)\|>")
END = "<|end_message|>"
EOS = "<|content_model_end_sampling|>"


class ContextBudgetExhausted(ValueError):
    """The full rendered prompt leaves no room for a sampled token."""


def parse_completion(text: str, tools: list[dict], *, include_undeclared: bool = False) -> dict:
    message = {"role": "assistant", "content": None}
    allowed = {tool["function"]["name"] for tool in tools}
    frames = list(FRAME.finditer(text))
    for index, frame in enumerate(frames):
        name, kind = frame.groups()
        boundary = frames[index + 1].start() if index + 1 < len(frames) else len(text)
        segment = text[frame.end() : boundary]
        content, delimiter, _ = segment.partition(END)
        if not delimiter:
            content = content.split(EOS, 1)[0]
        if kind == "text":
            message["content"] = (message["content"] or "") + content
        elif kind == "thinking":
            message["reasoning_content"] = message.get("reasoning_content", "") + content
        elif delimiter and (include_undeclared or name in allowed):
            try:
                invocation = json.loads(content)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(invocation, dict)
                and invocation.get("name") == name
                and isinstance(invocation.get("args"), dict)
            ):
                message.setdefault("tool_calls", []).append(
                    {
                        "type": "function",
                        "id": f"call_{uuid4().hex}",
                        "function": {"name": name, "arguments": json.dumps(invocation["args"])},
                    }
                )
    if not frames:
        message["content"] = text.split(EOS, 1)[0].split(END, 1)[0]
    return message


def chat_completion(
    policy,
    request: dict,
    *,
    max_context: int,
    max_output: int,
    effort: str,
    include_undeclared: bool = False,
) -> dict:
    if request.get("stream") or request.get("n", 1) != 1:
        raise ValueError("The capture endpoint requires one nonstreaming choice")
    tools = request.get("tools") or []
    messages = []
    for original in request["messages"]:
        message = dict(original)
        if "reasoning" in message:
            message["reasoning_content"] = message.pop("reasoning")
        content = message.get("content")
        if isinstance(content, list) and any(
            not isinstance(part, dict) or part.get("type") != "text" for part in content
        ):
            raise ValueError("Only text message parts are supported")
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
                calls.append(dict(call, function=function))
            message["tool_calls"] = calls
        messages.append(message)

    prompt = policy.prompt_ids(messages, tools, effort)
    requested = request.get("max_completion_tokens") or request.get("max_tokens") or max_output
    remaining = max_context - len(prompt)
    if remaining < 1:
        raise ContextBudgetExhausted(
            f"Prompt has {len(prompt)} tokens and exceeds the {max_context}-token context"
        )
    limit = min(requested, max_output, remaining)
    if limit < 1:
        raise ValueError("The completion limit must allow at least one token")
    turn = policy.sample(prompt, limit, float(request.get("temperature", 1)))
    message = parse_completion(
        policy.decode(turn.completion_token_ids), tools, include_undeclared=include_undeclared
    )
    finish = "length" if len(turn.completion_token_ids) == limit else "stop"
    if turn.completion_token_ids[-1] in policy.tokenizer.eos_token_ids:
        finish = "stop"
    if message.get("tool_calls"):
        finish = "tool_calls"
    logprobs = [
        {"token": f"token_id:{token}", "logprob": logprob, "bytes": None, "top_logprobs": []}
        for token, logprob in zip(turn.completion_token_ids, turn.old_logprobs, strict=True)
    ]
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
                "token_ids": list(turn.completion_token_ids),
                "logprobs": {"content": logprobs},
            }
        ],
        "prompt_token_ids": list(turn.prompt_token_ids),
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(logprobs),
            "total_tokens": len(prompt) + len(logprobs),
        },
    }
