"""Chat-completions endpoint with exact sampled IDs and log probabilities."""

import json
from time import monotonic
from uuid import uuid4

from .telemetry import MODEL_REQUEST, ModelRequest


class ContextBudgetExhausted(ValueError):
    """The full rendered prompt leaves no room for a sampled token."""


def function_call(name: str, arguments: dict) -> dict:
    return {
        "type": "function",
        "id": f"call_{uuid4().hex}",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def chat_completion(
    policy,
    request: dict,
    *,
    max_context: int,
    max_output: int,
    effort: str,
    include_undeclared: bool = False,
    telemetry: ModelRequest | None = None,
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

    if telemetry is None:
        prompt = policy.prompt_ids(messages, tools, effort)
    else:
        started = monotonic()
        try:
            prompt = policy.prompt_ids(messages, tools, effort)
            telemetry.rendered_prompt_tokens = len(prompt)
        finally:
            telemetry.tokenization_seconds = monotonic() - started
    requested = request.get("max_completion_tokens") or request.get("max_tokens") or max_output
    remaining = max_context - len(prompt)
    if remaining < 1:
        raise ContextBudgetExhausted(
            f"Prompt has {len(prompt)} tokens and exceeds the {max_context}-token context"
        )
    limit = min(requested, max_output, remaining)
    if limit < 1:
        raise ValueError("The completion limit must allow at least one token")
    temperature = float(request.get("temperature", 1))
    if telemetry is None:
        turn = policy.sample(prompt, limit, temperature)
    else:
        started = monotonic()
        token = MODEL_REQUEST.set(telemetry)
        try:
            telemetry.sampling_started = True
            turn = policy.sample(prompt, limit, temperature)
            telemetry.sampled_tokens = len(turn.completion_token_ids)
            telemetry.observed_sampled_tokens = telemetry.sampled_tokens
        finally:
            telemetry.inference_seconds = monotonic() - started
            if telemetry.prefill_to_first_token_seconds is not None:
                telemetry.decode_seconds = (
                    telemetry.inference_seconds - telemetry.prefill_to_first_token_seconds
                )
            MODEL_REQUEST.reset(token)
        postprocess_started = monotonic()
    message = policy.parse_completion(
        turn.completion_token_ids, tools, include_undeclared=include_undeclared, effort=effort
    )
    finish = "length" if len(turn.completion_token_ids) == limit else "stop"
    if policy.is_stop_token(turn.completion_token_ids[-1]):
        finish = "stop"
    if message.get("tool_calls"):
        finish = "tool_calls"
    logprobs = [
        {"token": f"token_id:{token}", "logprob": logprob, "bytes": None, "top_logprobs": []}
        for token, logprob in zip(turn.completion_token_ids, turn.old_logprobs, strict=True)
    ]
    payload = {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "model": policy.model_name,
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
    if telemetry is not None:
        telemetry.response_id = payload["id"]
        telemetry.finish_reason = finish
        if finish == "length":
            telemetry.limit_reason = (
                "context" if remaining < min(requested, max_output) else "output"
            )
        reasoning_count = getattr(policy, "reasoning_token_count", None)
        if reasoning_count is not None:
            telemetry.reasoning_tokens = reasoning_count(prompt, turn.completion_token_ids, message)
        telemetry.postprocess_seconds = monotonic() - postprocess_started
    return payload
