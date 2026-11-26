"""LoRA and JSON tool calls for standard Qwen3 checkpoints."""

import json

from ..endpoint import function_call
from .base import MLXPolicy


class Qwen3Policy(MLXPolicy):
    backend = "qwen3"
    serving_name = None

    def configure_model(self) -> tuple[str, ...]:
        if self.model.model_type != "qwen3":
            raise ValueError("This backend expects a standard Qwen3 checkpoint")
        return (f"model.layers.{len(self.model.layers) - 1}.self_attn.o_proj",)

    def prompt_ids(self, messages: list[dict], tools: list[dict] | None, effort: str) -> list[int]:
        normalized = []
        for message in messages:
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(part["text"] for part in content)
            normalized.append(dict(message, content=content))
        return self.render_prompt(normalized, tools, enable_thinking=effort != "none")

    def parse_completion(self, tokens, tools, *, include_undeclared=False, effort="none") -> dict:
        if self.is_stop_token(tokens[-1]):
            tokens = tokens[:-1]
        return parse_completion(self.decode(tokens), tools, include_undeclared=include_undeclared)


def parse_completion(text: str, tools: list[dict], *, include_undeclared: bool = False) -> dict:
    message = {"role": "assistant", "content": None}
    text = text.lstrip()
    if text.startswith("<think>"):
        reasoning, _, text = text.removeprefix("<think>").partition("</think>")
        message["reasoning_content"] = reasoning.strip()

    content = []
    allowed = {tool["function"]["name"] for tool in tools}
    decoder = json.JSONDecoder()
    while "<tool_call>" in text:
        leading, _, body = text.partition("<tool_call>")
        content.append(leading)
        body = body.lstrip()
        try:
            call, end = decoder.raw_decode(body)
        except json.JSONDecodeError:
            _, _, text = body.partition("</tool_call>")
        else:
            closing = body[end:].lstrip()
            if not closing.startswith("</tool_call>"):
                _, _, text = closing.partition("</tool_call>")
                continue
            text = closing.removeprefix("</tool_call>")
            if isinstance(call, dict):
                name, arguments = call.get("name"), call.get("arguments")
                if (
                    isinstance(name, str)
                    and isinstance(arguments, dict)
                    and (include_undeclared or name in allowed)
                ):
                    message.setdefault("tool_calls", []).append(function_call(name, arguments))
    content.append(text)
    message["content"] = "".join(content).strip() or None
    return message
