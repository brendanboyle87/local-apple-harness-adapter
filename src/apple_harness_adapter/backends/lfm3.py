"""Pi-compatible LFM3 chat and the Hugging Face LoRA target set."""

from pathlib import Path

import mlx.nn as nn
from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call
from mlx_lm.utils import load_model, load_tokenizer

from ..endpoint import function_call
from .base import MLXPolicy
from .lfm3_model import Model, ModelArgs


def model_classes(config):
    if config.get("model_type") != "lfm3":
        raise ValueError("This backend expects an LFM3 checkpoint")
    return Model, ModelArgs


class LFM3Policy(MLXPolicy):
    backend = "lfm3"
    serving_name = "lfm3-1.7b"
    default_rank = 32
    lora_scale = 2.0

    @staticmethod
    def load_checkpoint(location: Path):
        model, config = load_model(location, get_model_classes=model_classes)
        tokenizer = load_tokenizer(location, eos_token_ids=config.get("eos_token_id"))
        return model, tokenizer

    def configure_model(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, module in self.model.named_modules()
            if name.startswith("model.layers.")
            and isinstance(module, (nn.Linear, nn.QuantizedLinear))
        )

    def prompt_ids(self, messages: list[dict], tools: list[dict] | None, effort: str) -> list[int]:
        return self.render_prompt(
            messages, tools, enable_thinking=effort != "none", preserve_thinking=True
        )

    def parse_completion(self, tokens, tools, *, include_undeclared=False, effort="none") -> dict:
        if self.is_stop_token(tokens[-1]):
            tokens = tokens[:-1]
        return parse_completion(
            self.decode(tokens),
            tools,
            thinking=effort != "none",
            include_undeclared=include_undeclared,
        )


def parse_completion(
    text: str, tools: list[dict], *, thinking=False, include_undeclared=False
) -> dict:
    message = {"role": "assistant", "content": None}
    text = text.lstrip()
    # The generation prompt already ends in <think> when reasoning is enabled.
    if thinking or text.startswith("<think>"):
        reasoning, _, text = text.removeprefix("<think>").partition("</think>")
        message["reasoning_content"] = reasoning

    allowed = {tool["function"]["name"] for tool in tools}
    content = []
    while "<tool_call>" in text:
        leading, _, body = text.partition("<tool_call>")
        content.append(leading)
        block, closing, text = body.partition("</tool_call>")
        if not closing:
            break
        try:
            call = parse_tool_call(block.strip(), tools)
        except ValueError:
            continue
        if include_undeclared or call["name"] in allowed:
            message.setdefault("tool_calls", []).append(
                function_call(call["name"], call["arguments"])
            )
    content.append(text)
    message["content"] = "".join(content).strip() or None
    return message
