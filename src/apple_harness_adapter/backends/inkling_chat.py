"""Decode Inkling message frames into a chat-completions message."""

import json
import re

from ..endpoint import function_call

FRAME = re.compile(r"<\|message_model\|>([^<]*?)<\|content_(text|thinking|invoke_tool_json)\|>")
END = "<|end_message|>"
EOS = "<|content_model_end_sampling|>"


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
                message.setdefault("tool_calls", []).append(function_call(name, invocation["args"]))
    if not frames:
        message["content"] = text.split(EOS, 1)[0].split(END, 1)[0]
    return message
