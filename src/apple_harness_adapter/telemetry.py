"""Numeric observations for collect-only Harbor attempts."""

import json
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

ATTEMPT = ContextVar("evaluation_attempt", default=None)
RELAY_ID = ContextVar("evaluation_relay_id", default=None)
MODEL_REQUEST = ContextVar("evaluation_model_request", default=None)


# An absent span means its value is unknown. Empty, observed spans sum to zero.
def observed_sum(values):
    items = list(values)
    return None if any(item is None for item in items) else sum(items)


@dataclass
class ModelRequest:
    relay_id: int | None
    endpoint_id: int
    outcome: str = "error"
    http_status: int | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    limit_reason: str | None = None
    rendered_prompt_tokens: int | None = None
    sampled_tokens: int | None = None
    observed_sampled_tokens: int | None = None
    reasoning_tokens: int | None = None
    sampling_started: bool = False
    prefill_processed_tokens: int | None = None
    prefill_reused_tokens: int | None = None
    tokenization_seconds: float | None = None
    prefill_to_first_token_seconds: float | None = None
    decode_seconds: float | None = None
    inference_seconds: float | None = None
    postprocess_seconds: float | None = None
    safety_review_seconds: float | None = None
    endpoint_seconds: float | None = None
    mlx_active_before_bytes: int | None = None
    mlx_peak_bytes: int | None = None


def relay_response_id(response):
    """Return the OpenAI ID from an assembled JSON or SSE reply."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        payloads = (
            line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")
        )
    else:
        payloads = (response.text,)
    for payload in payloads:
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        if isinstance(event, dict) and isinstance(event.get("id"), str):
            return event["id"]
    return None


def pi_spans(trial_dir):
    """Return delivered model turns and Pi's tool spans, when the session was uploaded."""
    if trial_dir is None:
        return None, None
    sessions = list((trial_dir / "agent/pi/sessions").glob("*.jsonl"))
    if len(sessions) != 1:
        return None, None
    calls = {}
    results = {}
    turns = 0
    saw_session = False
    try:
        with sessions[0].open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event["type"] == "session":
                    saw_session = True
                if event["type"] != "message":
                    continue
                message = event["message"]
                if message["role"] == "assistant":
                    response_id = message.get("responseId") or ""
                    turns += response_id.startswith("chatcmpl-")
                    for block in message["content"]:
                        if block["type"] == "toolCall":
                            calls[block["id"]] = (block["name"], response_id or None)
                elif message["role"] == "toolResult":
                    results[message["toolCallId"]] = message
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None, None
    if not saw_session:
        return None, None
    if not results.keys() <= calls.keys():
        return turns, None
    spans = []
    for call_id, (name, response_id) in calls.items():
        result = results.get(call_id)
        duration = result.get("durationMs") if result else None
        if type(duration) is int and duration >= 0 and type(result.get("isError")) is bool:
            outcome = "execution_error" if result["isError"] else "executed"
        elif result and "durationMs" not in result and result.get("isError") is True:
            # Pi records immediate errors from call preparation without an execution span.
            outcome = "rejected"
        else:
            outcome = "unobserved"
            duration = None
        spans.append(
            {
                "call_id": call_id,
                "response_id": response_id,
                "tool": name,
                "outcome": outcome,
                "duration_ms": duration,
            }
        )
    return turns, spans


class AttemptTelemetry:
    def __init__(self, journal: Path, memory):
        self.journal = journal
        self.journal.touch(exist_ok=False)
        self.memory = memory
        self.started = monotonic()
        self.model_requests = []
        self.relay_requests = []
        self.group_peak_before = memory.get_peak_memory()
        self.active_before = memory.get_active_memory()
        self.peak = self.active_before
        memory.reset_peak_memory()

    def append(self, event):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")

    def start_model_request(self, relay_id):
        request = ModelRequest(relay_id, len(self.model_requests))
        self.model_requests.append(request)
        return request

    def begin_mlx_scope(self, request):
        self.peak = max(self.peak, self.memory.get_peak_memory())
        request.mlx_active_before_bytes = self.memory.get_active_memory()
        self.memory.reset_peak_memory()

    def end_mlx_scope(self, request):
        request.mlx_peak_bytes = max(request.mlx_active_before_bytes, self.memory.get_peak_memory())
        self.peak = max(self.peak, request.mlx_peak_bytes)

    def record_endpoint(self, request):
        self.append({"stage": "endpoint"} | asdict(request))

    def record_relay(self, relay_id, elapsed, response, outcome):
        response_id = relay_response_id(response) if response is not None else None
        row = {
            "stage": "relay",
            "relay_id": relay_id,
            "relay_seconds": elapsed,
            "http_status": response.status_code if response is not None else None,
            "response_id": response_id,
            "budget_reply": response_id.startswith("budget-") if response_id else None,
            "outcome": outcome,
        }
        self.relay_requests.append(row)
        self.append(row)

    def finish(self, result, trial_dir, harness):
        elapsed = monotonic() - self.started
        self.peak = max(self.peak, self.memory.get_peak_memory())
        generated = [request for request in self.model_requests if request.sampling_started]
        prompt = observed_sum(request.rendered_prompt_tokens for request in generated)
        completion = observed_sum(request.sampled_tokens for request in generated)
        rendered = sum(
            request.rendered_prompt_tokens
            for request in self.model_requests
            if request.rendered_prompt_tokens is not None
        )
        turns, tools = pi_spans(trial_dir) if harness == "pi" else (None, None)
        executions = (
            None
            if tools is None or any(tool["outcome"] == "unobserved" for tool in tools)
            else [tool for tool in tools if tool["duration_ms"] is not None]
        )
        if turns is None and result is not None and harness != "pi":
            turns = sum(turn.role == "agent" and not turn.discarded for turn in result.turns)
        retained_calls = (
            sum(
                len(turn.tool_calls)
                for turn in result.turns
                if turn.role == "agent" and not turn.discarded
            )
            if result is not None
            else None
        )
        return {
            "schema_version": 1,
            "attempt_seconds": elapsed,
            "relay_requests": len(self.relay_requests),
            "backend_requests": len(self.model_requests),
            "generation_requests": len(generated),
            "agent_turns": turns,
            "agent_turns_source": (
                None if turns is None else "pi_session" if harness == "pi" else "retained_capture"
            ),
            "tool_calls": retained_calls,
            "pi_tool_calls": len(tools) if tools is not None else None,
            "tool_executions": len(executions) if executions is not None else None,
            "tool_execution_seconds": (
                sum(tool["duration_ms"] for tool in executions) / 1000
                if executions is not None
                else None
            ),
            "rendered_prompt_tokens": rendered,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "observed_completion_tokens": observed_sum(
                request.observed_sampled_tokens for request in generated
            ),
            "logical_tokens": prompt + completion
            if prompt is not None and completion is not None
            else None,
            "reasoning_tokens": observed_sum(request.reasoning_tokens for request in generated),
            "prefill_processed_tokens": observed_sum(
                request.prefill_processed_tokens if request.sampled_tokens is not None else None
                for request in generated
            ),
            "prefill_reused_tokens": observed_sum(
                request.prefill_reused_tokens for request in generated
            ),
            "prefill_to_first_token_seconds": observed_sum(
                request.prefill_to_first_token_seconds for request in generated
            ),
            "decode_seconds": observed_sum(request.decode_seconds for request in generated),
            "inference_seconds": observed_sum(request.inference_seconds for request in generated),
            "generation_relay_seconds": self.generation_relay_seconds(generated),
            "endpoint_seconds": observed_sum(
                request.endpoint_seconds for request in self.model_requests
            ),
            "mlx_active_before_bytes": self.active_before,
            "mlx_attempt_peak_bytes": self.peak,
            "mlx_group_peak_to_date_bytes": max(self.group_peak_before, self.peak),
            "mlx_max_request_peak_bytes": max(
                (
                    request.mlx_peak_bytes
                    for request in self.model_requests
                    if request.mlx_peak_bytes is not None
                ),
                default=None,
            ),
            "tool_spans": tools,
        }

    def generation_relay_seconds(self, requests):
        by_id = {row["relay_id"]: row["relay_seconds"] for row in self.relay_requests}
        # Count a relay span once in case it forwarded more than one backend request.
        return observed_sum(
            by_id.get(relay_id) for relay_id in dict.fromkeys(r.relay_id for r in requests)
        )
