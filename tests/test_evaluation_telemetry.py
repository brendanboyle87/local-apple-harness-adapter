import json

from apple_harness_adapter.telemetry import AttemptTelemetry, pi_spans


def test_request_memory_scopes_preserve_attempt_and_group_peaks(tmp_path):
    class Memory:
        active = 100
        peak = 900

        def get_active_memory(self):
            return self.active

        def get_peak_memory(self):
            return self.peak

        def reset_peak_memory(self):
            self.peak = 0

    memory = Memory()
    collector = AttemptTelemetry(tmp_path / "requests.jsonl", memory)
    first = collector.start_model_request(0)
    memory.peak = 180
    collector.begin_mlx_scope(first)
    memory.peak = 330
    collector.end_mlx_scope(first)

    second = collector.start_model_request(1)
    collector.begin_mlx_scope(second)
    memory.peak = 150
    collector.end_mlx_scope(second)
    second.sampling_started = True
    second.rendered_prompt_tokens = 32
    result = collector.finish(None, None, "pi")

    assert first.mlx_active_before_bytes == 100 and first.mlx_peak_bytes == 330
    assert second.mlx_peak_bytes == 150
    assert result["mlx_attempt_peak_bytes"] == result["mlx_max_request_peak_bytes"] == 330
    assert result["mlx_group_peak_to_date_bytes"] == 900
    assert result["prompt_tokens"] == 32
    assert result["completion_tokens"] is result["logical_tokens"] is None
    assert result["tool_execution_seconds"] is None


def test_pi_spans_separate_rejected_tools_from_missing_results(tmp_path):
    directory = tmp_path / "agent/pi/sessions"
    directory.mkdir(parents=True)
    messages = [
        {
            "role": "assistant",
            "responseId": "chatcmpl-test",
            "content": [
                {"type": "toolCall", "id": name, "name": "write"}
                for name in ("executed", "rejected", "missing")
            ],
        },
        {
            "role": "assistant",
            "responseId": "budget-test",
            "content": [],
        },
        {
            "role": "toolResult",
            "toolCallId": "executed",
            "durationMs": 0,
            "isError": False,
        },
        {
            "role": "toolResult",
            "toolCallId": "rejected",
            "isError": True,
        },
    ]
    rows = [{"type": "session"}] + [{"type": "message", "message": m} for m in messages]
    (directory / "session.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    turns, calls = pi_spans(tmp_path)
    assert turns == 1
    assert [call["outcome"] for call in calls] == ["executed", "rejected", "unobserved"]
    assert [call["duration_ms"] for call in calls] == [0, None, None]
