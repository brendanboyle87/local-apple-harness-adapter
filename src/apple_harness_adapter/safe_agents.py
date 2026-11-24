"""Review Pi tool calls and record trusted per-attempt verdicts."""

from dataclasses import dataclass, field

from atif_scan import Context, Severity, Status, parse_trace
from safeagents.constants import ATIF_SCHEMA_VERSION, TRACE_QUALITY
from safeagents.guard import Blocked, Config, Guard, call_parts
from safeagents.rewrite import append_content
from safeagents.scan import Scanner, Signal, call_step


class TrainingScanner(Scanner):
    """Use the upstream checks and surface scanner errors to the rollout collector."""

    def signals(self, steps: list[dict], calls: list[tuple[str, str]]) -> list[Signal]:
        # Adapted from SafeAgents Scanner.signals. Its proxy can log scanner
        # errors and carry on, but a training trace must record a valid verdict.
        trace = parse_trace(
            {
                "schema_version": ATIF_SCHEMA_VERSION,
                "steps": [
                    {"step_id": index, **step}
                    for index, step in enumerate([*steps, call_step(calls)], start=1)
                ],
            }
        )
        assessments = self.engine.evaluate(trace, Context())
        errors = [item.spec.id for item in assessments if item.result.status == Status.ERROR]
        if errors:
            raise RuntimeError(f"atif-scan could not evaluate: {', '.join(errors)}")

        pending = (len(steps), len(calls) - 1)
        return [
            Signal(item.spec.id, item.spec.severity, item.spec.title)
            for item in assessments
            if item.counts
            and item.spec.severity > Severity.INFO
            and not item.spec.id.startswith(TRACE_QUALITY)
            and any((evidence.step, evidence.call) == pending for evidence in item.result.evidence)
        ]


@dataclass
class SafetyAudit:
    mode: str
    calls: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def blocked_tool_calls(self) -> int:
        return sum(call["verdict"] == "blocked" for call in self.calls)

    async def review_response(self, request: dict, response: dict, config: Config) -> None:
        # The local endpoint returns one non-streaming choice. OpenEnv replays it for Pi.
        choice = response["choices"][0]
        message = choice["message"]
        if not message.get("tool_calls"):
            return

        guard = Guard(request, config)
        kept, notices = [], []
        for call in message["tool_calls"]:
            name, arguments = call_parts(call)
            review = await guard.review(name, arguments)
            verdict = review.verdict
            if verdict == "blocked" and self.mode == "log-only":
                verdict = "flagged"
            self.calls.append(
                {
                    "response_id": response["id"],
                    "call_id": call["id"],
                    "tool": name,
                    "verdict": verdict,
                    "reason": review.reason,
                    "checks": [signal.id for signal in review.signals],
                }
            )
            if verdict == "blocked":
                notices.append(Blocked(name, review.reason).notice())
            else:
                kept.append(call)

        if notices:
            # These edits affect Pi's message only. Sampled IDs, probabilities and usage
            # stay on the response so the trainer can score the attempted actions.
            if kept:
                message["tool_calls"] = kept
            else:
                del message["tool_calls"]
                if choice["finish_reason"] == "tool_calls":
                    choice["finish_reason"] = "stop"
            for notice in notices:
                append_content(message, notice)
