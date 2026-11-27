"""Extract and compare numeric efficiency measurements for the archived LFM3/Pi evaluation."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean, median, pstdev, quantiles

from run import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, interval95, source_resamples

ARMS = ("base", "local")
TOKENIZER_HASHES = {
    "tokenizer.json": "41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44",
    "tokenizer_config.json": "b77214b9f12be74b9e5e1dab35d7dace077a31be969314bf38866fd6c8163853",
    "chat_template.jinja": "306497f109439afcfa248f4ef5800601de24942e2981d632ae63fc5c070ceb12",
}
METRICS = {
    "completion_tokens": "tokens",
    "prompt_tokens": "tokens",
    "logical_tokens": "tokens",
    "rendered_prompt_tokens": "tokens",
    "pi_requests": "requests",
    "model_requests": "requests",
    "generation_requests": "requests",
    "agent_turns": "turns",
    "tool_calls": "calls",
    "pi_tool_calls": "calls",
    "tool_executions": "calls",
    "task_seconds": "s",
    "harbor_agent_seconds": "s",
    "model_request_seconds": "s",
    "tool_execution_seconds": "s",
    "peak_mlx_gib": "GiB",
    "reasoning_tokens": "tokens",
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check_hash(path, expected):
    with path.open("rb") as source:
        require(hashlib.file_digest(source, "sha256").hexdigest() == expected, path)


def render_prompt(tokenizer, messages, tools):
    normalized = []
    for original in messages:
        message = dict(original)
        if "reasoning" in message:
            message["reasoning_content"] = message.pop("reasoning")
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                if isinstance(function["arguments"], str):
                    function["arguments"] = json.loads(function["arguments"])
                calls.append(dict(call, function=function))
            message["tool_calls"] = calls
        normalized.append(message)
    rendered = tokenizer.apply_chat_template(
        normalized,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def extend_prompt(messages, history):
    """Replay the text-only Pi OpenAI Completions message conversion for this archive."""
    messages = list(messages)
    for message in history:
        blocks = message["content"]
        if message["role"] == "assistant":
            require(
                all(block["type"] in ("text", "thinking", "toolCall") for block in blocks),
                "Unexpected assistant block in recorded Pi request history",
            )
            text = "".join(block["text"] for block in blocks if block["type"] == "text")
            thinking = "\n".join(
                block["thinking"]
                for block in blocks
                if block["type"] == "thinking" and block["thinking"].strip()
            )
            calls = [
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": block["arguments"]},
                }
                for block in blocks
                if block["type"] == "toolCall"
            ]
            if text or calls:
                assistant = {"role": "assistant", "content": text or None}
                if thinking:
                    assistant["reasoning_content"] = thinking
                if calls:
                    assistant["tool_calls"] = calls
                messages.append(assistant)
        else:
            require(
                message["role"] == "toolResult" and all(b["type"] == "text" for b in blocks),
                "Unexpected tool result in recorded Pi request history",
            )
            text = "\n".join(block["text"] for block in blocks)
            messages.append(
                {
                    "role": "tool",
                    "content": text if text.strip() else "(no tool output)",
                    "tool_call_id": message["toolCallId"],
                }
            )
    return messages


def usage(message):
    counts = message["usage"]
    prompt, completion = counts["input"], counts["output"]
    require(counts["totalTokens"] == prompt + completion, "Inconsistent Pi token accounting")
    require(counts["cacheRead"] == counts["cacheWrite"] == 0, "Unexpected Pi cache accounting")
    require(counts["reasoning"] == 0, "Unexpected provider-reported reasoning usage")
    return prompt, completion


def duration_ms(message):
    value = message["durationMs"]
    require(type(value) is int and value >= 0, "Invalid Pi duration")
    return value


def duration_s(record):
    delta = datetime.fromisoformat(record["finished_at"]) - datetime.fromisoformat(
        record["started_at"]
    )
    require(delta.total_seconds() >= 0, "Inverted Harbor timestamps")
    return delta.total_seconds()


def thinking_length(prompt, completion, tokenizer):
    require(prompt[-1] == 124901, "Prompt does not open the LFM3 thinking region")
    closing = completion.index(124902) if 124902 in completion else len(completion)
    if closing == len(completion) and completion[-1] == 124900:
        closing -= 1
    thought = tokenizer.decode(completion[:closing], skip_special_tokens=False).lstrip()
    body = completion[:-1] if completion[-1] == 124900 else completion
    require(
        thought
        == tokenizer.decode(body, skip_special_tokens=False).lstrip().partition("</think>")[0],
        "Thinking delimiter does not fall on its recorded token boundary",
    )
    return closing, thought


def extract_attempt(directory, measurement, tokenizer, config):
    require(not list(directory.parent.glob(f"{directory.name}-*")), "Unexpected retry directory")
    require(read_json(directory / "measurement.json") == measurement, "Measurement changed")
    rollout = read_json(directory / "config.json")
    require(
        all(rollout[key] == config[key] for key in ("max_turns", "max_output", "max_context"))
        and rollout["collect_only"]
        and rollout["attempts"] == 1
        and rollout["seed"] == measurement["seed"]
        and rollout["effort"] == "low",
        "Unexpected historical rollout settings",
    )
    group = directory / "group-0001"
    (attempt_file,) = group.glob("attempt-*.json")
    attempt = read_json(attempt_file)
    metrics = read_json(group / "metrics.json")
    require(
        attempt["ok"]
        and attempt["atif"] == "match"
        and attempt["capture_level"] == "tokens"
        and attempt["trace_source"] == "atif"
        and attempt["exception_type"] is None
        and not measurement["agent_timed_out"]
        and not measurement["blocked_calls"]
        and attempt["reward"] == measurement["reward"] == metrics["correctness"][0]
        and measurement["tool_calls"] == metrics["tool_calls"][0]
        and measurement["peak_mlx_gib"] == metrics["collection_peak_mlx_gib"],
        "Attempt, measurement and group metrics do not reconcile",
    )
    trial = group / "trials" / attempt["trial_name"]
    result = read_json(trial / "result.json")
    trajectory = read_json(trial / "agent/trajectory.json")
    (session,) = (trial / "agent/pi/sessions").glob("*.jsonl")
    messages = [event["message"] for event in read_jsonl(session) if event["type"] == "message"]
    events = messages[2:]
    require(
        [message["role"] for message in messages[:2]] == ["system", "user"]
        and all(message["role"] in ("assistant", "toolResult") for message in events),
        "Unexpected intervening Pi messages",
    )
    replies = [i for i, message in enumerate(events) if message["role"] == "assistant"]
    generated = [i for i in replies if events[i]["responseId"].startswith("chatcmpl-")]
    stops = [i for i in replies if events[i]["responseId"].startswith("budget-")]
    require(
        len(set(events[i]["responseId"] for i in replies)) == len(replies)
        and len(replies) == len(generated) + len(stops)
        and len(generated) == attempt["n_turns"]
        and len(stops) == attempt["budget_stop_count"] <= 1
        and all(usage(events[i]) == (0, 0) for i in stops),
        "Pi replies and capture graph counts do not reconcile",
    )
    prompt_tokens, completion_tokens = map(sum, zip(*(usage(events[i]) for i in generated)))
    final = trajectory["final_metrics"]
    agent = result["agent_result"]
    require(
        (prompt_tokens, completion_tokens)
        == (final["total_prompt_tokens"], final["total_completion_tokens"])
        == (agent["n_input_tokens"], agent["n_output_tokens"]),
        "Pi usage, Harbor ATIF and Harbor agent usage differ",
    )

    turns = attempt["turns"]
    require(
        len({turn["node_id"] for turn in turns}) == len(turns)
        and sum(
            len(turn["tool_calls"])
            for turn in turns
            if turn["role"] == "agent" and not turn["discarded"]
        )
        == measurement["tool_calls"],
        "Capture node IDs or retained tool counts do not reconcile",
    )
    require(
        attempt["n_trainable_tokens"]
        == sum(
            sum(turn["loss_mask"])
            if turn["loss_mask"] is not None
            else len(turn["completion_token_ids"])
            for turn in turns
            if turn["trainable"] and not turn["discarded"]
        ),
        "OpenEnv training-mask accounting differs",
    )
    # Node IDs belong to OpenEnv. Align by the unique usage pairs within each Pi session.
    by_position = {}
    reasoning_retained = 0
    for turn in turns:
        prompt, completion = turn["prompt_token_ids"], turn["completion_token_ids"]
        (position,) = [i for i in generated if usage(events[i]) == (len(prompt), len(completion))]
        require(position not in by_position, "Multiple capture rows for the same Pi request")
        by_position[position] = turn
        require(
            render_prompt(tokenizer, turn["request_messages"], turn["request_tools"]) == prompt,
            "Pinned tokenizer could not reproduce the captured prompt IDs",
        )
        count, thought = thinking_length(prompt, completion, tokenizer)
        observed = "".join(
            b["thinking"] for b in events[position]["content"] if b["type"] == "thinking"
        )
        require(thought == observed, "Capture and Pi thinking text do not reconcile")
        reasoning_retained += count

    previous = None
    for position in generated:
        if previous is not None:
            preceding = by_position[previous]
            replay = render_prompt(
                tokenizer,
                extend_prompt(preceding["request_messages"], events[previous:position]),
                preceding["request_tools"],
            )
            if position in by_position:
                require(
                    replay == by_position[position]["prompt_token_ids"], "Prompt replay differs"
                )
            else:
                require(
                    len(replay) == usage(events[position])[0], "Discarded prompt length differs"
                )
        if position in by_position:
            previous = position

    findings = [f for f in attempt["findings"] if f.startswith("[WARN] context_budget_exhausted:")]
    clipped = 0
    rejected = 0
    recovered_prompt = 0
    recovered_completion = 0
    timed_generations = list(generated)
    if stops:
        (position,) = stops
        require(position == replies[-1], "Budget stop was not terminal")
        if findings:
            (finding,) = findings
            require(previous == generated[-1], "Context stop lacks the preceding captured request")
            preceding = by_position[previous]
            next_prompt = render_prompt(
                tokenizer,
                extend_prompt(preceding["request_messages"], events[previous:position]),
                preceding["request_tools"],
            )
            if "upstream returned 400" in finding:
                rejected = int(re.search(r"Prompt has (\d+) tokens", finding)[1])
                require(
                    rejected == len(next_prompt) >= config["max_context"], "Rejected count differs"
                )
            else:
                require("engine stopped at the context window" in finding, "Unknown context stop")
                clipped = 1
                recovered_prompt = len(next_prompt)
                recovered_completion = config["max_context"] - recovered_prompt
                require(0 < recovered_completion < config["max_output"], "Invalid clipped response")
                timed_generations.append(position)
        else:
            require(len(generated) == config["max_turns"], "Unexpected early model-call limit")
    else:
        require(not findings, "Missing Pi budget stop")

    tool_results = [message for message in events if message["role"] == "toolResult"]
    tool_ids = Counter(
        block["id"]
        for i in generated
        for block in events[i]["content"]
        if block["type"] == "toolCall"
    )
    require(
        tool_ids
        == Counter(message["toolCallId"] for message in tool_results)
        == Counter(call["call_id"] for call in attempt["safe_agents"]["calls"])
        and len(tool_results)
        == sum(len(step.get("tool_calls", [])) for step in trajectory["steps"]),
        "Pi tool calls, results, safety audit and Harbor ATIF counts differ",
    )
    tool_ms = []
    validation_rejections = 0
    for message in tool_results:
        if "durationMs" in message:
            tool_ms.append(duration_ms(message))
        else:
            require(
                message["isError"]
                and message["content"][0]["text"].startswith(
                    f'Validation failed for tool "{message["toolName"]}":'
                ),
                "A tool execution is missing its timing",
            )
            validation_rejections += 1

    full_prompt = prompt_tokens + recovered_prompt
    full_completion = completion_tokens + recovered_completion
    omitted = len(generated) - len(turns)
    return {
        "reward": measurement["reward"],
        "prompt_tokens": full_prompt,
        "completion_tokens": full_completion,
        "logical_tokens": full_prompt + full_completion,
        "rendered_prompt_tokens": full_prompt + rejected,
        "pi_requests": len(replies),
        "model_requests": len(timed_generations) + bool(rejected),
        "generation_requests": len(timed_generations),
        "agent_turns": len(generated),
        "tool_calls": measurement["tool_calls"],
        "pi_tool_calls": len(tool_results),
        "tool_executions": len(tool_ms),
        "task_seconds": measurement["seconds"],
        "harbor_agent_seconds": duration_s(result["agent_execution"]),
        "model_request_seconds": sum(duration_ms(events[i]) for i in timed_generations) / 1000,
        "tool_execution_seconds": sum(tool_ms) / 1000,
        "peak_mlx_gib": measurement["peak_mlx_gib"],
        "reasoning_tokens": reasoning_retained if not (omitted or clipped) else None,
        "audit": {
            "captured_turns": len(turns),
            "retained_trainable_tokens": attempt["n_trainable_tokens"],
            "captured_reasoning_tokens": reasoning_retained,
            "pi_prompt_tokens": prompt_tokens,
            "pi_completion_tokens": completion_tokens,
            "capture_omitted_turns": omitted,
            "capture_omitted_prompt_tokens": (
                prompt_tokens - sum(len(turn["prompt_token_ids"]) for turn in turns)
            ),
            "capture_omitted_completion_tokens": (
                completion_tokens - sum(len(turn["completion_token_ids"]) for turn in turns)
            ),
            "captured_discarded_turns": sum(turn["discarded"] for turn in turns),
            "captured_nontrainable_turns": sum(not turn["trainable"] for turn in turns),
            "reconstructed_prompt_tokens": recovered_prompt,
            "reconstructed_completion_tokens": recovered_completion,
            "rejected_prompt_tokens": rejected,
            "context_clipped_requests": clipped,
            "context_rejected_requests": int(bool(rejected)),
            "model_limit_stops": len(stops) - clipped - bool(rejected),
            "tool_validation_rejections": validation_rejections,
            "harbor_trial_seconds": duration_s(result),
            "openenv_wall_seconds": attempt["wall_s"],
            "collection_seconds": metrics["collection_seconds"],
        },
    }


def extract(archive, snapshot, historical, provenance):
    from transformers import AutoTokenizer

    for name, digest in TOKENIZER_HASHES.items():
        check_hash(snapshot / name, digest)
    raw = archive / "evaluation"
    hashes = provenance["original_artifacts_sha256"]
    for filename in ("evaluation/pairs.json", "evaluation/summary.json", "plan.json"):
        check_hash(archive / filename, hashes[f"lfm3-150/{filename}"])
    for filename in ("endpoint.py", "backends/lfm3.py", "backends/base.py", "cli.py", "harbor.py"):
        name = f"apple_harness_adapter/{filename}"
        check_hash(archive / "engine" / name, provenance["historical_engine_files_sha256"][name])
    require(
        read_json(raw / "pairs.json")
        == [{k: v for k, v in row.items() if k != "task_digest"} for row in historical],
        "Archived pairs differ from the published measurements",
    )
    plan = read_json(archive / "plan.json")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rows = []
    for pair in historical:
        measurements = {}
        # Also replay the original third arm to cross-check the accounting rules.
        for arm, measurement in pair["arms"].items():
            directory = raw / pair["task"] / arm
            try:
                data = extract_attempt(directory, measurement, tokenizer, plan)
            except (ValueError, KeyError, OSError) as error:
                raise ValueError(f"{pair['task']}/{arm}: {error}") from error
            if arm in ARMS:
                measurements[arm] = data
        rows.append(
            {
                "task": pair["task"],
                "bucket_prefix": pair["bucket_prefix"],
                "seed": pair["seed"],
                "arms": measurements,
            }
        )
    return rows


def percent_change(before, after):
    return 100 * (after / before - 1) if before else None


def distribution(values, unit):
    if not values:
        return None
    breaks = quantiles(values, n=20, method="inclusive") if len(values) > 1 else values * 19
    return {
        "total": sum(values) if unit != "GiB" else None,
        "mean": mean(values),
        "median": median(values),
        "sd": pstdev(values),
        "min": min(values),
        "p10": breaks[1],
        "p90": breaks[17],
        "p95": breaks[18],
        "max": max(values),
    }


def cohort_statistics(rows, metric, samples):
    groups = defaultdict(list)
    for source, before, after in rows:
        if before is not None and after is not None:
            groups[source].append((before, after))
    pairs = [pair for group in groups.values() for pair in group]
    before, after = zip(*pairs) if pairs else ((), ())
    changes, differences = [], []
    aggregates = {
        source: (len(group), sum(b for b, _ in group), sum(a for _, a in group))
        for source, group in groups.items()
    }
    for sample in samples:
        n, b, a = map(sum, zip(*(aggregates.get(source, (0, 0, 0)) for source in sample)))
        if n:
            differences.append((a - b) / n)
            if b:
                changes.append(percent_change(b, a))
    return {
        "pairs": len(pairs),
        "source_datasets": len(groups),
        "base": distribution(before, METRICS[metric]),
        "local": distribution(after, METRICS[metric]),
        "local_lower_tied_higher": [
            sum(a < b for b, a in pairs),
            sum(a == b for b, a in pairs),
            sum(a > b for b, a in pairs),
        ],
        "mean_difference": mean(after) - mean(before) if pairs else None,
        "mean_difference_ci95": interval95(differences),
        "mean_change_percent": percent_change(sum(before), sum(after)),
        "median_change_percent": percent_change(median(before), median(after)) if pairs else None,
        "mean_change_percent_ci95": interval95(changes),
        "valid_ratio_resamples": len(changes),
    }


def per_solve_statistics(rows, samples):
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for source, before, after, base_pass, local_pass in rows:
        values = groups[source]
        for i, value in enumerate((before, after, base_pass, local_pass)):
            values[i] += value
    base_total, local_total, base_passes, local_passes = map(sum, zip(*groups.values()))
    if not (base_passes and local_passes):
        return None
    base, local = base_total / base_passes, local_total / local_passes
    changes, differences = [], []
    for sample in samples:
        b, a, bp, ap = map(sum, zip(*(groups[source] for source in sample)))
        if bp and ap:
            differences.append(a / ap - b / bp)
            if b:
                changes.append(percent_change(b / bp, a / ap))
    return {
        "base": base,
        "local": local,
        "difference": local - base,
        "difference_ci95": interval95(differences),
        "change_percent": percent_change(base, local),
        "change_percent_ci95": interval95(changes),
        "valid_ratio_resamples": len(changes),
    }


def summarize_efficiency(rows, historical):
    sources = list(dict.fromkeys(row["bucket_prefix"] for row in historical))
    samples = list(source_resamples(sources))
    paired = []
    for row, original in zip(rows, historical, strict=True):
        require(
            all(row[key] == original[key] for key in ("task", "bucket_prefix", "seed"))
            and set(row["arms"]) == set(ARMS),
            "Pairs differ from the original ordered task list",
        )
        observations = row["arms"]
        measures = original["arms"]
        for arm in ARMS:
            for new, old in (
                ("reward", "reward"),
                ("tool_calls", "tool_calls"),
                ("task_seconds", "seconds"),
                ("peak_mlx_gib", "peak_mlx_gib"),
            ):
                require(observations[arm][new] == measures[arm][old], f"Original {old} differs")
            point = observations[arm]
            require(
                point["logical_tokens"] == point["prompt_tokens"] + point["completion_tokens"],
                "Inconsistent released token accounting",
            )
        outcomes = [measures[arm]["reward"] == 1 for arm in ARMS]
        paired.append((original["bucket_prefix"], observations, *outcomes))
    shared = [row for row in paired if row[2] and row[3]]
    cohorts = {}
    for label, observations in (("all_tasks", paired), ("shared_successes", shared)):
        cohorts[label] = {
            "tasks": len(observations),
            "passes": dict(zip(ARMS, map(sum, zip(*(row[2:] for row in observations))))),
            "metrics": {
                metric: cohort_statistics(
                    [
                        (source, points["base"][metric], points["local"][metric])
                        for source, points, _, _ in observations
                    ],
                    metric,
                    samples,
                )
                for metric in METRICS
            },
        }
    per_solve = {}
    for metric in METRICS:
        values = [
            (source, arms["base"][metric], arms["local"][metric], bp, ap)
            for source, arms, bp, ap in paired
        ]
        per_solve[metric] = (
            per_solve_statistics(values, samples)
            if metric != "peak_mlx_gib"
            and all(v[1] is not None and v[2] is not None for v in values)
            else None
        )
    return {
        "schema_version": 1,
        "comparison": "local_vs_base",
        "bootstrap": {
            "unit": "source_dataset",
            "source_datasets": len(sources),
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
        },
        "units": METRICS,
        "coverage": {
            arm: {
                key: sum(row["arms"][arm]["audit"][key] for row in rows)
                for key in rows[0]["arms"][arm]["audit"]
                if not key.endswith("_seconds")
            }
            | {
                "complete_reasoning_tasks": sum(
                    row["arms"][arm]["reasoning_tokens"] is not None for row in rows
                ),
            }
            for arm in ARMS
        },
        **cohorts,
        "per_success": {
            "tasks": len(paired),
            "passes": cohorts["all_tasks"]["passes"],
            "metrics": per_solve,
        },
    }


def plot_efficiency(results):
    """Draw the paired percent changes and intervals in the release's SVG format."""
    from math import ceil, floor

    measures = {
        "tool_calls": "Original tool calls",
        "model_requests": "Model requests",
        "prompt_tokens": "Logical prompt tokens",
        "completion_tokens": "Generated tokens",
        "logical_tokens": "Logical input and output",
        "task_seconds": "End-to-end time",
        "model_request_seconds": "Model request time",
        "tool_execution_seconds": "Tool execution time",
        "peak_mlx_gib": "MLX collection peak",
    }
    cohorts = (
        ("all_tasks", "All 150 task pairs"),
        ("shared_successes", "59 shared successes"),
        ("per_success", "Per successful task"),
    )

    def effect(cohort, metric):
        values = results[cohort]["metrics"][metric]
        if values is None:
            return None
        prefix = "" if cohort == "per_success" else "mean_"
        return values[f"{prefix}change_percent"], values[f"{prefix}change_percent_ci95"]

    effects = {
        (cohort, metric): effect(cohort, metric) for cohort, _ in cohorts for metric in measures
    }
    bounds = [n for item in effects.values() if item for n in item[1]]
    low, high = 20 * floor(min(bounds) / 20), 20 * ceil(max(bounds) / 20)
    height = 149 + 29 * len(measures)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-labelledby="title description" '
        'font-family="system-ui, sans-serif">',
        '<title id="title">LFM3 and Pi efficiency</title>',
        '<desc id="description">Paired percentage changes with 95% confidence intervals '
        "for the local adapter versus the base model. Negative values indicate lower use.</desc>",
        f'<rect width="960" height="{height}" fill="#fff"/>',
    ]

    def label(x, y, value, size=12, color="#1b2d3e", weight=400, anchor="start"):
        svg.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{value}</text>'
        )

    def line(x1, y1, x2, y2, color, width=1):
        svg.append(
            f'<line x1="{x1:.2f}" y1="{y1}" x2="{x2:.2f}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"/>'
        )

    label(28, 33, "LFM3 BF16 + Pi efficiency", size=22, weight=700)
    label(28, 57, "Local adapter change (%) with paired 95% confidence intervals", size=14)
    for i, (cohort, title) in enumerate(cohorts):
        left = 214 + 244 * i
        width = 210

        def scale(value):
            return left + width * (value - low) / (high - low)

        label(left + width // 2, 88, title, size=14, weight=600, anchor="middle")
        for tick in range(low, high + 1, 20):
            x = scale(tick)
            line(x, 111, x, height - 45, "#899aaa" if tick == 0 else "#dee5eb")
            label(round(x, 2), 105, f"{tick:+}" if tick else "0", size=10, anchor="middle")
        for j, (metric, name) in enumerate(measures.items()):
            y = 128 + 29 * j
            if i == 0:
                label(28, y + 4, name)
            value = effects[cohort, metric]
            if value is None:
                svg.append(
                    f'<rect x="{left - 3}" y="{y - 11}" width="{width + 6}" height="22" fill="#fff"/>'
                )
                label(left + width // 2, y + 4, "—", color="#48596a", anchor="middle")
                continue
            change, (lower, upper) = value
            color = "#536578" if lower <= 0 <= upper else "#146460" if upper < 0 else "#9b4b2e"
            lo, hi, mid = scale(lower), scale(upper), scale(change)
            svg.append(f"<g><title>{name} {change:+.2f}% [{lower:+.2f}%, {upper:+.2f}%]</title>")
            line(lo, y, hi, y, color, 2.5)
            line(lo, y - 4, lo, y + 4, color, 1.5)
            line(hi, y - 4, hi, y + 4, color, 1.5)
            svg.append(f'<circle cx="{mid:.2f}" cy="{y}" r="4.1" fill="{color}"/></g>')
    label(
        28,
        height - 19,
        "Negative values indicate lower use. Per-success totals include attempts that failed.",
        color="#48596a",
    )
    return "\n".join(svg + ["</svg>", ""])
