from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def efficiency(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples/smoldataenvs"))
    monkeypatch.syspath_prepend(str(root / "results/lfm3-pi"))
    return import_module("efficiency")


def test_per_success_charges_failed_attempts_and_recomputes_pass_counts(efficiency):
    rows = [
        ("a", 10, 20, True, True),
        ("a", 70, 30, False, True),
        ("b", 20, 10, True, False),
    ]
    result = efficiency.per_solve_statistics(rows, [["a", "b"], ["a", "a"], ["b", "b"]])
    assert (result["base"], result["local"]) == (50, 30)
    assert result["change_percent"] == -40
    assert result["valid_ratio_resamples"] == 2
    assert result["change_percent_ci95"] == efficiency.interval95([-40, -68.75])


def test_paired_reasoning_uses_complete_pairs_without_filling_missing_values(efficiency):
    rows = [("a", 100, None), ("b", 10, 20), ("b", None, 100)]
    result = efficiency.cohort_statistics(rows, "reasoning_tokens", [["a"], ["b", "b"], ["a", "b"]])
    assert result["pairs"] == 1
    assert (result["base"]["total"], result["local"]["total"]) == (10, 20)
    assert result["valid_ratio_resamples"] == 2
    assert result["mean_change_percent_ci95"] == [100, 100]
    missing = efficiency.cohort_statistics(rows[:1], "reasoning_tokens", [["a"], ["a"]])
    assert missing["base"] is missing["local"] is missing["mean_change_percent_ci95"] is None


def test_evaluation_summary_keeps_old_rows_and_exposes_incomplete_coverage(efficiency, monkeypatch):
    run = import_module("run")
    monkeypatch.setattr(run, "source_resamples", lambda _sources: iter([["a", "b"], ["a", "a"]]))
    rows = [
        {
            "source": "a",
            "base": {
                "reward": 1,
                "calls": 1,
                "task_seconds": 2,
                "efficiency": {"prompt_tokens": 5},
            },
            "adapter": {
                "reward": 1,
                "calls": 1,
                "task_seconds": 1,
                "efficiency": {"prompt_tokens": 3},
            },
        },
        {
            "source": "b",
            "base": {
                "reward": 0,
                "calls": 1,
                "task_seconds": 4,
                "efficiency": {"prompt_tokens": 5},
            },
            "adapter": {"reward": 1, "calls": 1, "task_seconds": 2},
        },
    ]
    summary = run.summarize_measured_resources(rows)
    all_tasks = summary["all_tasks"]
    assert all_tasks["arms"]["adapter"]["metrics"]["prompt_tokens"]["observed_attempts"] == 1
    assert all_tasks["arms"]["adapter"]["metrics"]["prompt_tokens"]["total"] is None
    assert all_tasks["paired"]["prompt_tokens"]["mean_change_ci95"] is None
    assert all_tasks["paired"]["task_seconds"]["mean_change_percent"] == -50
    assert all_tasks["paired"]["task_seconds"]["per_success_change_percent"] == -75
    assert summary["shared_successes"]["paired"]["prompt_tokens"]["mean_change_percent"] == -40
