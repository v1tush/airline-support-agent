from __future__ import annotations

import json
import sys
from types import SimpleNamespace as NS

import pytest

from alisa import chat_demo
from alisa.agent import run_agent_turn
from alisa.cache import CachingCompletions
from alisa.monitoring import (
    BudgetExceededError,
    MonitoringCompletions,
    SessionMonitor,
)
from alisa.parser import as_client

MODEL = "gpt://folder/qwen3.6-35b-a3b/latest"
USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "prompt_tokens_details": {"cached_tokens": 0},
}


def response(text="ok", *, reason="stop", usage=USAGE):
    return {
        "choices": [{"finish_reason": reason, "message": {"content": text}}],
        "usage": usage,
    }


class CountingCompletions:
    def __init__(self, actions=None):
        self.calls = 0
        self.actions = list(actions or [])

    def create(self, **kwargs):
        self.calls += 1
        if self.actions:
            action = self.actions.pop(0)
            if isinstance(action, Exception):
                raise action
            return action
        return response(f"answer-{self.calls}")


def test_cache_and_monitor_together_count_only_misses():
    inner = CountingCompletions()
    monitor = SessionMonitor()
    cache = CachingCompletions(MonitoringCompletions(inner, monitor))
    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Найди рейс"}],
        "max_tokens": 100,
        "reasoning_effort": "medium",
        "tools": [{"type": "function", "function": {"description": "искать рейсы"}}],
    }

    first = cache.create(**request)
    first["choices"][0]["message"]["content"] = "испорчено"
    second = cache.create(**request)
    assert second["choices"][0]["message"]["content"] == "answer-1"
    assert (inner.calls, cache.hits, cache.misses, len(monitor.log)) == (1, 1, 1, 1)
    assert monitor.total_cost_rub == pytest.approx(0.35)

    cache.create(**{**request, "max_tokens": 101})
    cache.create(**{**request, "reasoning_effort": "none"})
    cache.create(
        **{
            **request,
            "tools": [
                {"type": "function", "function": {"description": "искать правила"}}
            ],
        }
    )
    assert (inner.calls, cache.hits, cache.misses, len(monitor.log)) == (4, 1, 4, 4)
    print(
        "CACHE_MONITOR "
        + json.dumps(
            {
                "inner_calls": inner.calls,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "monitor_entries": len(monitor.log),
                "known_cost_rub": monitor.total_cost_rub,
            }
        )
    )


def test_failed_and_truncated_responses_are_not_cached():
    inner = CountingCompletions([ConnectionError("сеть недоступна"), response("ready")])
    cache = CachingCompletions(inner)
    request = {"model": MODEL, "messages": [{"role": "user", "content": "тест"}]}
    with pytest.raises(ConnectionError):
        cache.create(**request)
    assert cache.create(**request)["choices"][0]["message"]["content"] == "ready"
    assert (inner.calls, cache.hits, cache.misses) == (2, 0, 2)

    truncated = CountingCompletions([response("часть", reason="length")] * 2)
    cache = CachingCompletions(truncated)
    for _ in range(2):
        with pytest.raises(ValueError, match="length"):
            cache.create(**request)
    assert (truncated.calls, cache.hits, cache.misses) == (2, 0, 2)
    print(
        "FAILED_CACHE "
        + json.dumps(
            {"network_attempts": inner.calls, "length_attempts": truncated.calls}
        )
    )


def test_missing_usage_is_unknown_and_soft_budget_stops_next_call():
    failed_inner = CountingCompletions([ConnectionError("сеть недоступна")])
    failed_monitor = SessionMonitor()
    failed = MonitoringCompletions(failed_inner, failed_monitor)
    with pytest.raises(ConnectionError):
        failed.create(model=MODEL, messages=[])
    assert failed_monitor.unknown_usage_calls == 1
    assert failed_monitor.log[0].cost_rub is None
    with pytest.raises(BudgetExceededError):
        failed.create(model=MODEL, messages=[])
    assert failed_inner.calls == 1

    budget_inner = CountingCompletions()
    budget_monitor = SessionMonitor(budget_rub=0.30)
    budgeted = MonitoringCompletions(budget_inner, budget_monitor)
    budgeted.create(model=MODEL, messages=[])
    assert budget_monitor.total_cost_rub == pytest.approx(0.35)
    with pytest.raises(BudgetExceededError):
        budgeted.create(model=MODEL, messages=[])
    assert budget_inner.calls == 1
    print(
        "BUDGET "
        + json.dumps(
            {
                "limit_rub": budget_monitor.budget_rub,
                "cost_after_first_call_rub": budget_monitor.total_cost_rub,
                "inner_calls": budget_inner.calls,
                "failed_call_cost_rub": failed_monitor.log[0].cost_rub,
                "unknown_usage_calls": failed_monitor.unknown_usage_calls,
            }
        )
    )


@pytest.mark.parametrize(
    ("action", "fragment"),
    [
        (ConnectionError("сеть недоступна"), "Запрос не завершён"),
        (response("частичный текст", reason="length"), "Незавершённый ответ: length"),
    ],
)
def test_chat_reports_api_failure_and_truncation(monkeypatch, capsys, action, fragment):
    inner = CountingCompletions([action])
    monkeypatch.setattr(chat_demo, "build_client", lambda: as_client(inner))
    monkeypatch.setattr(sys, "argv", ["chat_demo", "--today", "2026-07-21"])
    answers = iter(["Найди рейс", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    chat_demo.main()
    output = capsys.readouterr().out
    assert fragment in output
    assert "Подходящих рейсов нет" not in output
    assert inner.calls == 1
    print("CHAT_FAILURE " + json.dumps({"response": output.strip()}))


def test_chat_reports_soft_budget_after_last_allowed_call(monkeypatch, capsys):
    first = NS(
        choices=[
            NS(
                finish_reason="stop",
                message=NS(content="Первый ответ", tool_calls=None, refusal=None),
            )
        ],
        usage=USAGE,
    )
    inner = CountingCompletions([first])
    monkeypatch.setattr(chat_demo, "build_client", lambda: as_client(inner))
    monkeypatch.setattr(
        "alisa.monitoring.SessionMonitor",
        lambda budget_rub: SessionMonitor(budget_rub=0.30),
    )
    monkeypatch.setattr(sys, "argv", ["chat_demo", "--today", "2026-07-21"])
    answers = iter(["Привет", "Привет ещё раз", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    chat_demo.main()
    output = capsys.readouterr().out
    assert "Алиса: Первый ответ" in output
    assert "Запрос не завершён: BudgetExceededError" in output
    assert "Известный расход, ₽: 0.35" in output
    assert inner.calls == 1
    print(
        "CHAT_BUDGET "
        + json.dumps({"response": output.strip(), "inner_calls": inner.calls})
    )


def test_agent_step_limit_returns_failure_to_user():
    class RepeatingTool:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            tool = NS(
                id=f"t{self.calls}",
                function=NS(
                    name="search_flights",
                    arguments=json.dumps({"from_city": "MOW", "to_city": "AER"}),
                ),
            )
            return NS(
                choices=[
                    NS(
                        finish_reason="tool_calls",
                        message=NS(content=None, tool_calls=[tool], refusal=None),
                    )
                ]
            )

    inner = RepeatingTool()
    result = run_agent_turn(
        "Рейсы Москва — Сочи",
        as_client(inner),
        guard_client=False,
        max_tool_iterations=2,
    )
    assert inner.calls == 2
    assert result["error"] == "tool_limit"
    assert "Не удалось получить ответ" in result["final_text"]
    print(
        "STEP_LIMIT "
        + json.dumps(
            {"app_calls": inner.calls, "user_text": result["final_text"]},
        )
    )
