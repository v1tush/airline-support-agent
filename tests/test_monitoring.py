from __future__ import annotations

import pytest

from alisa.monitoring import (
    BudgetExceededError,
    MonitoringCompletions,
    SessionMonitor,
    estimate_cost,
    model_name,
)

QWEN = "qwen3.6-35b-a3b"
QWEN_URI = "gpt://b1gexample/qwen3.6-35b-a3b"
FAST = "qwen3.6-35b-a3b"
BIG = "qwen3-235b-a22b-fp8"


class _FakeDetails:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, cached_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _FakeDetails(cached_tokens)


class _FakeResponse:
    def __init__(self, usage: _FakeUsage):
        self.usage = usage


class _FakeInnerCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_model_name_strips_folder_id_and_version_suffix():
    assert model_name(QWEN_URI) == QWEN
    assert model_name("gpt://b1gexample/yandexgpt-5-lite/latest") == "yandexgpt-5-lite"
    assert model_name("emb://b1gexample/text-search-doc/latest") == "text-search-doc"
    # У других OpenAI-совместимых провайдеров модель называется просто
    # по имени - такая строка должна проходить без изменений.
    assert model_name(FAST) == FAST


def test_estimate_cost_known_model_matches_manual_calculation():
    usage = _FakeUsage(prompt_tokens=1000, completion_tokens=500)
    cost = estimate_cost(usage, QWEN)
    # 1000 * 0.2/1000 + 500 * 0.3/1000 = 0.2 + 0.15 = 0.35 ₽
    assert cost == pytest.approx(0.35)


def test_estimate_cost_works_with_full_model_uri():
    # В запрос уходит URI с идентификатором каталога, а PRICING знает
    # только имя модели - цену это менять не должно.
    usage = _FakeUsage(prompt_tokens=1000, completion_tokens=500)
    assert estimate_cost(usage, QWEN_URI) == pytest.approx(estimate_cost(usage, QWEN))


def test_estimate_cost_does_not_double_count_cached_tokens():
    # prompt_tokens включает кэшированные токены. 1000 входных, из них
    # 800 из кэша: 200 * 0.2/1000 + 800 * 0.05/1000 = 0.04 + 0.04 = 0.08 ₽.
    usage = _FakeUsage(prompt_tokens=1000, cached_tokens=800)
    assert estimate_cost(usage, QWEN) == pytest.approx(0.08)


def test_estimate_cost_cached_tokens_are_cheaper_than_fresh_input():
    usage_no_cache = _FakeUsage(prompt_tokens=1000)
    usage_all_cached = _FakeUsage(prompt_tokens=1000, cached_tokens=1000)
    assert estimate_cost(usage_all_cached, QWEN) < estimate_cost(usage_no_cache, QWEN)


def test_estimate_cost_unknown_model_is_unknown():
    usage = _FakeUsage(prompt_tokens=1000)
    assert estimate_cost(usage, "some-future-model") is None
    assert estimate_cost(usage, BIG) is None


def test_session_monitor_accumulates_cost_across_calls():
    monitor = SessionMonitor()
    monitor.record(
        QWEN, _FakeUsage(prompt_tokens=1000, completion_tokens=500), latency_seconds=0.1
    )
    monitor.record(
        QWEN, _FakeUsage(prompt_tokens=1000, completion_tokens=500), latency_seconds=0.1
    )

    assert len(monitor.log) == 2
    assert monitor.total_cost_rub == pytest.approx(0.7)


def test_session_monitor_check_budget_raises_when_exceeded():
    monitor = SessionMonitor(budget_rub=0.1)
    monitor.record(QWEN, _FakeUsage(prompt_tokens=100_000), latency_seconds=0.1)

    assert monitor.unknown_usage_calls == 0
    assert monitor.total_cost_rub == pytest.approx(20.0)

    with pytest.raises(BudgetExceededError):
        monitor.check_budget()


def test_session_monitor_check_budget_passes_when_under_limit():
    monitor = SessionMonitor(budget_rub=10.0)
    monitor.record(FAST, _FakeUsage(prompt_tokens=100), latency_seconds=0.1)
    monitor.check_budget()  # не должно бросить исключение


def test_monitoring_completions_records_usage_and_forwards_response():
    expected_response = _FakeResponse(
        _FakeUsage(prompt_tokens=10, completion_tokens=20)
    )
    inner = _FakeInnerCompletions([expected_response])
    monitor = SessionMonitor()
    monitored = MonitoringCompletions(inner, monitor)

    response = monitored.create(model=QWEN, messages=[])

    assert (
        response is expected_response
    )  # декоратор не подменяет и не оборачивает сам ответ
    assert len(monitor.log) == 1
    assert monitor.log[0].model == QWEN
    assert len(inner.calls) == 1


def test_monitoring_completions_stops_before_call_when_budget_already_exceeded():
    inner = _FakeInnerCompletions([_FakeResponse(_FakeUsage(prompt_tokens=10))])
    monitor = SessionMonitor(budget_rub=0.0000001)
    monitor.record(QWEN, _FakeUsage(prompt_tokens=1000), latency_seconds=0.1)
    assert monitor.unknown_usage_calls == 0
    monitored = MonitoringCompletions(inner, monitor)

    with pytest.raises(BudgetExceededError):
        monitored.create(model=QWEN, messages=[])

    assert len(inner.calls) == 0  # не потратили ни одного лишнего вызова
