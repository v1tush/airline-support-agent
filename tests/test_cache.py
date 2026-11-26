from __future__ import annotations

from alisa.cache import CachingCompletions, cache_savings_from_usage, cached_tokens

MODEL = "gpt://folder/qwen3.6-35b-a3b"


class _FakeResponse:
    def __init__(self, text: str):
        from types import SimpleNamespace as NS

        self.text = text
        self.choices = [
            NS(
                finish_reason="stop",
                message=NS(content=text, tool_calls=None, refusal=None),
            )
        ]


class _FakeInnerCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(f"response-{len(self.calls)}")


def test_repeated_identical_query_hits_cache_without_calling_inner():
    inner = _FakeInnerCompletions()
    caching = CachingCompletions(inner)

    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "рейс из Москвы в Сочи"},
        ],
        "tools": [{"type": "function", "function": {"name": "search_flights"}}],
    }

    first = caching.create(**kwargs)
    second = caching.create(**kwargs)

    assert first.text == second.text
    assert first is not second  # изменение ответа не испортит сохранённый объект
    assert len(inner.calls) == 1
    assert caching.hits == 1
    assert caching.misses == 1


def test_different_query_is_a_cache_miss():
    inner = _FakeInnerCompletions()
    caching = CachingCompletions(inner)
    tools = [{"type": "function", "function": {"name": "search_flights"}}]

    caching.create(
        model=MODEL, tools=tools, messages=[{"role": "user", "content": "рейс в Сочи"}]
    )
    caching.create(
        model=MODEL,
        tools=tools,
        messages=[{"role": "user", "content": "рейс в Казань"}],
    )

    assert len(inner.calls) == 2
    assert caching.hits == 0
    assert caching.misses == 2


def test_shared_cache_dict_is_reused_across_two_instances():
    inner_a = _FakeInnerCompletions()
    inner_b = _FakeInnerCompletions()
    shared_cache: dict = {}
    caching_a = CachingCompletions(inner_a, cache=shared_cache)
    caching_b = CachingCompletions(inner_b, cache=shared_cache)

    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "рейс в Сочи"}],
        "tools": [],
    }

    caching_a.create(**kwargs)
    caching_b.create(**kwargs)  # разные "клиенты" (например, разные сессии)

    assert len(inner_a.calls) == 1
    assert len(inner_b.calls) == 0  # второй вызов взял ответ первого из общего кэша
    assert caching_b.hits == 1


def test_cached_tokens_is_zero_when_provider_did_not_report_the_field():
    # Кэш провайдера не гарантируется: поля может не быть вовсе.
    # Нет поля и ноль кэшированных токенов для нас одно и то же.
    class _UsageWithoutDetails:
        prompt_tokens = 100

    assert cached_tokens(_UsageWithoutDetails()) == 0
    assert cached_tokens({"prompt_tokens": 100}) == 0


def test_cached_tokens_reads_nested_field_from_dict_and_object():
    assert cached_tokens({"prompt_tokens_details": {"cached_tokens": 200}}) == 200

    class _Details:
        cached_tokens = 300

    class _Usage:
        prompt_tokens_details = _Details()

    assert cached_tokens(_Usage()) == 300


def test_cache_savings_subtracts_cached_tokens_from_prompt_tokens():
    # Главная ловушка расчётов: prompt_tokens уже включает кэшированные
    # токены, поэтому по полной цене тарифицируется разность.
    usage = {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 800}}

    assert cache_savings_from_usage(usage) == {
        "prompt_tokens": 1000,
        "cached_tokens": 800,
        "billable_input_tokens": 200,
    }


def test_cache_savings_without_cache_reports_everything_as_billable():
    usage = {"prompt_tokens": 500}

    assert cache_savings_from_usage(usage) == {
        "prompt_tokens": 500,
        "cached_tokens": 0,
        "billable_input_tokens": 500,
    }
