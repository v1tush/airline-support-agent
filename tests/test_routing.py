from __future__ import annotations

import pytest

from alisa.config import FAST_MODEL, MODEL, STRONG_MODEL
from alisa.routing import (
    FallbackCompletions,
    RoutingCompletions,
    choose_model,
    is_retryable_error,
)


class _FakeStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _FakeResponse:
    def __init__(self, model_used: str):
        self.model_used = model_used


class _FakeInnerCompletions:
    def __init__(self, behavior):
        self._behavior = list(behavior)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._behavior.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(model_used=kwargs.get("model"))


def test_choose_model_short_simple_query_goes_to_fast_model():
    assert choose_model("Рейс в Сочи") == FAST_MODEL


def test_choose_model_long_query_goes_to_strong_model():
    long_query = "Найди, пожалуйста, самый дешёвый рейс из Москвы в Сочи с минимальной пересадкой"
    assert choose_model(long_query) == STRONG_MODEL


def test_choose_model_ambiguous_date_marker_goes_to_strong_model():
    assert choose_model("Слетать на майские") == STRONG_MODEL


def test_is_retryable_error_true_for_429_500_503_529():
    assert is_retryable_error(_FakeStatusError(429)) is True
    assert is_retryable_error(_FakeStatusError(500)) is True
    assert is_retryable_error(_FakeStatusError(503)) is True
    assert is_retryable_error(_FakeStatusError(529)) is True


def test_is_retryable_error_false_for_400_and_plain_exception():
    assert is_retryable_error(_FakeStatusError(400)) is False
    assert is_retryable_error(ValueError("что-то не так")) is False


def test_routing_assigns_model_when_not_explicitly_set():
    inner = _FakeInnerCompletions(["ignored"])
    routing = RoutingCompletions(inner)

    response = routing.create(
        model=None, messages=[{"role": "user", "content": "Рейс в Сочи"}]
    )

    assert response.model_used == FAST_MODEL


def test_routing_respects_explicit_model():
    inner = _FakeInnerCompletions(["ignored"])
    routing = RoutingCompletions(inner)

    response = routing.create(
        model=MODEL, messages=[{"role": "user", "content": "Рейс в Сочи"}]
    )

    assert response.model_used == MODEL


def test_routing_ignores_tool_messages_when_looking_for_the_user_question():
    # Маршрут выбираем по сообщению пользователя, а не по результату инструмента.
    inner = _FakeInnerCompletions(["ignored"])
    routing = RoutingCompletions(inner)

    response = routing.create(
        model=None,
        messages=[
            {"role": "user", "content": "Рейс в Сочи"},
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": "очень длинный результат инструмента про майские праздники и пересадки",
            },
        ],
    )

    assert response.model_used == FAST_MODEL


def test_fallback_retries_on_retryable_error():
    inner = _FakeInnerCompletions([_FakeStatusError(529), "recovered"])
    fallback = FallbackCompletions(inner, fallback_model=FAST_MODEL)

    response = fallback.create(model=STRONG_MODEL, messages=[])

    assert response.model_used == FAST_MODEL
    assert fallback.fallback_used_count == 1
    assert len(inner.calls) == 2


def test_fallback_reraises_non_retryable_error():
    inner = _FakeInnerCompletions([_FakeStatusError(400)])
    fallback = FallbackCompletions(inner, fallback_model=FAST_MODEL)

    with pytest.raises(_FakeStatusError):
        fallback.create(model=STRONG_MODEL, messages=[])

    assert len(inner.calls) == 1  # без повторной попытки на невосстановимой ошибке
