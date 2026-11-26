from __future__ import annotations

import json
from datetime import date

import pytest

from alisa.parser import as_client, build_system_prompt, parse_query

FIXED_TODAY = date(2026, 7, 21)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, payload: dict, finish_reason: str = "stop"):
        self.choices = [_FakeChoice(json.dumps(payload), finish_reason)]


class _FakeCompletions:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_call_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response


def _fake(response: _FakeResponse):
    completions = _FakeCompletions(response)
    return as_client(completions), completions


def _sample_payload(**overrides) -> dict:
    base = {
        "from": "MOW",
        "to": "AER",
        "dates": ["2026-07-23"],
        "passengers": 2,
        "max_price": None,
        "max_layover_hours": 3,
        "clarification": None,
    }
    base.update(overrides)
    return base


def test_parse_query_returns_dict_from_model_response():
    payload = _sample_payload()
    client, _ = _fake(_FakeResponse(payload))

    result = parse_query("хочу рейс из Москвы в Сочи", client, today=FIXED_TODAY)

    assert result == payload


def test_parse_query_sends_json_schema_response_format():
    client, completions = _fake(_FakeResponse(_sample_payload()))

    parse_query("рейс в Сочи", client, today=FIXED_TODAY)

    sent = completions.last_call_kwargs
    assert sent["response_format"]["type"] == "json_schema"
    assert "required" in sent["response_format"]["json_schema"]["schema"]


def test_parse_query_puts_system_prompt_in_first_message():
    # В формате chat/completions системный промпт - не отдельный
    # параметр, а первое сообщение с ролью "system".
    client, completions = _fake(_FakeResponse(_sample_payload()))

    parse_query("рейс в Сочи послезавтра", client, today=FIXED_TODAY)

    messages = completions.last_call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "2026-07-21" in messages[0]["content"]
    assert messages[1]["role"] == "user"


def test_parse_query_raises_on_content_filter():
    client, _ = _fake(_FakeResponse(_sample_payload(), finish_reason="content_filter"))

    with pytest.raises(ValueError):
        parse_query("рейс в Сочи", client, today=FIXED_TODAY)


def test_parse_query_raises_on_length_cutoff():
    client, _ = _fake(_FakeResponse(_sample_payload(), finish_reason="length"))

    with pytest.raises(ValueError):
        parse_query("рейс в Сочи", client, today=FIXED_TODAY)


def test_build_system_prompt_contains_todays_date():
    prompt = build_system_prompt(FIXED_TODAY)

    assert "2026-07-21" in prompt
    assert "не додумывай" in prompt
