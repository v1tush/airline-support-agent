from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any, Protocol

from alisa.config import MODEL
from alisa.schemas import FLIGHT_QUERY_SCHEMA, json_schema_format


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class ChatClient(Protocol):
    chat: _Chat


def as_client(completions: Any) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def read_field(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        value = source.get(name, default)
    else:
        value = getattr(source, name, default)
    return default if value is None else value


def build_system_prompt(today):
    return (
        "Ты - Алиса, учебный ассистент по авиаперелётам. Правило: не додумывай факты. "
        "Города: MOW Москва, AER Сочи, LED Санкт-Петербург, KZN Казань, JFK Нью-Йорк. "
        "Неизвестный или отсутствующий город - null. Если не названы даты - []. "
        "Неоднозначные праздники требуют уточнения года и дат: dates=[], clarification с вопросом. "
        "Если пассажиры не названы - 1. max_price - общий бюджет группы, "
        "бюджет на человека умножай на passengers. Дёшево не задаёт сумму: max_price=null. "
        "Без пересадок означает max_layover_hours=0, дробные часы допустимы. "
        "При нехватке городов или неоднозначности заполни clarification, иначе null. "
        f"Сегодня: {today.isoformat()}. Завтра: {(today + timedelta(days=1)).isoformat()}; "
        f"послезавтра: {(today + timedelta(days=2)).isoformat()}. "
        "Если написано завтра или послезавтра, dates должен содержать соответствующую дату. "
        "Если from или to равен null, clarification обязательно содержит вопрос о недостающем городе. "
        'Пример: В Сочи -> from=null, to=AER, dates=[], clarification="Откуда вы летите?". '
        "Верни только JSON по схеме."
    )


def parse_query(
    user_text: str,
    client: ChatClient,
    *,
    today: date | None = None,
    model: str = MODEL,
) -> dict:
    today = today or date.today()
    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": build_system_prompt(today)},
            {"role": "user", "content": user_text},
        ],
        response_format=json_schema_format("flight_query", FLIGHT_QUERY_SCHEMA),
    )
    from alisa.schemas import validate_query

    message = checked_message(response)
    payload = json.loads(read_field(message, "content"))
    if not isinstance(payload, dict) or set(payload) != set(
        FLIGHT_QUERY_SCHEMA["required"]
    ):
        raise ValueError("Ответ должен содержать все семь полей схемы")
    return validate_query(payload)


def checked_message(response, *, allow_tools=False):
    choices = read_field(response, "choices")
    if not choices:
        raise ValueError("API не вернул choices")
    choice = choices[0]
    message = read_field(choice, "message")
    reason = read_field(choice, "finish_reason")
    if reason not in ({"stop", "tool_calls"} if allow_tools else {"stop"}):
        raise ValueError(f"Незавершённый ответ: {reason}")
    if read_field(message, "refusal"):
        raise ValueError("Модель отказалась отвечать")
    calls = read_field(message, "tool_calls") or []
    if calls:
        if not allow_tools or reason != "tool_calls":
            raise ValueError("Несогласованный ответ с tool_calls")
    elif not read_field(message, "content") or reason != "stop":
        raise ValueError("В ответе нет завершённого текста")
    return message
