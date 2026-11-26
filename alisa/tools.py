from __future__ import annotations

from alisa.rag import search_rules
from alisa.ranking import rank_top_n
from alisa.search import filter_flights

SEARCH_FLIGHTS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Ищет рейсы по датасету Алисы и возвращает "
        "отранжированный топ-3 варианта (жёсткая фильтрация и "
        "ранжирование выполняются в коде, не моделью). Вызывай, "
        "когда у пользователя известны хотя бы город вылета и "
        "город назначения. Если пользователь не назвал даты, "
        "бюджет или максимальную пересадку - вызывай инструмент "
        "без этих полей, а не придумывай значения.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_city": {
                    "type": "string",
                    "description": "IATA-код города вылета, например MOW",
                },
                "to_city": {
                    "type": "string",
                    "description": "IATA-код города назначения, например AER",
                },
                "dates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Даты в формате "
                    "YYYY-MM-DD, если "
                    "пользователь их "
                    "назвал",
                },
                "max_price": {
                    "type": "integer",
                    "description": "Общий бюджет группы в рублях",
                },
                "max_layover_hours": {
                    "type": "number",
                    "description": "Максимальная "
                    "длительность "
                    "пересадки "
                    "в "
                    "часах, "
                    "если "
                    "названа",
                },
                "passengers": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "Число пассажиров, по умолчанию 1",
                },
            },
            "required": ["from_city", "to_city"],
            "additionalProperties": False,
        },
    },
}


def execute_search_flights(tool_input: dict, flights: list[dict]) -> dict:
    if not isinstance(tool_input, dict) or tool_input.keys() - {
        "from_city",
        "to_city",
        "dates",
        "passengers",
        "max_price",
        "max_layover_hours",
    }:
        raise ValueError("Неверные аргументы поиска")
    query = {
        "from": tool_input["from_city"],
        "to": tool_input["to_city"],
        "dates": tool_input.get("dates", []),
        "passengers": tool_input.get("passengers", 1),
        "max_price": tool_input.get("max_price"),
        "max_layover_hours": tool_input.get("max_layover_hours"),
    }
    candidates = filter_flights(query, flights)
    top3 = rank_top_n(candidates, n=3)
    return {"candidates_count": len(candidates), "top3": top3}


SEARCH_AIRLINE_RULES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_airline_rules",
        "description": "Ищет релевантный фрагмент правил тарифов и багажа "
        "авиакомпаний (RAG: embeddings + косинусное сходство), а "
        "не по точному совпадению слов. Вызывай, когда "
        "пользователь спрашивает про ручную кладь, багаж, провоз "
        "спортивного инвентаря, возврат или обмен билета - не "
        "выдумывай эти детали сам, они всегда специфичны для "
        "конкретной авиакомпании. Вызывай по вопросу "
        "пользователя. Если пользователь спросил только о рейсах, "
        "дополнительный поиск правил не нужен.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "О чём спрашивает "
                    "пользователь, "
                    "например 'провоз "
                    "лыж'",
                },
                "airline": {
                    "type": "string",
                    "description": "Авиакомпания, если известна, например 'Победа'",
                },
            },
            "required": ["query"],
        },
    },
}


def execute_search_airline_rules(
    tool_input: dict, chunks: list, embedder, top_k: int = 2, guard_client=False
) -> dict:
    if not isinstance(tool_input, dict) or tool_input.keys() - {"query", "airline"}:
        raise ValueError("Неверные аргументы поиска правил")
    query = tool_input["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Нужен непустой вопрос")
    if tool_input.get("airline") is not None and not isinstance(
        tool_input["airline"], str
    ):
        raise ValueError("Неверное название авиакомпании")
    airline = tool_input.get("airline")
    pool = chunks
    if airline:
        narrowed = [c for c in chunks if c.airline.lower() == airline.lower()]
        pool = narrowed
    top = search_rules(query, pool, embedder, top_k=top_k)
    from alisa.guardrails import contains_suspicious_instructions, screen_tool_output

    suspicious = [c.doc_id for c in top if contains_suspicious_instructions(c.text)]
    if suspicious:
        raise ValueError(
            "Заблокирован подозрительный источник: " + ", ".join(suspicious)
        )
    if guard_client and top:
        blocked = []
        try:
            for chunk in top:
                verdict = screen_tool_output(chunk.text, guard_client)
                if verdict["verdict"] != "safe":
                    blocked.append(chunk.doc_id)
        except Exception as exc:
            raise ValueError(
                "Проверка источника недоступна; результат не выдан"
            ) from exc
        if blocked:
            raise ValueError("Классификатор заблокировал: " + ", ".join(blocked))
    return {
        "rules": [
            {"source": c.doc_id, "trusted": False, "airline": c.airline, "text": c.text}
            for c in top
        ]
    }
