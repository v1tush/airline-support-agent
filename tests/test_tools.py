from pathlib import Path

from alisa.flights_data import FLIGHTS
from alisa.rag import build_word_overlap_embedder, load_rule_documents
from alisa.tools import (
    SEARCH_AIRLINE_RULES_TOOL,
    SEARCH_FLIGHTS_TOOL,
    execute_search_airline_rules,
    execute_search_flights,
)

RULES_DIR = Path(__file__).parent.parent / "alisa" / "rules"


def test_search_flights_tool_requires_only_route():
    assert SEARCH_FLIGHTS_TOOL["function"]["parameters"]["required"] == [
        "from_city",
        "to_city",
    ]


def test_search_flights_tool_has_openai_function_shape():
    # Форма описания инструмента задана API: внешний объект с
    # type="function" и вложенный "function" с name/description/parameters.
    assert SEARCH_FLIGHTS_TOOL["type"] == "function"
    assert SEARCH_FLIGHTS_TOOL["function"]["name"] == "search_flights"
    assert len(SEARCH_FLIGHTS_TOOL["function"]["description"]) > 20
    assert "properties" in SEARCH_FLIGHTS_TOOL["function"]["parameters"]


def test_execute_search_flights_with_full_arguments():
    tool_input = {
        "from_city": "MOW",
        "to_city": "AER",
        "dates": ["2026-07-23"],
        "max_price": 10000,
        "max_layover_hours": 3,
    }
    result = execute_search_flights(tool_input, FLIGHTS)
    assert result["candidates_count"] > 0
    assert len(result["top3"]) <= 3
    assert all(f["price"] <= 10000 for f in result["top3"])


def test_execute_search_flights_defaults_missing_optional_fields():
    # Модель назвала только маршрут - dates/max_price/max_layover_hours
    # не пришли вовсе, а не пришли как null или 0.
    tool_input = {"from_city": "MOW", "to_city": "LED"}
    result = execute_search_flights(tool_input, FLIGHTS)
    assert result["candidates_count"] == 3  # F009, F010, F011


def test_execute_search_flights_raises_keyerror_without_required_route():
    # Пропущенный обязательный аргумент должен вызвать KeyError.
    import pytest

    with pytest.raises(KeyError):
        execute_search_flights({"to_city": "AER"}, FLIGHTS)


def test_search_airline_rules_tool_requires_only_query():
    assert SEARCH_AIRLINE_RULES_TOOL["type"] == "function"
    assert SEARCH_AIRLINE_RULES_TOOL["function"]["parameters"]["required"] == ["query"]


def test_execute_search_airline_rules_without_airline_filter():
    chunks = load_rule_documents(RULES_DIR)
    embedder = build_word_overlap_embedder(chunks)

    result = execute_search_airline_rules(
        {"query": "провоз лыж"}, chunks, embedder, top_k=1
    )

    assert len(result["rules"]) == 1
    assert (
        "лыж" in result["rules"][0]["text"].lower()
        or "спортивного инвентаря" in result["rules"][0]["text"].lower()
    )


def test_execute_search_airline_rules_narrows_to_named_airline():
    chunks = load_rule_documents(RULES_DIR)
    embedder = build_word_overlap_embedder(chunks)

    result = execute_search_airline_rules(
        {"query": "возврат билета", "airline": "S7"}, chunks, embedder, top_k=1
    )

    assert result["rules"][0]["airline"] == "S7"


def test_execute_search_airline_rules_returns_empty_for_unknown_airline():
    # Правила другой компании не отвечают на вопрос об указанной.
    chunks = load_rule_documents(RULES_DIR)
    embedder = build_word_overlap_embedder(chunks)

    result = execute_search_airline_rules(
        {"query": "провоз багажа", "airline": "Несуществующая авиакомпания"},
        chunks,
        embedder,
        top_k=1,
    )

    assert result["rules"] == []
