from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from alisa.agent import _default_rule_chunks
from alisa.config import build_client
from alisa.flights_data import FLIGHTS
from alisa.rag import YandexEmbedder, build_word_overlap_embedder
from alisa.tools import execute_search_airline_rules, execute_search_flights

mcp = FastMCP("alisa")

# ALISA_SEMANTIC=1 включает embeddings; по умолчанию поиск локальный.
_rule_chunks = _default_rule_chunks()


_rule_embedder = (
    YandexEmbedder(build_client())
    if os.getenv("ALISA_SEMANTIC") == "1"
    else build_word_overlap_embedder(_rule_chunks)
)


@mcp.tool()
def search_flights(
    from_city: str,
    to_city: str,
    dates: list[str] | None = None,
    passengers: int = 1,
    max_price: int | None = None,
    max_layover_hours: float | None = None,
) -> dict:
    """Возвращает три лучших рейса из учебного набора.

    Args:
        from_city: IATA-код города вылета, например MOW.
        to_city: IATA-код города назначения, например AER.
        dates: Даты в формате YYYY-MM-DD, если пользователь их назвал.
        passengers: Число пассажиров, по умолчанию 1.
        max_price: Общий бюджет группы в рублях, если назван.
        max_layover_hours: Максимальная длительность пересадки в часах, если названа.
    """
    tool_input = {
        "from_city": from_city,
        "to_city": to_city,
        "dates": dates or [],
        "passengers": passengers,
        "max_price": max_price,
        "max_layover_hours": max_layover_hours,
    }
    return execute_search_flights(tool_input, FLIGHTS)


@mcp.tool()
def search_airline_rules(query: str, airline: str | None = None) -> dict:
    """Ищет фрагменты правил тарифов и багажа.
    По умолчанию поиск лексический; ALISA_SEMANTIC=1 включает embeddings v2.

    Args:
        query: О чём спрашивает пользователь, например 'провоз лыж'.
        airline: Авиакомпания, если известна, например 'Победа'.
    """
    tool_input = {"query": query, "airline": airline}

    guard = build_client() if os.getenv("ALISA_LLM_GUARD") == "1" else False
    return execute_search_airline_rules(
        tool_input, _rule_chunks, _rule_embedder, guard_client=guard
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
