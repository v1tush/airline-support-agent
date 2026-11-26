from __future__ import annotations

import logging

from alisa.explain import explain_top_flights
from alisa.flights_data import FLIGHTS
from alisa.parser import ChatClient, parse_query
from alisa.ranking import rank_top_n
from alisa.search import filter_flights

logger = logging.getLogger("alisa.pipeline")


def run_pipeline(
    user_text: str,
    client: ChatClient,
    flights: list[dict] | None = None,
    *,
    today=None,
) -> dict:
    flights = FLIGHTS if flights is None else flights

    query = parse_query(user_text, client, today=today)
    if query.get("clarification") or not query.get("from") or not query.get("to"):
        return {
            "query": query,
            "candidates_count": 0,
            "top3": [],
            "explanation": query.get("clarification") or "Уточните маршрут.",
        }
    logger.info("parsed query: %s", query)

    candidates = filter_flights(query, flights)
    logger.info("filtered candidates: %d of %d", len(candidates), len(flights))

    top3 = rank_top_n(candidates, n=3)
    logger.info("top-3 ids: %s", [f["id"] for f in top3])

    explanation = explain_top_flights(query, top3, client)

    return {
        "query": query,
        "candidates_count": len(candidates),
        "top3": top3,
        "explanation": explanation,
    }
