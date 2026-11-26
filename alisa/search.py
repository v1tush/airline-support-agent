from __future__ import annotations


def filter_flights(query: dict, flights: list[dict]) -> list[dict]:
    from alisa.schemas import validate_query

    validate_query(query)
    passengers = query.get("passengers", 1)
    if not query.get("from") or not query.get("to") or query.get("clarification"):
        raise ValueError("Нужно уточнить условия поиска")
    result = []
    for flight in flights:
        if flight["from"] != query["from"] or flight["to"] != query["to"]:
            continue
        if query.get("dates") and flight["date"] not in query["dates"]:
            continue
        if (
            query.get("max_price") is not None
            and flight["price"] * passengers > query["max_price"]
        ):
            continue
        if (
            query.get("max_layover_hours") is not None
            and flight["layover_hours"] > query["max_layover_hours"]
        ):
            continue
        if flight.get("seats", 4) < passengers:
            continue
        result.append({**flight, "total_price": flight["price"] * passengers})
    return result
