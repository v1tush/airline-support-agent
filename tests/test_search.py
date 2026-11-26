from alisa.search import filter_flights

FLIGHTS = [
    {
        "id": "A",
        "from": "MOW",
        "to": "AER",
        "date": "2026-07-23",
        "price": 8000,
        "layover_hours": 0.0,
    },
    {
        "id": "B",
        "from": "MOW",
        "to": "AER",
        "date": "2026-07-23",
        "price": 15000,
        "layover_hours": 4.0,
    },
    {
        "id": "C",
        "from": "MOW",
        "to": "AER",
        "date": "2026-07-24",
        "price": 7000,
        "layover_hours": 1.0,
    },
    {
        "id": "D",
        "from": "MOW",
        "to": "LED",
        "date": "2026-07-23",
        "price": 3000,
        "layover_hours": 0.0,
    },
]

BASE_QUERY = {
    "from": "MOW",
    "to": "AER",
    "dates": ["2026-07-23"],
    "passengers": 1,
    "max_price": None,
    "max_layover_hours": None,
    "clarification": None,
}


def test_filters_by_route():
    result = filter_flights(BASE_QUERY, FLIGHTS)
    ids = {f["id"] for f in result}
    assert "D" not in ids  # другой город назначения


def test_filters_by_date():
    result = filter_flights(BASE_QUERY, FLIGHTS)
    ids = {f["id"] for f in result}
    assert "C" not in ids  # другая дата
    assert ids == {"A", "B"}


def test_filters_by_max_price():
    query = {**BASE_QUERY, "max_price": 10000}
    result = filter_flights(query, FLIGHTS)
    ids = {f["id"] for f in result}
    assert ids == {"A"}  # B дороже 10000


def test_filters_by_max_layover_hours():
    query = {**BASE_QUERY, "max_layover_hours": 2.0}
    result = filter_flights(query, FLIGHTS)
    ids = {f["id"] for f in result}
    assert ids == {"A"}  # B - пересадка 4 часа, не проходит


def test_no_date_filter_when_dates_empty():
    query = {**BASE_QUERY, "dates": []}
    result = filter_flights(query, FLIGHTS)
    ids = {f["id"] for f in result}
    assert ids == {"A", "B", "C"}  # дата не ограничивает - только маршрут


def test_returns_empty_list_when_nothing_matches():
    query = {**BASE_QUERY, "to": "KZN"}
    assert filter_flights(query, FLIGHTS) == []
