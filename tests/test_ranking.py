from alisa.ranking import rank_top_n, score_flights

FLIGHTS = [
    {"id": "cheap_direct", "price": 6000, "layover_hours": 0.0, "duration_hours": 2.5},
    {
        "id": "expensive_direct",
        "price": 20000,
        "layover_hours": 0.0,
        "duration_hours": 2.3,
    },
    {
        "id": "cheap_long_layover",
        "price": 5000,
        "layover_hours": 5.0,
        "duration_hours": 8.0,
    },
    {"id": "mid", "price": 9000, "layover_hours": 1.0, "duration_hours": 3.5},
]


def test_score_flights_returns_one_pair_per_flight():
    scored = score_flights(FLIGHTS)
    assert len(scored) == len(FLIGHTS)
    for flight, score in scored:
        assert isinstance(score, float)


def test_rank_top_n_respects_n():
    top2 = rank_top_n(FLIGHTS, n=2)
    assert len(top2) == 2


def test_rank_top_n_prefers_cheap_direct_over_expensive_direct():
    ranked = rank_top_n(FLIGHTS, n=4)
    order = [f["id"] for f in ranked]
    assert order.index("cheap_direct") < order.index("expensive_direct")


def test_rank_top_n_penalizes_long_layover_despite_low_price():
    # Низкая цена не должна перевешивать долгую пересадку в этом наборе.
    ranked = rank_top_n(FLIGHTS, n=4)
    order = [f["id"] for f in ranked]
    assert order.index("cheap_direct") < order.index("cheap_long_layover")


def test_rank_top_n_empty_list_returns_empty():
    assert rank_top_n([], n=3) == []


def test_custom_weights_change_order():
    price_only_weights = {"price": 0.0, "layover_hours": 1.0, "duration_hours": 0.0}
    ranked = rank_top_n(FLIGHTS, n=4, weights=price_only_weights)
    assert ranked[0]["id"] in ("cheap_direct", "expensive_direct")
    assert ranked[-1]["id"] == "cheap_long_layover"
