from __future__ import annotations

DEFAULT_WEIGHTS = {"price": 0.6, "layover_hours": 0.3, "duration_hours": 0.1}


def _normalize(value: float, values: list[float]) -> float:
    lo, hi = min(values), max(values)
    if hi == lo:
        return 0.0
    return (value - lo) / (hi - lo)


def score_flights(flights, weights=None):
    import math

    weights = DEFAULT_WEIGHTS if weights is None else weights
    if (
        set(weights) != set(DEFAULT_WEIGHTS)
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in weights.values()
        )
        or sum(weights.values()) <= 0
    ):
        raise ValueError(
            "Нужны неотрицательные веса трёх критериев с положительной суммой"
        )
    if not flights:
        return []
    bounds = {
        k: (min(f[k] for f in flights), max(f[k] for f in flights)) for k in weights
    }
    scored = []
    for flight in flights:
        score = 0
        for key, weight in weights.items():
            lo, hi = bounds[key]
            score += weight * (0 if lo == hi else (flight[key] - lo) / (hi - lo))
        scored.append((flight, score))
    return scored


def rank_top_n(
    flights: list[dict], n: int = 3, weights: dict | None = None
) -> list[dict]:
    scored = score_flights(flights, weights)
    scored.sort(key=lambda pair: (pair[1], pair[0]["id"]))
    return [flight for flight, _ in scored[:n]]
