import json

from alisa.parser import as_client
from alisa.pipeline import run_pipeline

FLIGHTS = [
    {
        "id": "A",
        "from": "MOW",
        "to": "AER",
        "date": "2026-07-23",
        "price": 8000,
        "layover_hours": 0.0,
        "duration_hours": 2.5,
        "airline": "Аэрофлот",
    },
    {
        "id": "B",
        "from": "MOW",
        "to": "AER",
        "date": "2026-07-23",
        "price": 15000,
        "layover_hours": 4.0,
        "duration_hours": 6.0,
        "airline": "S7",
    },
    {
        "id": "C",
        "from": "MOW",
        "to": "LED",
        "date": "2026-07-23",
        "price": 3000,
        "layover_hours": 0.0,
        "duration_hours": 1.3,
        "airline": "Победа",
    },
]

PARSED_QUERY = {
    "from": "MOW",
    "to": "AER",
    "dates": ["2026-07-23"],
    "passengers": 1,
    "max_price": None,
    "max_layover_hours": None,
    "clarification": None,
}


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, text, finish_reason="stop"):
        self.choices = [_FakeChoice(text, finish_reason)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_run_pipeline_connects_parse_filter_rank_explain():
    completions = _FakeCompletions(
        [
            _FakeResponse(json.dumps(PARSED_QUERY)),
            _FakeResponse("Вариант A - самый быстрый и дешёвый из подходящих."),
        ]
    )
    client = as_client(completions)

    result = run_pipeline("рейс из Москвы в Сочи", client, flights=FLIGHTS)

    assert result["candidates_count"] == 2  # A и B (C - другой город)
    assert [f["id"] for f in result["top3"]] == ["A", "B"]  # A дешевле и без пересадки
    assert "самый быстрый" in result["explanation"]
    assert len(completions.calls) == 2
