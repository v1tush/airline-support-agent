from alisa.explain import build_explain_prompt, explain_top_flights
from alisa.parser import as_client

QUERY = {
    "from": "MOW",
    "to": "AER",
    "dates": ["2026-07-23"],
    "passengers": 1,
    "max_price": None,
    "max_layover_hours": None,
    "clarification": None,
}

TOP3 = [
    {
        "id": "A",
        "airline": "Аэрофлот",
        "price": 8500,
        "layover_hours": 0.0,
        "duration_hours": 2.5,
    },
    {
        "id": "B",
        "airline": "Победа",
        "price": 6200,
        "layover_hours": 2.5,
        "duration_hours": 5.0,
    },
    {
        "id": "C",
        "airline": "S7",
        "price": 12000,
        "layover_hours": 0.0,
        "duration_hours": 2.3,
    },
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, text):
        self.choices = [_FakeChoice(text)]


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response


def _fake(response):
    completions = _FakeCompletions(response)
    return as_client(completions), completions


def test_build_explain_prompt_mentions_all_three_flights():
    prompt = build_explain_prompt(QUERY, TOP3)
    assert "Аэрофлот" in prompt
    assert "Победа" in prompt
    assert "S7" in prompt


def test_explain_top_flights_returns_model_text():
    client, completions = _fake(_FakeResponse("Вот три хороших варианта..."))

    result = explain_top_flights(QUERY, TOP3, client)

    assert result == "Вот три хороших варианта..."
    assert completions.last_call_kwargs is not None


def test_explain_top_flights_skips_api_call_when_no_flights():
    client, completions = _fake(_FakeResponse("не должно быть вызвано"))

    result = explain_top_flights(QUERY, [], client)

    assert "не нашлось" in result
    assert completions.last_call_kwargs is None
