from __future__ import annotations

import json

from alisa.eval_harness import check_tool_call, judge_answer, run_golden_set
from alisa.golden_set import GoldenCase, load_golden_set
from alisa.parser import as_client


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = json.dumps(arguments, ensure_ascii=False)


class _FakeToolCall:
    type = "function"

    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    role = "assistant"

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, choices):
        self.choices = choices


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _text_response(text):
    return _FakeResponse([_FakeChoice(_FakeMessage(content=text), "stop")])


def _tool_response(*tool_calls):
    return _FakeResponse(
        [_FakeChoice(_FakeMessage(tool_calls=list(tool_calls)), "tool_calls")]
    )


def _judge_response(verdict, reasoning):
    return _text_response(json.dumps({"verdict": verdict, "reasoning": reasoning}))


def _fake(*responses):
    completions = _FakeCompletions(responses)
    return as_client(completions), completions


def _assistant_message(*tool_calls):
    return _FakeMessage(tool_calls=list(tool_calls) or None)


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
]


def test_load_golden_set_reads_all_cases_from_repo_file():
    cases = load_golden_set()
    assert len(cases) >= 20
    assert all(isinstance(c, GoldenCase) for c in cases)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "id golden-кейсов должны быть уникальны"


def test_check_tool_call_passes_when_expected_tool_called_with_matching_args():
    messages = [
        _assistant_message(
            _FakeToolCall(
                "t1", "search_flights", {"from_city": "MOW", "to_city": "AER"}
            )
        )
    ]
    passed, detail = check_tool_call(messages, "search_flights", {"from_city": "MOW"})
    assert passed is True


def test_check_tool_call_fails_when_wrong_tool_called():
    messages = [
        _assistant_message(
            _FakeToolCall("t1", "search_airline_rules", {"query": "багаж"})
        )
    ]
    passed, detail = check_tool_call(messages, "search_flights", {})
    assert passed is False
    assert "search_flights" in detail


def test_check_tool_call_fails_when_argument_mismatched():
    messages = [
        _assistant_message(
            _FakeToolCall(
                "t1", "search_flights", {"from_city": "MOW", "to_city": "LED"}
            )
        )
    ]
    passed, detail = check_tool_call(messages, "search_flights", {"to_city": "AER"})
    assert passed is False
    assert "to_city" in detail


def test_check_tool_call_expects_no_tool_and_passes_when_none_called():
    messages = [_FakeMessage(content="Всегда рада помочь!")]
    passed, detail = check_tool_call(messages, None, {})
    assert passed is True


def test_check_tool_call_expects_no_tool_but_fails_when_tool_was_called():
    messages = [
        _assistant_message(
            _FakeToolCall(
                "t1", "search_flights", {"from_city": "MOW", "to_city": "AER"}
            )
        )
    ]
    passed, detail = check_tool_call(messages, None, {})
    assert passed is False
    assert "search_flights" in detail


def test_check_tool_call_ignores_tool_role_messages_in_history():
    # В истории теперь есть отдельные сообщения с ролью "tool" -
    # грейдер должен считать вызовы у assistant, а не результаты.
    messages = [
        _assistant_message(_FakeToolCall("t1", "search_flights", {"from_city": "MOW"})),
        {"role": "tool", "tool_call_id": "t1", "content": "{}"},
    ]
    passed, _ = check_tool_call(messages, "search_flights", {"from_city": "MOW"})
    assert passed is True


def test_judge_answer_parses_structured_verdict():
    judge_client, judge_completions = _fake(
        _judge_response(
            "correct", "Ответ упоминает конкретную цену из результата инструмента."
        ),
    )

    verdict = judge_answer(
        "Найди рейсы Москва-Сочи дешевле 10000",
        "Ответ должен упоминать только варианты дешевле 10000 рублей",
        "Нашла рейс Аэрофлотом за 8000 рублей.",
        judge_client,
    )

    assert verdict["verdict"] == "correct"
    assert verdict["reasoning"]
    # Запрос к судье обязан содержать structured output схему -
    # иначе результат не гарантированно распарсится как JSON.
    assert judge_completions.calls[0]["response_format"]["type"] == "json_schema"


def test_run_golden_set_marks_case_failed_if_judge_says_incorrect_even_when_tool_call_correct():
    cases = [
        GoldenCase(
            id="gs-test-1",
            query="рейс из Москвы в Сочи",
            expected_tool="search_flights",
            expected_args_contains={"from_city": "MOW", "to_city": "AER"},
            rubric="ответ должен упоминать конкретную цену",
        ),
    ]
    client, _ = _fake(
        _tool_response(
            _FakeToolCall(
                "t1", "search_flights", {"from_city": "MOW", "to_city": "AER"}
            )
        ),
        _text_response("Нашла несколько вариантов."),
    )
    judge_client, _ = _fake(
        _judge_response(
            "incorrect",
            "Ответ не называет конкретную цену, хотя рубрика этого требует.",
        ),
    )

    report = run_golden_set(cases, client, judge_client=judge_client, flights=FLIGHTS)

    assert len(report.results) == 1
    result = report.results[0]
    assert result.tool_check_passed is True
    assert result.judge_verdict == "incorrect"
    assert result.passed is False
    assert report.pass_rate == 0.0
    assert report.failures() == [result]


def test_run_golden_set_marks_case_passed_when_both_graders_agree():
    cases = [
        GoldenCase(
            id="gs-test-2",
            query="рейс из Москвы в Сочи",
            expected_tool="search_flights",
            expected_args_contains={"from_city": "MOW", "to_city": "AER"},
            rubric="ответ должен упоминать конкретную цену",
        ),
    ]
    client, _ = _fake(
        _tool_response(
            _FakeToolCall(
                "t1", "search_flights", {"from_city": "MOW", "to_city": "AER"}
            )
        ),
        _text_response("Нашла рейс Аэрофлотом за 8000 рублей."),
    )
    judge_client, _ = _fake(
        _judge_response("correct", "Ответ называет конкретную цену - 8000 рублей."),
    )

    report = run_golden_set(cases, client, judge_client=judge_client, flights=FLIGHTS)

    assert report.pass_rate == 1.0
    assert report.failures() == []


def test_run_golden_set_defaults_judge_client_to_main_client_when_not_given():
    cases = [
        GoldenCase(
            id="gs-test-3",
            query="спасибо",
            expected_tool=None,
            expected_args_contains={},
            rubric="короткая уместная реплика без выдуманных данных",
        ),
    ]
    # Один и тот же фейковый клиент обслуживает и агента, и судью -
    # поэтому очередь ответов общая: сначала ответ агента, затем ответ судьи.
    client, _ = _fake(
        _text_response("Всегда рада помочь!"),
        _judge_response("correct", "Короткая уместная реплика, ничего не выдумано."),
    )

    report = run_golden_set(cases, client, flights=FLIGHTS)

    assert report.pass_rate == 1.0
