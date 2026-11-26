from types import SimpleNamespace as NS

import pytest

from alisa.cache import CachingCompletions


class Inner:
    def __init__(self):
        self.calls = 0
        self.finish = "stop"

    def create(self, **kwargs):
        self.calls += 1
        return NS(
            choices=[
                NS(
                    finish_reason=self.finish,
                    message=NS(content="ok", tool_calls=[], refusal=None),
                )
            ]
        )


def test_modes_and_mutation_are_separate():
    inner = Inner()
    cache = CachingCompletions(inner)
    request = {"model": "model", "messages": [{"role": "user", "content": "hello"}]}
    first = cache.create(**request)
    first.choices[0].message.content = "changed"
    assert cache.create(**request).choices[0].message.content == "ok"
    cache.create(**request, reasoning_effort="none")
    cache.create(**request, max_tokens=20)
    cache.create(**request, response_format={"type": "json_object"})
    assert inner.calls == 4


def test_length_is_not_cached():
    inner = Inner()
    inner.finish = "length"
    cache = CachingCompletions(inner)
    for _ in range(2):
        with pytest.raises(ValueError):
            cache.create(model="model", messages=[])
    assert inner.calls == 2
