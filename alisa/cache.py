import copy
import json

from alisa.parser import checked_message, read_field


def _cache_key(kwargs):
    def encode(x):
        if hasattr(x, "model_dump"):
            return x.model_dump(exclude_none=True)
        return vars(x)

    return json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=encode)


class CachingCompletions:
    def __init__(self, inner_completions, cache=None):
        self._inner = inner_completions
        self._cache = cache if cache is not None else {}
        self.hits = self.misses = 0

    def create(self, **kwargs):
        key = _cache_key(kwargs)
        if key in self._cache:
            self.hits += 1
            return copy.deepcopy(self._cache[key])
        self.misses += 1
        response = self._inner.create(**kwargs)
        checked_message(response, allow_tools=True)
        if len(self._cache) >= 32:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = copy.deepcopy(response)
        return response


def cached_tokens(usage):
    return read_field(read_field(usage, "prompt_tokens_details"), "cached_tokens", 0)


def cache_savings_from_usage(usage):
    prompt = read_field(usage, "prompt_tokens")
    if prompt is None:
        return {
            "prompt_tokens": None,
            "cached_tokens": None,
            "billable_input_tokens": None,
        }
    cached = cached_tokens(usage)
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "billable_input_tokens": max(prompt - cached, 0),
    }
