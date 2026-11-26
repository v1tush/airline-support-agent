from __future__ import annotations

from typing import Any

from alisa.config import FAST_MODEL, STRONG_MODEL

# Повторяем только временные ошибки.
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}

_COMPLEX_MARKERS = ("майск", "праздник", "послезавтра", "в районе", "почему")


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code in RETRYABLE_STATUS_CODES or type(exc).__name__ in (
        "APIConnectionError",
        "APITimeoutError",
    )


def choose_model(
    user_text: str, *, simple_model: str = FAST_MODEL, complex_model: str = STRONG_MODEL
) -> str:
    text_lower = user_text.lower()
    if len(user_text) > 60 or any(marker in text_lower for marker in _COMPLEX_MARKERS):
        return complex_model
    return simple_model


class RoutingCompletions:
    def __init__(
        self,
        inner_completions: Any,
        *,
        simple_model: str = FAST_MODEL,
        complex_model: str = STRONG_MODEL,
    ):
        self._inner = inner_completions
        self._simple_model = simple_model
        self._complex_model = complex_model

    def _last_user_text(self, messages: list) -> str:
        for msg in reversed(messages):
            role = (
                msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            )
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", None)
            )
            if role == "user" and isinstance(content, str):
                return content
        return ""

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("model") is None:
            user_text = self._last_user_text(kwargs.get("messages") or [])
            kwargs["model"] = choose_model(
                user_text,
                simple_model=self._simple_model,
                complex_model=self._complex_model,
            )
        return self._inner.create(**kwargs)


class FallbackCompletions:
    def __init__(self, inner_completions: Any, *, fallback_model: str = FAST_MODEL):
        self._inner = inner_completions
        self._fallback_model = fallback_model
        self.fallback_used_count = 0

    def create(self, **kwargs: Any) -> Any:
        try:
            return self._inner.create(**kwargs)
        except Exception as exc:
            if not is_retryable_error(exc):
                raise
            self.fallback_used_count += 1
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["model"] = self._fallback_model
            return self._inner.create(**fallback_kwargs)
