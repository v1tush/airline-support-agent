import time
from dataclasses import dataclass, field

from alisa.cache import cached_tokens
from alisa.parser import read_field

PRICING = {"qwen3.6-35b-a3b": {"input": 0.2, "cached": 0.05, "output": 0.3}}
PRICING_DATE = "2026-11-24"


class BudgetExceededError(RuntimeError):
    pass


def model_name(model):
    return model.rstrip("/").removesuffix("/latest").removesuffix("/rc").split("/")[-1]


def estimate_cost(usage, model):
    rates = PRICING.get(model_name(model))
    prompt, output = (
        read_field(usage, "prompt_tokens"),
        read_field(usage, "completion_tokens"),
    )
    if rates is None or prompt is None or output is None:
        return None
    cached = cached_tokens(usage)
    if not 0 <= cached <= prompt or output < 0:
        raise ValueError("Неверные счётчики usage")
    return (
        (prompt - cached) * rates["input"]
        + cached * rates["cached"]
        + output * rates["output"]
    ) / 1000


@dataclass
class UsageEntry:
    model: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    cost_rub: float | None


@dataclass
class SessionMonitor:
    budget_rub: float | None = None
    log: list = field(default_factory=list)

    @property
    def total_cost_rub(self):
        return sum(e.cost_rub for e in self.log if e.cost_rub is not None)

    @property
    def unknown_usage_calls(self):
        return sum(e.cost_rub is None for e in self.log)

    def check_budget(self):
        if self.unknown_usage_calls or (
            self.budget_rub is not None and self.total_cost_rub >= self.budget_rub
        ):
            raise BudgetExceededError(
                "Расход неизвестен или достигнут бюджет; проверьте отчёт перед новой сессией"
            )

    def record(self, model, usage, latency_seconds):
        entry = UsageEntry(
            model,
            latency_seconds,
            read_field(usage, "prompt_tokens"),
            read_field(usage, "completion_tokens"),
            cached_tokens(usage) if usage else None,
            estimate_cost(usage, model),
        )
        self.log.append(entry)
        return entry


class MonitoringCompletions:
    def __init__(self, inner_completions, monitor):
        self._inner, self._monitor = inner_completions, monitor

    def create(self, **kwargs):
        self._monitor.check_budget()
        start = time.monotonic()
        response = None
        try:
            response = self._inner.create(**kwargs)
            return response
        finally:
            self._monitor.record(
                kwargs.get("model", "unknown"),
                read_field(response, "usage"),
                time.monotonic() - start,
            )
