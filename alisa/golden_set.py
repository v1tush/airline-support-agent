from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GOLDEN_SET_PATH = Path(__file__).parent / "golden" / "golden_set.json"


@dataclass
class GoldenCase:
    id: str
    query: str
    expected_tool: str | None
    expected_args_contains: dict
    rubric: str
    turns: list = field(default_factory=list)
    expected_calls: list | None = None


def load_golden_set(path: Path | None = None) -> list[GoldenCase]:
    path = path or DEFAULT_GOLDEN_SET_PATH
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldenCase(
            id=item["id"],
            query=item.get("query", ""),
            expected_tool=item.get("expected_tool"),
            expected_args_contains=item.get("expected_args_contains") or {},
            rubric=item.get("rubric", ""),
            turns=item.get("turns", []),
            expected_calls=item.get("expected_calls"),
        )
        for item in raw
    ]
