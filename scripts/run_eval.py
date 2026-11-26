from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


def _configure_stdout() -> None:
    # Windows consoles may use cp1251; model text can contain characters outside it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="Прогон golden set Алисы через реальный API модели"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="печатать reasoning судьи по каждому кейсу",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.9,
        help="целевой pass rate (0.0-1.0) - скрипт завершится ненулевым кодом, если не достигнут",
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--today", type=date.fromisoformat, default=date(2026, 7, 21))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Файл отчёта уже существует")

    load_dotenv()

    from alisa.config import build_client
    from alisa.eval_harness import run_golden_set
    from alisa.golden_set import load_golden_set

    client = build_client()
    cases = load_golden_set(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    print(f"Golden set: {len(cases)} кейсов")

    from alisa.rag import YandexEmbedder

    report = run_golden_set(
        cases,
        client,
        today=args.today,
        rule_embedder=YandexEmbedder(client) if args.semantic else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(
            asdict(report),
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda x: (
                x.model_dump(exclude_none=True) if hasattr(x, "model_dump") else vars(x)
            ),
        )

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id}: {result.query}")
        if not result.passed:
            print(f"    tool-check: {result.tool_check_detail}")
            print(f"    judge: {result.judge_verdict} - {result.judge_reasoning}")
        elif args.verbose:
            print(f"    judge: {result.judge_verdict} - {result.judge_reasoning}")

    print()
    print(
        f"Pass rate: {report.pass_rate:.0%} ({len(report.results) - len(report.failures())}/{len(report.results)})"
    )

    if report.pass_rate < args.target:
        print(
            f"Не достигнута целевая метрика {args.target:.0%} - см. проваленные кейсы выше."
        )
        sys.exit(1)
    print(f"Целевая метрика {args.target:.0%} достигнута.")


if __name__ == "__main__":
    main()
