import argparse
import logging
from datetime import date

from alisa.agent import run_agent_turn
from alisa.config import build_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    client = build_client()
    from alisa.rag import YandexEmbedder

    embedder = YandexEmbedder(client) if args.semantic else None
    from alisa.cache import CachingCompletions
    from alisa.monitoring import MonitoringCompletions, SessionMonitor
    from alisa.parser import as_client

    monitor = SessionMonitor(budget_rub=50)
    client = as_client(
        CachingCompletions(MonitoringCompletions(client.chat.completions, monitor))
    )
    history = None
    while True:
        try:
            text = input("Вы: ").strip()
            if not text:
                break
            result = run_agent_turn(
                text, client, history=history, today=args.today, rule_embedder=embedder
            )
            history = result["messages"]
            print("Алиса:", result["final_text"])
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"Запрос не завершён: {type(exc).__name__}: {exc}")
    print(
        "Известный расход, ₽:",
        monitor.total_cost_rub,
        "; операций с неизвестным расходом:",
        monitor.unknown_usage_calls,
    )


if __name__ == "__main__":
    main()
