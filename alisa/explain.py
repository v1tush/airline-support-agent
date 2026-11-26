from __future__ import annotations

from alisa.config import MODEL
from alisa.parser import ChatClient


def build_explain_prompt(query: dict, top_flights: list[dict]) -> str:
    lines = [
        f"Пользователь искал рейсы {query['from']} -> {query['to']}.",
        "Вот найденные варианты, уже отобранных и отранжированных по цене, "
        "длительности пересадки и времени в пути:",
    ]
    for i, f in enumerate(top_flights, start=1):
        lines.append(
            f"{i}. ID {f['id']}, {f['airline']}, {f['price']}₽ на человека, {f.get('total_price', f['price'])}₽ на группу, пересадка "
            f"{f['layover_hours']} ч, в пути {f['duration_hours']} ч."
        )
    lines.append(
        "Объясни пользователю простым языком, почему эти варианты "
        "хорошие, не придумывая факты сверх приведённых чисел."
    )
    return "\n".join(lines)


def explain_top_flights(
    query: dict,
    top_flights: list[dict],
    client: ChatClient,
    model: str = MODEL,
) -> str:
    if not top_flights:
        return "По вашему запросу подходящих рейсов не нашлось."
    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        messages=[
            {"role": "user", "content": build_explain_prompt(query, top_flights)}
        ],
    )
    from alisa.parser import checked_message, read_field

    return read_field(checked_message(response), "content")
