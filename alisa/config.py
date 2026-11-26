import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
MODEL = f"gpt://{FOLDER_ID}/qwen3.6-35b-a3b/latest"

DOC_EMBEDDING_MODEL = f"emb://{FOLDER_ID}/text-embeddings-v2-doc/latest"
QUERY_EMBEDDING_MODEL = f"emb://{FOLDER_ID}/text-embeddings-v2-query/latest"

FAST_MODEL = MODEL
STRONG_MODEL = MODEL


def build_client():
    api_key = os.getenv("YANDEX_API_KEY", "")
    if not api_key or not FOLDER_ID:
        raise RuntimeError("Заполните YANDEX_API_KEY и YANDEX_FOLDER_ID в .env")
    return OpenAI(
        api_key=api_key,
        project=FOLDER_ID,
        base_url="https://ai.api.cloud.yandex.net/v1",
        timeout=60,
        max_retries=1,
    )


if __name__ == "__main__":
    print(f"Модель: {MODEL}")
    print("Ключ задан" if os.getenv("YANDEX_API_KEY") else "Ключ не задан")
    print("Это проверка настроек; доступ проверяет python -m alisa.hello_api")
