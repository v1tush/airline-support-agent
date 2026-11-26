from pathlib import Path

from alisa.rag import (
    RuleChunk,
    YandexEmbedder,
    build_vocabulary,
    build_word_overlap_embedder,
    cosine_similarity,
    load_rule_documents,
    search_rules,
)

RULES_DIR = Path(__file__).parent.parent / "alisa" / "rules"


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbeddingsResponse:
    def __init__(self, embedding):
        self.data = [_FakeEmbeddingItem(embedding)]


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, *, input, model, encoding_format, dimensions=256):
        self.calls.append(
            {"input": input, "model": model, "encoding_format": encoding_format}
        )
        return _FakeEmbeddingsResponse([1.0, 0.0])


class _FakeEmbeddingsClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


class _FakeQueryAwareEmbedder:
    def __init__(self):
        self.query_calls = []

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [1.0, 0.0]


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_handles_zero_vector_without_division_error():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_load_rule_documents_splits_into_sections_with_airline_title():
    chunks = load_rule_documents(RULES_DIR)
    assert len(chunks) > 0
    airlines = {c.airline for c in chunks}
    assert "Аэрофлот" in airlines
    assert "Победа" in airlines
    assert "S7" in airlines
    # Каждый файл разбит минимум на 3 раздела (кладь, багаж, возврат)
    aeroflot_chunks = [c for c in chunks if c.airline == "Аэрофлот"]
    assert len(aeroflot_chunks) >= 3


def test_build_vocabulary_is_deterministic_and_lowercased():
    vocab = build_vocabulary(["Багаж и Ручная кладь", "багаж"])
    assert vocab == sorted(vocab)
    assert "багаж" in vocab
    assert "Багаж" not in vocab


def test_search_rules_finds_baggage_section_for_baggage_question():
    chunks = load_rule_documents(RULES_DIR)
    embedder = build_word_overlap_embedder(chunks)

    top = search_rules("сколько багажа можно провезти", chunks, embedder, top_k=1)

    assert len(top) == 1
    assert "багаж" in top[0].text.lower()


def test_search_rules_narrows_to_specific_airline_when_filtered_upstream():
    chunks = load_rule_documents(RULES_DIR)
    pobeda_chunks = [c for c in chunks if c.airline == "Победа"]
    embedder = build_word_overlap_embedder(pobeda_chunks)

    top = search_rules("возврат билета", pobeda_chunks, embedder, top_k=1)

    assert top[0].airline == "Победа"
    assert "возврат" in top[0].text.lower() or "невозвратные" in top[0].text.lower()


def test_search_rules_returns_empty_list_for_empty_chunks():
    embedder = build_word_overlap_embedder([RuleChunk("x#0", "X", "текст")])
    assert search_rules("что угодно", [], embedder, top_k=2) == []


def test_yandex_embedder_uses_different_models_for_documents_and_queries():
    # Документы и запрос используют разные совместимые модели embeddings.
    client = _FakeEmbeddingsClient()
    embedder = YandexEmbedder(client, doc_model="doc-model", query_model="query-model")

    embedder.embed(["Багаж не включён в базовый тариф."])
    query_vector = embedder.embed_query("сколько багажа можно провезти")

    calls = client.embeddings.calls
    assert calls[0]["model"] == "doc-model"
    assert calls[1]["model"] == "query-model"
    assert calls[1]["input"] == "сколько багажа можно провезти"
    # input - одна строка, а не список: так требует API.
    assert isinstance(calls[0]["input"], str)
    assert calls[0]["encoding_format"] == "float"
    # embed_query отдаёт ОДИН вектор, а не список векторов
    assert query_vector == [1.0, 0.0]


def test_yandex_embedder_sends_one_request_per_text():
    # На каждый текст нужен отдельный запрос embeddings.
    client = _FakeEmbeddingsClient()
    embedder = YandexEmbedder(client)

    vectors = embedder.embed(["первый", "второй", "третий"])

    assert len(client.embeddings.calls) == 3
    assert len(vectors) == 3


def test_search_rules_uses_embed_query_when_embedder_provides_it():
    chunks = [RuleChunk("x#0", "X", "багаж"), RuleChunk("x#1", "X", "возврат билета")]
    embedder = _FakeQueryAwareEmbedder()

    top = search_rules("сколько багажа можно провезти", chunks, embedder, top_k=1)

    assert embedder.query_calls == ["сколько багажа можно провезти"]
    assert len(top) == 1


def test_search_rules_falls_back_to_embed_when_embedder_has_no_query_mode():
    # У WordOverlapEmbedder отдельного режима для запроса нет -
    # search_rules обязан продолжать работать через обычный embed().
    chunks = load_rule_documents(RULES_DIR)
    embedder = build_word_overlap_embedder(chunks)
    assert not hasattr(embedder, "embed_query")

    top = search_rules("сколько багажа можно провезти", chunks, embedder, top_k=1)

    assert "багаж" in top[0].text.lower()
