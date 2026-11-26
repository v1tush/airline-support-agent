from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from alisa.config import DOC_EMBEDDING_MODEL, QUERY_EMBEDDING_MODEL


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class RuleChunk:
    doc_id: str
    airline: str
    text: str


def _extract_title(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _split_sections(text: str) -> list[str]:
    body = re.sub(r"^#\s+.*\n", "", text, count=1)
    parts = re.split(r"\n(?=## )", body)
    return [p.strip() for p in parts if p.strip()]


def load_rule_documents(rules_dir: Path) -> list[RuleChunk]:
    chunks: list[RuleChunk] = []
    for path in sorted(Path(rules_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        airline = _extract_title(text) or path.stem
        for i, section in enumerate(_split_sections(text)):
            chunks.append(
                RuleChunk(doc_id=f"{path.stem}#{i}", airline=airline, text=section)
            )
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if (
        len(a) != len(b)
        or not a
        or not all(type(x) in (int, float) and math.isfinite(x) for x in [*a, *b])
    ):
        raise ValueError("Нужны конечные векторы одной ненулевой размерности")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def build_vocabulary(texts: list[str]) -> list[str]:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(_tokenize(t))
    return sorted(vocab)


class WordOverlapEmbedder:
    def __init__(self, vocabulary: list[str]):
        self._vocab = vocabulary

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(_tokenize(t).count(w)) for w in self._vocab] for t in texts]


def build_word_overlap_embedder(chunks: list[RuleChunk]) -> WordOverlapEmbedder:
    vocab = build_vocabulary([c.text for c in chunks])
    return WordOverlapEmbedder(vocab)


class YandexEmbedder:
    def __init__(
        self,
        client,
        *,
        doc_model: str = DOC_EMBEDDING_MODEL,
        query_model: str = QUERY_EMBEDDING_MODEL,
    ):
        self._client = client
        self._doc_model = doc_model
        self._query_model = query_model

    @lru_cache(maxsize=256)
    def _embed_one(self, text: str, model: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text, model=model, dimensions=256, encoding_format="float"
        )
        return response.data[0].embedding

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t, self._doc_model) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text, self._query_model)


def search_rules(
    query: str, chunks: list[RuleChunk], embedder: Embedder, top_k: int = 2
) -> list[RuleChunk]:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k должен быть положительным целым")
    if not chunks:
        return []
    chunk_vectors = embedder.embed([c.text for c in chunks])
    if hasattr(embedder, "embed_query"):
        query_vector = embedder.embed_query(query)
    else:
        query_vector = embedder.embed([query])[0]
    if len(chunk_vectors) != len(chunks):
        raise ValueError("Число векторов отличается от числа фрагментов")
    scored = [
        (chunk, cosine_similarity(query_vector, vec))
        for chunk, vec in zip(chunks, chunk_vectors)
    ]
    scored = [pair for pair in scored if pair[1] > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_k]]
