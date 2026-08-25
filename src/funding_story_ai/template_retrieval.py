from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from google.genai import types

from .selector import TemplateSelection


class TemplateRetrievalError(RuntimeError):
    pass


class NonExecutableTopResult(TemplateRetrievalError):
    pass


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingProvider:
    """Vertex AI embedding boundary with explicit retrieval task types."""

    def __init__(
        self,
        *,
        client: Any,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
    ) -> None:
        self.client = client
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dimensions,
            ),
        )
        return self._vectors(response, expected=len(texts))

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("retrieval query must not be empty")
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.dimensions,
            ),
        )
        return self._vectors(response, expected=1)[0]

    def _vectors(self, response: Any, *, expected: int) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None) or []
        vectors = [list(getattr(item, "values", None) or []) for item in embeddings]
        if len(vectors) != expected:
            raise TemplateRetrievalError(
                f"Embedding count mismatch: expected={expected}, actual={len(vectors)}"
            )
        if any(len(vector) != self.dimensions for vector in vectors):
            raise TemplateRetrievalError(
                f"Embedding dimensionality must be {self.dimensions}"
            )
        return vectors


@dataclass(frozen=True, slots=True)
class RankedTemplate:
    rank: int
    candidate_id: str
    name: str
    category: str
    executable_template_id: str | None
    semantic_score: float
    category_boost: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "name": self.name,
            "category": self.category,
            "executable_template_id": self.executable_template_id,
            "semantic_score": self.semantic_score,
            "category_boost": self.category_boost,
            "final_score": self.final_score,
        }


@dataclass(frozen=True, slots=True)
class TemplateRetrievalResult:
    query: str
    query_category: str
    category_boost: float
    ranked: tuple[RankedTemplate, ...]

    @property
    def selected_template_id(self) -> str:
        for candidate in self.ranked:
            if candidate.executable_template_id is not None:
                return candidate.executable_template_id
        raise NonExecutableTopResult("No executable template exists in the ranked results")

    @property
    def selected_candidate(self) -> RankedTemplate:
        template_id = self.selected_template_id
        return next(
            candidate
            for candidate in self.ranked
            if candidate.executable_template_id == template_id
        )


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise TemplateRetrievalError("Zero embedding cannot be normalized")
    return [value / magnitude for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise TemplateRetrievalError("Embedding dimensionality mismatch")
    return sum(
        a * b
        for a, b in zip(_normalize(left), _normalize(right), strict=True)
    )


def candidate_document(candidate: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"카테고리: {candidate['category']}",
            f"제품 유형: {', '.join(candidate['product_types'])}",
            f"문제: {', '.join(candidate['problems'])}",
            f"타깃: {candidate['target']}",
            f"핵심 메시지: {candidate['core_message']}",
            f"설득 축: {', '.join(candidate['persuasion_axis'])}",
            f"톤: {', '.join(candidate['tone_keywords'])}",
            f"섹션 역할: {', '.join(candidate['section_roles'])}",
        ]
    )


def brief_query_document(brief: dict[str, Any]) -> str:
    problems = ", ".join(item["description"] for item in brief["problems"])
    audiences = ", ".join(item["description"] for item in brief["audiences"])
    features = ", ".join(
        f"{item['name']}: {item['description']}" for item in brief["features"]
    )
    claims = ", ".join(item["statement"] for item in brief["claims"])
    return "\n".join(
        [
            f"카테고리: {brief['product']['category']}",
            f"제품 유형: {brief['product']['product_type']}",
            f"제품 요약: {brief['product']['summary']}",
            f"문제: {problems or '미제공'}",
            f"타깃: {audiences or '미제공'}",
            f"기능: {features or '미제공'}",
            f"주장: {claims or '미제공'}",
        ]
    )


class ExactKnnTemplateRetriever:
    """Exact cosine KNN plus a bounded same-category soft boost."""

    ALLOWED_BOOSTS = {0.0, 0.1, 0.15, 0.2}

    def __init__(
        self,
        *,
        index: dict[str, Any],
        embeddings: EmbeddingProvider,
        category_boost: float = 0.15,
    ) -> None:
        if category_boost not in self.ALLOWED_BOOSTS:
            raise ValueError(f"category_boost must be one of {sorted(self.ALLOWED_BOOSTS)}")
        self.index = index
        self.embeddings = embeddings
        self.category_boost = category_boost
        self._document_vectors: list[list[float]] | None = None

    def _vectors(self) -> list[list[float]]:
        if self._document_vectors is None:
            documents = [candidate_document(item) for item in self.index["candidates"]]
            self._document_vectors = self.embeddings.embed_documents(documents)
            if len(self._document_vectors) != len(documents):
                raise TemplateRetrievalError("Candidate embedding count mismatch")
        return self._document_vectors

    def rank(
        self,
        *,
        query: str,
        query_category: str,
        limit: int | None = None,
    ) -> TemplateRetrievalResult:
        query_vector = self.embeddings.embed_query(query)
        values: list[tuple[dict[str, Any], float, float, float]] = []
        for candidate, vector in zip(
            self.index["candidates"], self._vectors(), strict=True
        ):
            semantic_score = _cosine(query_vector, vector)
            boost = self.category_boost if candidate["category"] == query_category else 0.0
            values.append((candidate, semantic_score, boost, semantic_score + boost))
        values.sort(key=lambda item: (-item[3], item[0]["candidate_id"]))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            values = values[:limit]
        ranked = tuple(
            RankedTemplate(
                rank=position,
                candidate_id=candidate["candidate_id"],
                name=candidate["name"],
                category=candidate["category"],
                executable_template_id=candidate["executable_template_id"],
                semantic_score=round(semantic, 8),
                category_boost=boost,
                final_score=round(final, 8),
            )
            for position, (candidate, semantic, boost, final) in enumerate(values, start=1)
        )
        return TemplateRetrievalResult(
            query=query,
            query_category=query_category,
            category_boost=self.category_boost,
            ranked=ranked,
        )

    def select(self, brief: dict[str, Any]) -> TemplateRetrievalResult:
        return self.rank(
            query=brief_query_document(brief),
            query_category=brief["product"]["category"],
        )


class RetrievalTemplateSelector:
    """Bridge retrieval into the deterministic story pipeline."""

    def __init__(self, retriever: ExactKnnTemplateRetriever) -> None:
        self.retriever = retriever
        self.last_result: TemplateRetrievalResult | None = None

    def select(
        self,
        brief: dict[str, Any],
        templates: list[dict[str, Any]],
    ) -> TemplateSelection:
        result = self.retriever.select(brief)
        template_id = result.selected_template_id
        if template_id not in {template["id"] for template in templates}:
            raise TemplateRetrievalError(
                f"Retrieved executable template is unavailable: {template_id}"
            )
        self.last_result = result
        scores = {
            item.executable_template_id: round(item.final_score * 1_000_000)
            for item in result.ranked
            if item.executable_template_id is not None
        }
        selected = result.selected_candidate
        return TemplateSelection(
            template_id=template_id,
            score=round(selected.final_score * 1_000_000),
            scores=scores,
            reasons=(
                f"exact KNN executable rank {selected.rank}: {selected.candidate_id}",
                f"semantic score: {selected.semantic_score:.8f}",
                f"category soft boost: {selected.category_boost:.2f}",
            ),
        )
