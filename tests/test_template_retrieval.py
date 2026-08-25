from types import SimpleNamespace

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.template_retrieval import (
    ExactKnnTemplateRetriever,
    GeminiEmbeddingProvider,
)


class _ControlledEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "데일리 스킨케어" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "로봇청소기 가성비" in text or (
                "실속형 로봇청소기" in text and "가성비" in text
            ):
                vectors.append([0.995, 0.1, 0.0])
            elif "문제·자동화" in text or "먼지통과 물걸레 관리" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0, 0.0] if "자동화" in text else [1.0, 0.0, 0.0]


def test_category_soft_boost_recovers_same_category_executable() -> None:
    index = DataRepository().load_template_retrieval_index()
    without = ExactKnnTemplateRetriever(
        index=index, embeddings=_ControlledEmbeddings(), category_boost=0.0
    ).rank(query="가성비 실용형", query_category="테크·가전")
    assert without.ranked[0].candidate_id == "rc14"
    assert without.selected_candidate.rank > 1
    with_boost = ExactKnnTemplateRetriever(
        index=index, embeddings=_ControlledEmbeddings(), category_boost=0.15
    ).rank(query="가성비 실용형", query_category="테크·가전")
    assert with_boost.ranked[0].candidate_id == "rc05"
    assert with_boost.selected_template_id == "t05_value_practical_full_campaign"


def test_gemini_embedding_provider_freezes_task_types_and_dimensions() -> None:
    class Models:
        def __init__(self):
            self.calls = []

        def embed_content(self, *, model, contents, config):
            self.calls.append((model, contents, config))
            count = len(contents) if isinstance(contents, list) else 1
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1] * 768) for _ in range(count)]
            )

    models = Models()
    provider = GeminiEmbeddingProvider(client=SimpleNamespace(models=models))
    assert len(provider.embed_documents(["문서 1", "문서 2"])) == 2
    assert len(provider.embed_query("질의")) == 768
    document_call, query_call = models.calls
    assert document_call[2].task_type == "RETRIEVAL_DOCUMENT"
    assert query_call[2].task_type == "RETRIEVAL_QUERY"
    assert document_call[2].output_dimensionality == 768
