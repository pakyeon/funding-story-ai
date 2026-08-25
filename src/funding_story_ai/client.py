from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapter import GeminiAdapter
from .data_repository import DataRepository
from .pipeline import StoryPipeline
from .smoke import build_runtime


class StoryGenerator:
    """Small public facade over the structured generation pipeline."""

    def __init__(self, repository: DataRepository, pipeline: StoryPipeline) -> None:
        self.repository = repository
        self.pipeline = pipeline

    @classmethod
    def from_env(cls, *, root: Path | None = None) -> StoryGenerator:
        repository = DataRepository(root)
        settings = build_runtime()
        adapter = GeminiAdapter(settings)
        return cls(
            repository=repository,
            pipeline=StoryPipeline(repository=repository, adapter=adapter),
        )

    def generate(
        self,
        brief: dict[str, Any] | str | Path,
        *,
        template: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(brief, dict):
            normalized = brief
        else:
            normalized = self.repository.load_brief_path(Path(brief))
        return self.pipeline.invoke(normalized, template_id=template)
