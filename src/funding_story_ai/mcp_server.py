from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .adapter import GeminiAdapter
from .data_repository import DataRepository
from .engine import IntegratedStoryMakerExecutor, StoryExecutionInput, StoryExecutor
from .image_generation import (
    GeminiImageAdapter,
    ImageSettings,
    OpenAIImageAdapter,
    RetryingFallbackImageAdapter,
)
from .pipeline import StoryPipeline
from .run_store import LocalRunStore
from .smoke import build_runtime
from .template_retrieval import (
    ExactKnnTemplateRetriever,
    GeminiEmbeddingProvider,
    RetrievalTemplateSelector,
)
from .ui_support import build_run_resource_payload


class CreateStoryRequest(BaseModel):
    caller_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=256)
    brief: dict[str, Any]
    template_id: str | None = None
    reference_image_path: str | None = None


class CreateStoryResponse(BaseModel):
    run_id: str
    status: Literal["accepted", "completed", "failed"]
    result_uri: str
    template_id: str | None = None
    model: str | None = None
    warning_count: int | None = None
    output_status: Literal["complete", "partial"] | None = None
    review_required: Literal[True]
    idempotent_replay: bool
    error_type: str | None = None


class StoryMakerService:
    def __init__(
        self,
        *,
        executor: StoryExecutor,
        store: LocalRunStore,
        pool: ThreadPoolExecutor | None = None,
    ) -> None:
        self.executor = executor
        self.store = store
        self.pool = pool or ThreadPoolExecutor(max_workers=2, thread_name_prefix="story-maker")

    def create(self, request: CreateStoryRequest) -> CreateStoryResponse:
        payload = request.model_dump(exclude={"caller_id", "idempotency_key"}, mode="json")
        record, created = self.store.begin(
            caller_id=request.caller_id,
            idempotency_key=request.idempotency_key,
            request_payload=payload,
        )
        if not created:
            return self._response(record, idempotent_replay=True)
        self.pool.submit(self._execute, record["run_id"], request)
        return self._response(self.store.get(record["run_id"]), idempotent_replay=False)

    def _execute(self, run_id: str, request: CreateStoryRequest) -> None:
        try:
            result = self.executor.execute(
                StoryExecutionInput(
                    brief=request.brief,
                    template_id=request.template_id,
                    run_id=run_id,
                    output_dir=self.store.root / run_id,
                    reference_image_path=(
                        Path(request.reference_image_path)
                        if request.reference_image_path
                        else None
                    ),
                )
            )
            self.store.complete(run_id, result)
        except Exception as exc:
            self.store.fail(run_id, exc)

    @staticmethod
    def _response(
        record: dict[str, Any], *, idempotent_replay: bool
    ) -> CreateStoryResponse:
        result = record.get("result")
        status = {
            "running": "accepted",
            "completed": "completed",
            "failed": "failed",
        }[record["status"]]
        return CreateStoryResponse(
            run_id=record["run_id"],
            status=status,
            result_uri=record["result_uri"],
            template_id=result.get("template_id") if isinstance(result, dict) else None,
            model=result.get("model") if isinstance(result, dict) else None,
            warning_count=(
                int(result.get("warning_count", len(result.get("warnings", []))))
                if isinstance(result, dict)
                else None
            ),
            output_status=result.get("status", "complete") if isinstance(result, dict) else None,
            review_required=True,
            idempotent_replay=idempotent_replay,
            error_type=record.get("error_type"),
        )


def build_story_mcp_server(*, service: StoryMakerService) -> FastMCP:
    server = FastMCP("Funding Story Maker")

    @server.tool(name="create_crowdfunding_story")
    def create_crowdfunding_story(request: CreateStoryRequest) -> CreateStoryResponse:
        return service.create(request)

    @server.resource(
        "story://runs/{run_id}",
        name="crowdfunding_story_result",
        mime_type="application/json",
    )
    def crowdfunding_story_result(run_id: str) -> str:
        record = service.store.get(run_id)
        return json.dumps(
            build_run_resource_payload(service.store.root, record), ensure_ascii=False
        )

    return server


def build_live_service(root: Path | None = None) -> StoryMakerService:
    load_dotenv()
    repository = DataRepository(root)
    settings = build_runtime()
    adapter = GeminiAdapter(settings)
    retriever = ExactKnnTemplateRetriever(
        index=repository.load_template_retrieval_index(),
        embeddings=GeminiEmbeddingProvider(client=adapter.client),
        category_boost=float(os.getenv("TEMPLATE_CATEGORY_BOOST", "0.15")),
    )
    pipeline = StoryPipeline(
        repository=repository,
        adapter=adapter,
        selector=RetrievalTemplateSelector(retriever),
    )
    image_settings = ImageSettings.from_env()
    image_adapters = []
    if os.getenv("OPENAI_API_KEY", "").strip():
        image_adapters.append(OpenAIImageAdapter(image_settings))
    image_adapters.append(GeminiImageAdapter(image_settings, client=adapter.client))
    image_adapter = RetryingFallbackImageAdapter(
        image_adapters,
        attempts_per_provider=image_settings.attempts_per_provider,
    )
    executor = IntegratedStoryMakerExecutor(
        repository=repository,
        pipeline=pipeline,
        image_adapter=image_adapter,
        image_settings=image_settings,
    )
    store_root = Path(os.getenv("STORY_MCP_RUN_STORE", "artifacts/runs"))
    return StoryMakerService(executor=executor, store=LocalRunStore(store_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local story-maker MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1"}:
        raise ValueError("The local MCP server may bind only to a loopback address")
    server = build_story_mcp_server(service=build_live_service())
    server.run(transport="http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
