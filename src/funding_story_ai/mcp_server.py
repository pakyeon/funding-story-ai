from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .adapter import GeminiAdapter
from .data_repository import DataRepository
from .engine import IntegratedStoryMakerExecutor, StoryExecutionInput, StoryExecutor
from .image_generation import ImageSettings, ImageUsageLedger, OpenAIImageAdapter
from .pipeline import StoryPipeline
from .run_store import LocalRunStore
from .smoke import build_runtime
from .template_retrieval import (
    ExactKnnTemplateRetriever,
    GeminiEmbeddingProvider,
    RetrievalTemplateSelector,
)


class CreateStoryRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    caller_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=256)
    brief: dict[str, Any]
    template_id: str | None = None
    category_profile_id: str | None = None
    reference_image_path: str | None = None
    generate_images: bool = True


class CreateStoryResponse(BaseModel):
    run_id: str
    status: Literal["completed"]
    result_uri: str
    request_id: str
    template_id: str
    model: str
    warning_count: int
    output_status: Literal["complete", "partial"]
    review_required: Literal[True]
    idempotent_replay: bool


class StoryMakerService:
    def __init__(self, *, executor: StoryExecutor, store: LocalRunStore) -> None:
        self.executor = executor
        self.store = store

    def create(self, request: CreateStoryRequest) -> CreateStoryResponse:
        payload = request.model_dump(exclude={"caller_id", "idempotency_key"}, mode="json")
        record, created = self.store.begin(
            caller_id=request.caller_id,
            idempotency_key=request.idempotency_key,
            request_payload=payload,
        )
        if not created:
            return self._response(request.request_id, record, idempotent_replay=True)
        try:
            result = self.executor.execute(
                StoryExecutionInput(
                    brief=request.brief,
                    template_id=request.template_id,
                    category_profile_id=request.category_profile_id,
                    run_id=record["run_id"],
                    output_dir=self.store.root / record["run_id"],
                    reference_image_path=(
                        Path(request.reference_image_path)
                        if request.reference_image_path
                        else None
                    ),
                    generate_images=request.generate_images,
                )
            )
            record = self.store.complete(record["run_id"], result)
        except Exception as exc:
            self.store.fail(record["run_id"], exc)
            raise
        return self._response(request.request_id, record, idempotent_replay=False)

    @staticmethod
    def _response(
        request_id: str, record: dict[str, Any], *, idempotent_replay: bool
    ) -> CreateStoryResponse:
        result = record["result"]
        if record["status"] != "completed" or not isinstance(result, dict):
            raise RuntimeError(f"Run is not complete: {record['run_id']}")
        return CreateStoryResponse(
            run_id=record["run_id"],
            status="completed",
            result_uri=record["result_uri"],
            request_id=request_id,
            template_id=result["template_id"],
            model=result["model"],
            warning_count=int(result.get("warning_count", len(result.get("warnings", [])))),
            output_status=result.get("status", "complete"),
            review_required=True,
            idempotent_replay=idempotent_replay,
        )


def build_story_mcp_server(*, service: StoryMakerService) -> FastMCP:
    server = FastMCP("Funding Story Maker")

    @server.tool(name="create_crowdfunding_story", task=True)
    async def create_crowdfunding_story(request: CreateStoryRequest) -> CreateStoryResponse:
        return await asyncio.to_thread(service.create, request)

    @server.resource(
        "story://runs/{run_id}",
        name="crowdfunding_story_result",
        mime_type="application/json",
    )
    def crowdfunding_story_result(run_id: str) -> str:
        return json.dumps(service.store.get(run_id), ensure_ascii=False)

    return server


def build_live_service(root: Path | None = None) -> StoryMakerService:
    load_dotenv()
    repository = DataRepository(root)
    settings, ledger = build_runtime()
    adapter = GeminiAdapter(settings, ledger)
    retriever = ExactKnnTemplateRetriever(
        index=repository.load_template_retrieval_index(),
        embeddings=GeminiEmbeddingProvider(client=adapter.client),
        category_boost=float(os.getenv("TEMPLATE_CATEGORY_BOOST", "0.1")),
    )
    pipeline = StoryPipeline(
        repository=repository,
        adapter=adapter,
        selector=RetrievalTemplateSelector(retriever),
    )
    image_settings = ImageSettings.from_env()
    image_ledger = ImageUsageLedger(
        image_settings.ledger_path, image_settings.spend_limit_usd
    )
    executor = IntegratedStoryMakerExecutor(
        repository=repository,
        pipeline=pipeline,
        image_adapter=OpenAIImageAdapter(image_settings, image_ledger),
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
