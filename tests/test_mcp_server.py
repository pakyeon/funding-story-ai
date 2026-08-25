import asyncio
import json
import time
from typing import Any

from fastmcp import Client

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.engine import StoryExecutionInput
from funding_story_ai.mcp_server import StoryMakerService, build_story_mcp_server
from funding_story_ai.run_store import LocalRunStore


class _Executor:
    def __init__(self) -> None:
        self.calls: list[StoryExecutionInput] = []

    def execute(self, value: StoryExecutionInput) -> dict[str, Any]:
        self.calls.append(value)
        return {
            "status": "complete",
            "template_id": value.template_id or "t04_full_campaign",
            "model": "fake-model",
            "warnings": [],
            "review_required": True,
        }


def test_mcp_contract_task_resource_and_idempotency(tmp_path) -> None:
    executor = _Executor()
    service = StoryMakerService(executor=executor, store=LocalRunStore(tmp_path / "runs"))
    server = build_story_mcp_server(service=service)
    arguments = {
        "request": {
            "caller_id": "worker-one",
            "idempotency_key": "idempotency-one",
            "brief": DataRepository().load_brief(),
            "template_id": "t02_problem_solution_automation",
        }
    }

    async def exercise() -> None:
        async with Client(server) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools] == ["create_crowdfunding_story"]
            resources = await client.list_resource_templates()
            assert [str(resource.uriTemplate) for resource in resources] == [
                "story://runs/{run_id}"
            ]
            first_result = await client.call_tool("create_crowdfunding_story", arguments)
            first = first_result.structured_content
            assert first is not None
            assert first["idempotent_replay"] is False
            assert first["status"] in {"accepted", "completed"}
            for _ in range(100):
                contents = await client.read_resource(first["result_uri"])
                record = json.loads(contents[0].text)
                if record["status"] == "completed":
                    break
                time.sleep(0.01)
            assert record["result"]["review_required"] is True
            replay_result = await client.call_tool("create_crowdfunding_story", arguments)
            replay = replay_result.structured_content
            assert replay is not None
            assert replay["run_id"] == first["run_id"]
            assert replay["idempotent_replay"] is True

    asyncio.run(exercise())
    assert len(executor.calls) == 1
