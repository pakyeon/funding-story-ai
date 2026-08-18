import asyncio
import json
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
            "request_id": "request-one",
            "caller_id": "worker-one",
            "idempotency_key": "idempotency-one",
            "brief": DataRepository().load_brief(),
            "template_id": "t02_problem_solution_automation",
            "category_profile_id": "robot-vacuum-ko-v1",
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
            first_task = await client.call_tool(
                "create_crowdfunding_story", arguments, task=True
            )
            first = (await first_task.result()).structured_content
            assert first is not None
            assert first["idempotent_replay"] is False
            contents = await client.read_resource(first["result_uri"])
            assert json.loads(contents[0].text)["result"]["review_required"] is True
            replay_task = await client.call_tool(
                "create_crowdfunding_story", arguments, task=True
            )
            replay = (await replay_task.result()).structured_content
            assert replay is not None
            assert replay["run_id"] == first["run_id"]
            assert replay["idempotent_replay"] is True

    asyncio.run(exercise())
    assert len(executor.calls) == 1
