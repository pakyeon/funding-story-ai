from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from fastmcp import Client

from .data_repository import DataRepository


async def _run(args: argparse.Namespace) -> None:
    repository = DataRepository()
    brief = repository.load_brief_path(args.brief_path)
    arguments = {
        "request": {
            "request_id": args.request_id or str(uuid.uuid4()),
            "caller_id": args.caller_id,
            "idempotency_key": args.idempotency_key or str(uuid.uuid4()),
            "brief": brief,
            "template_id": args.template,
            "category_profile_id": args.category_profile,
            "reference_image_path": (
                str(args.reference_image) if args.reference_image else None
            ),
            "generate_images": True,
        }
    }
    async with Client(args.server_url) as client:
        tools = await client.list_tools()
        if [tool.name for tool in tools] != ["create_crowdfunding_story"]:
            raise RuntimeError("Unexpected MCP tool allowlist")
        task = await client.call_tool("create_crowdfunding_story", arguments, task=True)
        result = await task.result()
        print(json.dumps(result.structured_content, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a structured brief to the local FastMCP server"
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument("--brief-path", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--caller-id", default="local-cli")
    parser.add_argument("--request-id")
    parser.add_argument("--idempotency-key")
    parser.add_argument("--template")
    parser.add_argument("--category-profile", default="robot-vacuum-ko-v1")
    parser.add_argument("--live", action="store_true", required=True)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
