from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from fastmcp import Client

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_ALLOWED_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
_LOCAL_IMAGE_SOURCE = re.compile(r'src="(images/[A-Za-z0-9_.-]+)"')


def save_uploaded_image(*, root: Path, input_id: str, filename: str, content: bytes) -> Path:
    """Persist one validated chat attachment under the ignored artifact directory."""

    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("PNG, JPEG, WebP 이미지만 업로드할 수 있습니다.")
    if not content:
        raise ValueError("빈 이미지 파일은 업로드할 수 없습니다.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("이미지는 10MB 이하여야 합니다.")
    safe_input_id = re.sub(r"[^a-zA-Z0-9_-]", "-", input_id).strip("-")
    if not safe_input_id:
        raise ValueError("Invalid UI input id")
    upload_dir = root / safe_input_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"reference{suffix}"
    target.write_bytes(content)
    return target


def _data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def inline_preview_images(preview_html: str, run_dir: Path) -> str:
    """Embed only run-local images in an HTML preview."""

    resolved_run = run_dir.resolve()

    def replace(match: re.Match[str]) -> str:
        path = (resolved_run / Path(match.group(1))).resolve()
        if resolved_run not in path.parents or not path.is_file():
            return match.group(0)
        return f'src="{_data_url(path)}"'

    return _LOCAL_IMAGE_SOURCE.sub(replace, preview_html)


def build_run_resource_payload(store_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Build the UI-safe resource representation at the MCP server boundary."""

    payload = dict(record)
    if record["status"] != "completed":
        return payload
    run_id = str(record["run_id"])
    if not re.fullmatch(r"run-[a-f0-9-]+", run_id):
        raise ValueError("Invalid generated run id")
    run_dir = store_root.resolve() / run_id
    result = record.get("result") or {}
    if not {"story", "images", "preview"}.issubset(result):
        return payload

    def artifact_path(name: str) -> Path:
        relative = Path(result[name]["path"])
        path = (run_dir / relative).resolve()
        if run_dir.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Missing run artifact: {name}")
        return path

    story = json.loads(artifact_path("story").read_text(encoding="utf-8"))
    manifest_path = run_dir / result["images"]["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preview_path = artifact_path("preview")
    image_data = {
        asset["path"]: _data_url(run_dir / "images" / asset["path"])
        for asset in manifest["assets"]
        if asset["status"] == "success" and asset.get("path")
    }
    payload["artifacts"] = {
        "story": story,
        "manifest": manifest,
        "preview_html": inline_preview_images(
            preview_path.read_text(encoding="utf-8"), run_dir
        ),
        "image_data": image_data,
    }
    return payload


async def read_run_resource(server_url: str, result_uri: str) -> dict[str, Any]:
    """Read a run through FastMCP instead of reaching into the server filesystem."""

    async with Client(server_url) as client:
        contents = await client.read_resource(result_uri)
    value = json.loads(contents[0].text)
    if not isinstance(value, dict):
        raise ValueError("Run resource must contain a JSON object")
    return value
