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
_RUN_ID = re.compile(r"run-[a-f0-9-]+$")
_LOCAL_IMAGE_SOURCE = re.compile(r'src=["\'](images/[A-Za-z0-9_.-]+)["\']')


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
        relative = Path(match.group(1))
        path = (resolved_run / relative).resolve()
        if resolved_run not in path.parents or not path.is_file():
            return match.group(0)
        quote = match.group(0)[4]
        return f"src={quote}{_data_url(path)}{quote}"

    return _LOCAL_IMAGE_SOURCE.sub(replace, preview_html)


def _run_dir(store_root: Path, run_id: str) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid generated run id")
    root = store_root.resolve()
    run_dir = (root / run_id).resolve()
    if root not in run_dir.parents:
        raise ValueError("Run directory escapes the configured artifact root")
    return run_dir


def _artifact_path(run_dir: Path, relative: str) -> Path:
    path = (run_dir / Path(relative)).resolve()
    if run_dir not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Missing run artifact: {relative}")
    return path


def load_run_artifacts(store_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Load user-facing HTML and source files for a completed MCP run."""

    if record.get("status") != "completed":
        raise ValueError("Only completed runs have display artifacts")
    result = record.get("result")
    if not isinstance(result, dict):
        raise ValueError("Completed run has no result")
    run_dir = _run_dir(store_root, str(record["run_id"]))
    story_path = _artifact_path(run_dir, result["story"]["path"])
    manifest_path = _artifact_path(run_dir, result["images"]["manifest"]["path"])
    media_plan_path = _artifact_path(run_dir, result["media_plan"]["path"])
    draft_path = _artifact_path(run_dir, result["draft_html"]["path"])
    publishable_info = result.get("publishable_html")
    publishable_path = (
        _artifact_path(run_dir, publishable_info["path"])
        if isinstance(publishable_info, dict)
        else None
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_data = {
        asset["path"]: _data_url(_artifact_path(run_dir, f"images/{asset['path']}"))
        for asset in manifest["assets"]
        if asset["status"] == "success" and asset.get("path")
    }
    generated_reference_data = {
        asset["asset_id"]: _data_url(_artifact_path(run_dir, asset["path"]))
        for asset in manifest.get("generated_references", [])
    }
    source_files = {
        "story.json": story_path.read_text(encoding="utf-8"),
        "media-facts.json": _artifact_path(run_dir, result["media_facts"]["path"]).read_text(
            encoding="utf-8"
        ),
        "media-plan.json": media_plan_path.read_text(encoding="utf-8"),
        "images/manifest.json": manifest_path.read_text(encoding="utf-8"),
        "draft.html": draft_path.read_text(encoding="utf-8"),
    }
    if publishable_path is not None:
        source_files["publishable.html"] = publishable_path.read_text(encoding="utf-8")
    return {
        "run_dir": run_dir,
        "story": json.loads(story_path.read_text(encoding="utf-8")),
        "manifest": manifest,
        "draft_html": inline_preview_images(source_files["draft.html"], run_dir),
        "publishable_html": (
            inline_preview_images(source_files["publishable.html"], run_dir)
            if "publishable.html" in source_files
            else None
        ),
        "image_data": image_data,
        "generated_reference_data": generated_reference_data,
        "source_files": source_files,
    }


def build_run_resource_payload(store_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Add display-safe artifacts to a completed MCP resource response."""

    payload = dict(record)
    if record.get("status") != "completed":
        return payload
    result = record.get("result")
    if not isinstance(result, dict) or not {
        "story",
        "images",
        "media_facts",
        "media_plan",
        "draft_html",
    }.issubset(result):
        return payload
    artifacts = load_run_artifacts(store_root, record)
    payload["artifacts"] = {
        "story": artifacts["story"],
        "manifest": artifacts["manifest"],
        "draft_html": artifacts["draft_html"],
        "publishable_html": artifacts["publishable_html"],
        "image_data": artifacts["image_data"],
        "generated_reference_data": artifacts["generated_reference_data"],
        "source_files": artifacts["source_files"],
    }
    return payload


async def read_run_resource(server_url: str | Any, result_uri: str) -> dict[str, Any]:
    """Read a run through FastMCP instead of bypassing the server boundary."""

    async with Client(server_url) as client:
        contents = await client.read_resource(result_uri)
    if not contents or not getattr(contents[0], "text", None):
        raise ValueError("Run resource returned no JSON content")
    value = json.loads(contents[0].text)
    if not isinstance(value, dict):
        raise ValueError("Run resource must contain a JSON object")
    return value
