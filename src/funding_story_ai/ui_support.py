from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

_ANSWERED_FLAGS = {
    "primary-details": "primary_answered",
    "secondary-details": "secondary_answered",
    "combined-details": "combined_answered",
}
_ALLOWED_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
_LOCAL_IMAGE_SOURCE = re.compile(r'src="(images/[A-Za-z0-9_.-]+)"')


def mark_stage_answered(stage: str | None, flags: dict[str, bool]) -> dict[str, bool]:
    """Return updated turn flags after a user answers the current worker question."""

    updated = dict(flags)
    flag = _ANSWERED_FLAGS.get(stage or "")
    if flag is not None:
        updated[flag] = True
    return updated


def conversation_payload(messages: list[dict[str, str]]) -> tuple[str, tuple[str, ...]]:
    """Convert chat messages into the worker's initial and question-aware follow-ups."""

    initial = ""
    followups: list[str] = []
    pending_question: str | None = None
    for message in messages:
        role = message.get("role")
        content = message.get("content", "").strip()
        if not content:
            continue
        if role == "assistant":
            pending_question = content
            continue
        if role != "user":
            continue
        if not initial:
            initial = content
        elif pending_question:
            followups.append(f"질문: {pending_question}\n답변: {content}")
        else:
            followups.append(content)
        pending_question = None
    return initial, tuple(followups)


def save_uploaded_image(*, root: Path, input_id: str, filename: str, content: bytes) -> Path:
    """Persist one uploaded reference image under the ignored local artifact directory."""

    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("PNG, JPEG, WebP 이미지만 업로드할 수 있습니다.")
    safe_input_id = re.sub(r"[^a-zA-Z0-9_-]", "-", input_id).strip("-")
    if not safe_input_id:
        raise ValueError("Invalid UI input id")
    upload_dir = root / safe_input_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"reference{suffix}"
    target.write_bytes(content)
    return target


def resolve_run_directory(store_root: Path, run_id: str) -> Path:
    """Resolve a generated run without allowing a caller-controlled path traversal."""

    if not re.fullmatch(r"run-[a-f0-9-]+", run_id):
        raise ValueError("Invalid generated run id")
    return store_root.resolve() / run_id


def load_run_artifacts(store_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = resolve_run_directory(store_root, run_id)
    required = {
        "story": run_dir / "story.json",
        "manifest": run_dir / "images" / "manifest.json",
        "preview": run_dir / "preview.html",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"생성 결과 파일이 없습니다: {', '.join(missing)}")
    return {
        "run_dir": run_dir,
        "story": json.loads(required["story"].read_text(encoding="utf-8")),
        "manifest": json.loads(required["manifest"].read_text(encoding="utf-8")),
        "preview_html": required["preview"].read_text(encoding="utf-8"),
    }


def inline_preview_images(preview_html: str, run_dir: Path) -> str:
    """Embed local preview images so a Streamlit iframe can render the run faithfully."""

    def replace(match: re.Match[str]) -> str:
        relative = Path(match.group(1))
        path = (run_dir / relative).resolve()
        if run_dir.resolve() not in path.parents or not path.is_file():
            return match.group(0)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:{media_type};base64,{encoded}"'

    return _LOCAL_IMAGE_SOURCE.sub(replace, preview_html)
