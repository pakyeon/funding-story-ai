from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.ui_support import (
    conversation_payload,
    inline_preview_images,
    load_run_artifacts,
    mark_stage_answered,
    save_uploaded_image,
)
from funding_story_ai.worker import WorkerOutcome, WorkerRequest, build_live_worker

ROOT = Path(__file__).resolve().parent
DEFAULT_FLAGS = {
    "primary_answered": False,
    "combined_answered": False,
    "secondary_answered": False,
}


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "input_id": f"ui-{uuid.uuid4()}",
        "messages": [],
        "stage": None,
        "answer_flags": dict(DEFAULT_FLAGS),
        "tool_result": None,
        "reference_image_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset() -> None:
    for key in (
        "input_id",
        "messages",
        "stage",
        "answer_flags",
        "tool_result",
        "reference_image_path",
        "reference_image",
    ):
        st.session_state.pop(key, None)


def _worker_request(
    *, server_url: str, profile_id: str, confirmed: bool = False, skip: bool = False
) -> WorkerOutcome:
    initial, followups = conversation_payload(st.session_state.messages)
    flags = st.session_state.answer_flags
    request = WorkerRequest(
        input_id=st.session_state.input_id,
        initial_message=initial,
        followup_messages=followups,
        image_path=st.session_state.reference_image_path,
        profile_id=profile_id,
        primary_answered=flags["primary_answered"],
        combined_answered=flags["combined_answered"],
        secondary_answered=flags["secondary_answered"],
        skip_requested=skip,
        confirmed=confirmed,
        caller_id="streamlit-demo",
        idempotency_key=f"streamlit-{st.session_state.input_id}-v1",
    )
    worker = build_live_worker(server_url=server_url, root=ROOT)
    return asyncio.run(worker.handle(request))


def _handle_outcome(outcome: WorkerOutcome) -> None:
    st.session_state.stage = outcome.stage
    if outcome.status == "submitted":
        st.session_state.tool_result = outcome.tool_result
        return
    for question in outcome.questions:
        message = {"role": "assistant", "content": question}
        if not st.session_state.messages or st.session_state.messages[-1] != message:
            st.session_state.messages.append(message)


def _run_worker(
    *, server_url: str, profile_id: str, confirmed: bool = False, skip: bool = False
) -> bool:
    label = "스토리를 생성하고 있습니다…" if confirmed or skip else "입력을 확인하고 있습니다…"
    try:
        with st.spinner(label):
            outcome = _worker_request(
                server_url=server_url,
                profile_id=profile_id,
                confirmed=confirmed,
                skip=skip,
            )
        _handle_outcome(outcome)
        return True
    except Exception as exc:
        st.error(f"실행하지 못했습니다: {type(exc).__name__}: {exc}")
        return False


def _render_result() -> None:
    tool_result = st.session_state.tool_result
    if not tool_result:
        return
    run_id = tool_result.get("run_id")
    if not run_id:
        st.error("생성 응답에 run_id가 없습니다.")
        return
    load_dotenv(ROOT / ".env")
    store_root = Path(os.getenv("STORY_MCP_RUN_STORE", "artifacts/runs"))
    if not store_root.is_absolute():
        store_root = ROOT / store_root
    try:
        artifacts = load_run_artifacts(store_root, run_id)
    except Exception as exc:
        st.error(f"로컬 생성 결과를 읽지 못했습니다: {exc}")
        return

    story = artifacts["story"]
    manifest = artifacts["manifest"]
    st.success("스토리 초안이 생성되었습니다. 게시 전에 모든 사실과 이미지를 검토해 주세요.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("템플릿", story["template_id"])
    metric_columns[1].metric("스토리 섹션", len(story["sections"]))
    metric_columns[2].metric("생성 이미지", f"{manifest['succeeded']}/{manifest['requested']}")
    metric_columns[3].metric("자동 검증 경고", len(story["warnings"]))

    preview_tab, sections_tab, images_tab, data_tab = st.tabs(
        ["페이지 미리보기", "섹션별 원고", "생성 이미지", "JSON"]
    )
    with preview_tab:
        preview = inline_preview_images(artifacts["preview_html"], artifacts["run_dir"])
        components.html(preview, height=900, scrolling=True)
    with sections_tab:
        st.subheader(story["title_candidates"][0])
        for section in story["sections"]:
            expanded = section["template_section_id"] == "hero"
            with st.expander(section["heading"], expanded=expanded):
                st.markdown(section["body"])
                st.caption("출처 필드: " + ", ".join(section["source_fields"]))
    with images_tab:
        successful = [asset for asset in manifest["assets"] if asset["status"] == "success"]
        if not successful:
            st.info("생성된 이미지가 없습니다.")
        for asset in successful:
            image_path = artifacts["run_dir"] / "images" / asset["path"]
            caption = f"{asset['section_id']} · 검토 상태: {asset['qa_status']}"
            st.image(image_path, caption=caption)
    with data_tab:
        st.download_button(
            "story.json 다운로드",
            data=json.dumps(story, ensure_ascii=False, indent=2),
            file_name=f"{run_id}-story.json",
            mime="application/json",
        )
        st.json({"tool_result": tool_result, "story": story, "image_manifest": manifest})


st.set_page_config(
    page_title="Funding Story AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
_initialize_state()

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stChatMessage"] {border: 1px solid #e5e7eb; border-radius: 16px; padding: .25rem;}
    .demo-kicker {color:#0f766e; font-weight:800; letter-spacing:.08em; text-transform:uppercase;}
    .demo-copy {max-width:760px; color:#475569; font-size:1.05rem;}
    </style>
    <div class="demo-kicker">Local demonstration</div>
    """,
    unsafe_allow_html=True,
)
st.title("Funding Story AI")
st.markdown(
    '<p class="demo-copy">제품 설명과 기준 이미지를 바탕으로 필요한 정보를 질문하고, '
    "검토 가능한 펀딩 스토리 원고·이미지·HTML을 생성합니다.</p>",
    unsafe_allow_html=True,
)

repository = DataRepository(ROOT)
profiles = repository.load_category_profiles()
profile_options = {profile["profile_id"]: profile for profile in profiles}
conversation_started = bool(st.session_state.messages)

with st.sidebar:
    st.header("실행 설정")
    server_url = st.text_input(
        "FastMCP 서버",
        value="http://127.0.0.1:8765/mcp",
        disabled=conversation_started,
    )
    profile_id = st.selectbox(
        "제품 프로필",
        options=list(profile_options),
        format_func=lambda value: f"{profile_options[value]['category']} · {value}",
        disabled=conversation_started,
    )
    uploaded = st.file_uploader(
        "제품 기준 이미지",
        type=["png", "jpg", "jpeg", "webp"],
        disabled=conversation_started,
        key="reference_image",
    )
    if uploaded is not None and not conversation_started:
        try:
            st.session_state.reference_image_path = save_uploaded_image(
                root=ROOT / "artifacts" / "ui-uploads",
                input_id=st.session_state.input_id,
                filename=uploaded.name,
                content=uploaded.getvalue(),
            )
            st.image(uploaded, caption="이번 대화의 제품 기준 이미지")
        except ValueError as exc:
            st.error(str(exc))
    st.caption("생성은 로컬 FastMCP 서버를 통해 실행됩니다.")
    st.code("uv run funding-story server", language="bash")
    if st.button("새로 시작", use_container_width=True):
        _reset()
        st.rerun()

if not st.session_state.messages:
    st.info(
        "예: ‘Cleanforge R1은 가구 아래 반복 청소가 필요한 사용자를 위한 "
        "얇은 로봇청소기입니다. 자동 먼지 비움 도크를 제공합니다.’"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.tool_result:
    _render_result()
else:
    if st.session_state.stage == "confirmation":
        left, right = st.columns([2, 1])
        if left.button("이 정보로 스토리 생성", type="primary", use_container_width=True):
            if _run_worker(server_url=server_url, profile_id=profile_id, confirmed=True):
                st.rerun()
        if right.button("질문 건너뛰고 생성", use_container_width=True):
            if _run_worker(server_url=server_url, profile_id=profile_id, skip=True):
                st.rerun()
    elif st.session_state.stage in {
        "primary-details",
        "secondary-details",
        "combined-details",
    }:
        if st.button("남은 질문 건너뛰고 생성"):
            if _run_worker(server_url=server_url, profile_id=profile_id, skip=True):
                st.rerun()

    user_message = st.chat_input("제품과 만들고 싶은 펀딩 스토리를 설명해 주세요")
    if user_message:
        st.session_state.answer_flags = mark_stage_answered(
            st.session_state.stage,
            st.session_state.answer_flags,
        )
        st.session_state.messages.append({"role": "user", "content": user_message})
        if _run_worker(server_url=server_url, profile_id=profile_id):
            st.rerun()
