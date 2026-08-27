from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from funding_story_ai.ui_support import (
    read_run_resource,
    save_uploaded_image,
)
from funding_story_ai.worker import WorkerOutcome, WorkerRequest, build_live_worker

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
SERVER_URL = os.getenv("STORY_MCP_SERVER_URL", "http://127.0.0.1:8765/mcp")


def _initialize_state() -> None:
    conversation_id = f"ui-{uuid.uuid4()}"
    defaults: dict[str, Any] = {
        "thread_id": conversation_id,
        "input_id": conversation_id,
        "messages": [],
        "stage": None,
        "facts": {},
        "summary_version": 0,
        "tool_result": None,
        "reference_image_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset() -> None:
    thread_id = st.session_state.get("thread_id")
    if thread_id:
        _get_worker().delete_thread(thread_id)
    for key in (
        "thread_id",
        "input_id",
        "messages",
        "stage",
        "facts",
        "summary_version",
        "tool_result",
        "reference_image_path",
        "story_chat_input",
    ):
        st.session_state.pop(key, None)


@st.cache_resource
def _get_worker():
    return build_live_worker(server_url=SERVER_URL, root=ROOT)


def _worker_request(message: str) -> WorkerOutcome:
    request = WorkerRequest(
        thread_id=st.session_state.thread_id,
        input_id=st.session_state.input_id,
        message=message,
        message_id=f"ui-message-{uuid.uuid4()}",
        image_path=st.session_state.reference_image_path,
        caller_id="streamlit-demo",
    )
    return asyncio.run(_get_worker().handle(request))


def _handle_outcome(outcome: WorkerOutcome) -> None:
    st.session_state.stage = outcome.stage
    st.session_state.facts = outcome.facts
    st.session_state.summary_version = outcome.summary_version
    if outcome.status == "submitted":
        st.session_state.tool_result = outcome.tool_result
        return
    for question in outcome.questions:
        message = {"role": "assistant", "content": question}
        if not st.session_state.messages or st.session_state.messages[-1] != message:
            st.session_state.messages.append(message)


def _run_worker(message: str, *, approving: bool = False) -> bool:
    label = "생성 작업을 제출하고 있습니다…" if approving else "입력을 확인하고 있습니다…"
    try:
        with st.spinner(label):
            outcome = _worker_request(message)
        _handle_outcome(outcome)
        return True
    except Exception as exc:
        st.error(f"실행하지 못했습니다: {type(exc).__name__}: {exc}")
        return False


def _render_result() -> None:
    tool_result = st.session_state.tool_result
    if not tool_result:
        return
    result_uri = tool_result.get("result_uri")
    if not result_uri:
        st.error("생성 응답에 결과 리소스 URI가 없습니다.")
        return
    try:
        record = asyncio.run(read_run_resource(SERVER_URL, result_uri))
    except Exception as exc:
        st.error(f"FastMCP 결과 리소스를 읽지 못했습니다: {exc}")
        return
    if record["status"] == "running":
        st.info("스토리를 생성하고 있습니다. 완료되면 결과 리소스에서 표시됩니다.")
        if st.button("생성 상태 새로고침", type="primary"):
            st.rerun()
        return
    if record["status"] == "failed":
        st.error(f"생성에 실패했습니다: {record.get('error_type') or 'unknown error'}")
        return

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        st.error("완료 리소스에 표시 가능한 산출물이 없습니다.")
        return
    story = artifacts["story"]
    manifest = artifacts["manifest"]
    st.success("스토리 초안이 생성되었습니다. 게시 전에 모든 사실과 이미지를 검토해 주세요.")
    columns = st.columns(4)
    columns[0].metric("구성 양식", story["template_id"])
    columns[1].metric("스토리 영역", len(story["sections"]))
    columns[2].metric("생성 이미지", f"{manifest['succeeded']}/{manifest['requested']}")
    columns[3].metric("자동 검증 경고", len(story["warnings"]))

    preview_tab, sections_tab, images_tab, data_tab = st.tabs(
        ["페이지 미리보기", "영역별 원고", "생성 이미지", "JSON"]
    )
    with preview_tab:
        encoded = base64.b64encode(artifacts["preview_html"].encode()).decode()
        st.iframe(f"data:text/html;base64,{encoded}", height=900)
    with sections_tab:
        st.subheader(story["title_candidates"][0])
        for section in story["sections"]:
            with st.expander(
                section["heading"],
                expanded=section["template_section_id"] == "hero",
            ):
                st.markdown(section["body"])
                st.caption("출처 필드: " + ", ".join(section["source_fields"]))
    with images_tab:
        successful = [asset for asset in manifest["assets"] if asset["status"] == "success"]
        if not successful:
            st.info("생성된 이미지가 없습니다.")
        for asset in successful:
            caption = (
                f"{asset['section_id']} · 검토 상태: {asset['qa_status']} · "
                f"{asset['provider']}/{asset['model']}"
            )
            st.image(artifacts["image_data"][asset["path"]], caption=caption)
    with data_tab:
        run_id = record["run_id"]
        st.download_button(
            "story.json 다운로드",
            data=json.dumps(story, ensure_ascii=False, indent=2),
            file_name=f"{run_id}-story.json",
            mime="application/json",
        )
        st.json({"run": record["result"], "story": story, "image_manifest": manifest})


st.set_page_config(
    page_title="Funding Story AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
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
header, reset = st.columns([5, 1])
header.title("Funding Story AI")
if reset.button("새로 시작", use_container_width=True):
    _reset()
    st.rerun()
st.markdown(
    '<p class="demo-copy">제품 설명과 선택 이미지를 하나의 대화로 입력하면 필요한 정보를 '
    "질문하고, 검토 가능한 펀딩 스토리 원고·이미지·HTML을 생성합니다.</p>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.info(
        "예: ‘Cleanforge R1은 가구 아래 반복 청소가 필요한 사용자를 위한 "
        "얇은 로봇청소기입니다. 자동 먼지 비움 도크를 제공합니다.’"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image_path"):
            st.image(message["image_path"], caption="제품 참조 이미지", width=360)
        st.markdown(message["content"])

if st.session_state.tool_result:
    _render_result()
else:
    if st.session_state.stage == "awaiting-approval":
        if st.button("이 정보로 스토리 생성", type="primary", use_container_width=True):
            approval_message = "요약한 내용 그대로 스토리를 생성해줘"
            st.session_state.messages.append(
                {"role": "user", "content": approval_message, "image_path": None}
            )
            if _run_worker(approval_message, approving=True):
                st.rerun()

    submitted = st.chat_input(
        "제품과 만들고 싶은 펀딩 스토리를 설명해 주세요",
        key="story_chat_input",
        max_chars=1_000,
        max_upload_size=10,
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp"],
    )
    if submitted:
        text = submitted if isinstance(submitted, str) else submitted.text
        files = [] if isinstance(submitted, str) else list(submitted.files)
        if not text.strip():
            st.error("제품 설명을 함께 입력해 주세요.")
        else:
            image_path = None
            if files:
                if st.session_state.reference_image_path is not None:
                    st.error("한 대화에는 제품 이미지 1개만 사용할 수 있습니다.")
                    st.stop()
                uploaded = files[0]
                try:
                    image_path = save_uploaded_image(
                        root=ROOT / "artifacts" / "ui-uploads",
                        input_id=st.session_state.input_id,
                        filename=uploaded.name,
                        content=uploaded.getvalue(),
                    )
                    st.session_state.reference_image_path = image_path
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": text.strip(),
                    "image_path": str(image_path) if image_path else None,
                }
            )
            if _run_worker(text.strip()):
                st.rerun()
