from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.engine import review_integrated_story_run
from funding_story_ai.run_store import LocalRunStore
from funding_story_ai.ui_support import (
    load_run_artifacts,
    read_run_resource,
    save_uploaded_image,
)
from funding_story_ai.worker import (
    FastMcpGenerationTool,
    GroundedBriefBuilder,
    StoryGenerationDispatcher,
    WorkerOutcome,
    WorkerRequest,
    build_live_worker,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT.parent / ".env", override=False)
SERVER_URL = os.getenv("STORY_MCP_SERVER_URL", "http://127.0.0.1:8765/mcp")
DEFAULT_STORE_ROOT = ROOT / "artifacts" / "runs"

_STAGE_LABELS = {
    "collecting": "필수 정보 수집",
    "awaiting-approval": "요약 확인 및 승인",
    "generation-ready": "생성 실행 대기",
    "generating": "스토리 생성 중",
    "completed": "결과 검수",
}
_REVIEW_LABELS = {
    "scene_distinctness": "다른 이미지와 장면이 구분됨",
    "product_fidelity": "참조 제품의 외형이 유지됨",
    "text_legibility": "이미지 안 문자가 읽힘",
    "claim_grounding": "승인된 사실만 표현됨",
}
_REVIEW_STATUS_LABELS = {
    "pending": "검수 대기",
    "conditional": "조건부",
    "pass": "통과",
    "fail": "반려",
}
_REVIEW_STATUS_FROM_LABEL = {value: key for key, value in _REVIEW_STATUS_LABELS.items()}


def _store_root() -> Path:
    configured = Path(os.getenv("STORY_MCP_RUN_STORE", str(DEFAULT_STORE_ROOT)))
    return configured if configured.is_absolute() else ROOT / configured


def _initialize_state() -> None:
    conversation_id = f"ui-{uuid.uuid4()}"
    defaults: dict[str, Any] = {
        "thread_id": conversation_id,
        "input_id": conversation_id,
        "messages": [],
        "stage": "collecting",
        "facts": {},
        "summary": None,
        "summary_version": 0,
        "approved_summary_version": None,
        "generation_response": None,
        "run_record": None,
        "reference_image_path": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def _get_worker() -> Any:
    checkpoint_path = ROOT / "artifacts" / "state" / "conversations.sqlite3"
    return build_live_worker(root=ROOT, checkpoint_path=checkpoint_path)


def _reset() -> None:
    thread_id = st.session_state.get("thread_id")
    if thread_id:
        try:
            _get_worker().delete_thread(thread_id)
        except Exception:
            pass
    for key in (
        "thread_id",
        "input_id",
        "messages",
        "stage",
        "facts",
        "summary",
        "summary_version",
        "approved_summary_version",
        "generation_response",
        "run_record",
        "reference_image_path",
        "last_error",
        "story_chat_input",
    ):
        st.session_state.pop(key, None)


def _append_message(role: str, content: str, *, image_path: str | None = None) -> None:
    message = {"role": role, "content": content, "image_path": image_path}
    if not st.session_state.messages or st.session_state.messages[-1] != message:
        st.session_state.messages.append(message)


def _worker_request(message: str, *, image_path: Path | None = None) -> WorkerOutcome:
    request = WorkerRequest(
        thread_id=st.session_state.thread_id,
        input_id=st.session_state.input_id,
        message=message,
        message_id=f"ui-message-{uuid.uuid4()}",
        image_path=image_path,
        caller_id="streamlit-demo",
    )
    return asyncio.run(_get_worker().handle(request))


def _handle_outcome(outcome: WorkerOutcome) -> None:
    st.session_state.stage = outcome.stage
    st.session_state.facts = outcome.facts
    st.session_state.summary = outcome.current_summary
    st.session_state.summary_version = outcome.summary_version
    st.session_state.approved_summary_version = outcome.approved_summary_version
    st.session_state.last_error = None
    if outcome.reply:
        _append_message("assistant", outcome.reply)


def _run_worker(message: str, *, image_path: Path | None = None) -> bool:
    try:
        with st.spinner("대화 내용을 확인하고 다음 단계를 준비하고 있습니다…"):
            outcome = _worker_request(message, image_path=image_path)
        _handle_outcome(outcome)
        if outcome.temporary_error:
            st.warning(outcome.reply)
        return True
    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.error("입력을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.")
        return False


def _dispatch_generation() -> bool:
    try:
        with st.spinner("승인된 정보로 스토리와 이미지를 생성하고 있습니다…"):
            state = _get_worker().get_state(st.session_state.thread_id)
            request = WorkerRequest(
                thread_id=st.session_state.thread_id,
                input_id=st.session_state.input_id,
                message="승인된 요약으로 생성 실행",
                message_id=f"ui-dispatch-{uuid.uuid4()}",
                caller_id="streamlit-demo",
                idempotency_key=(
                    f"streamlit-{st.session_state.thread_id}-summary-"
                    f"{st.session_state.approved_summary_version}"
                ),
            )
            repository = DataRepository(ROOT)
            dispatcher = StoryGenerationDispatcher(
                repository=repository,
                brief_builder=GroundedBriefBuilder(repository=repository),
                generation_tool=FastMcpGenerationTool(SERVER_URL),
            )
            response = asyncio.run(dispatcher.submit(request=request, state=state))
        st.session_state.generation_response = response
        st.session_state.stage = "generating"
        return True
    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.error("스토리 생성을 시작하지 못했습니다. 승인된 입력을 다시 확인해 주세요.")
        return False


def _render_progress_panel() -> None:
    st.subheader("진행 상태")
    stage = st.session_state.stage or "collecting"
    labels = ["collecting", "awaiting-approval", "generation-ready", "generating", "completed"]
    current_index = labels.index(stage) if stage in labels else 0
    for index, key in enumerate(labels):
        marker = "●" if index <= current_index else "○"
        st.markdown(f"{marker} {_STAGE_LABELS[key]}")
    if st.session_state.reference_image_path:
        st.caption("제품 참조 이미지가 대화에 포함되어 있습니다.")
    if st.session_state.summary is not None:
        st.caption(f"요약 버전 {st.session_state.summary_version}")
    response = st.session_state.generation_response
    if isinstance(response, dict) and response.get("run_id"):
        st.caption("생성 결과를 불러오는 중입니다.")


def _render_conversation() -> None:
    if not st.session_state.messages:
        st.info(
            "제품명·제품 유형·카테고리·핵심 강점을 설명해 주세요. "
            "제품 참조 이미지는 아래 대화 입력에 함께 첨부할 수 있습니다."
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("image_path"):
                st.image(message["image_path"], caption="제품 참조 이미지", width=360)
            st.markdown(message["content"])


def _render_summary_and_actions() -> None:
    if st.session_state.stage == "awaiting-approval" and st.session_state.summary:
        summary = st.session_state.summary
        with st.container(border=True):
            st.subheader("생성 전 요약")
            st.markdown(summary["summary_text"])
            if st.button("이 요약으로 생성 준비", type="primary", use_container_width=True):
                approval_message = "요약한 내용 그대로 스토리를 생성해줘."
                _append_message("user", approval_message)
                if _run_worker(approval_message):
                    st.rerun()
    if st.session_state.stage == "generation-ready":
        st.success("요약 승인이 완료되었습니다. 아래 버튼을 눌러 별도 생성을 시작하세요.")
        if st.button("승인된 정보로 생성 실행", type="primary", use_container_width=True):
            if _dispatch_generation():
                st.rerun()


def _render_chat_input() -> None:
    if st.session_state.generation_response:
        return
    submitted = st.chat_input(
        "제품 설명이나 추가 정보를 입력하세요",
        key="story_chat_input",
        max_chars=1_000,
        max_upload_size=10,
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp"],
    )
    if not submitted:
        return
    text = submitted if isinstance(submitted, str) else submitted.text
    files = [] if isinstance(submitted, str) else list(submitted.files)
    if not text.strip():
        st.error("제품 설명을 함께 입력해 주세요.")
        return
    image_path: Path | None = None
    if files:
        if st.session_state.reference_image_path is not None:
            st.error("한 대화에는 제품 참조 이미지 1개만 첨부할 수 있습니다.")
            return
        uploaded = files[0]
        try:
            image_path = save_uploaded_image(
                root=ROOT / "artifacts" / "ui-uploads",
                input_id=st.session_state.input_id,
                filename=uploaded.name,
                content=uploaded.getvalue(),
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.reference_image_path = image_path
    _append_message("user", text.strip(), image_path=str(image_path) if image_path else None)
    if _run_worker(text.strip(), image_path=image_path):
        st.rerun()


def _render_review_form(record: dict[str, Any], artifacts: dict[str, Any]) -> None:
    manifest = artifacts["manifest"]
    successful = [asset for asset in manifest["assets"] if asset["status"] == "success"]
    if not successful:
        return
    st.subheader("이미지 검수")
    st.caption("각 장면을 원본 제품과 승인된 사실에 대조한 뒤 게시 가능 여부를 저장하세요.")
    with st.form(f"review-{record['run_id']}"):
        selections: dict[str, dict[str, Any]] = {}
        for asset in successful:
            st.markdown(f"**{asset['section_id']}**")
            image = artifacts["image_data"].get(asset.get("path"))
            if image:
                st.image(image, width=420)
            current_status = _REVIEW_STATUS_LABELS.get(
                asset.get("qa_status", "pending"), "검수 대기"
            )
            status_label = st.selectbox(
                "장면 검수 결과",
                list(_REVIEW_STATUS_FROM_LABEL),
                index=list(_REVIEW_STATUS_FROM_LABEL).index(current_status),
                key=f"review-status-{record['run_id']}-{asset['slot_id']}",
            )
            checks: dict[str, str] = {}
            for check, label in _REVIEW_LABELS.items():
                checked = st.checkbox(
                    label,
                    value=asset.get("review_checks", {}).get(check) == "pass",
                    key=f"review-check-{record['run_id']}-{asset['slot_id']}-{check}",
                )
                checks[check] = "pass" if checked else "pending"
            selections[asset["slot_id"]] = {
                "qa_status": _REVIEW_STATUS_FROM_LABEL[status_label],
                "review_checks": checks,
            }
        submitted = st.form_submit_button("검수 저장", type="primary")
    if not submitted:
        return
    try:
        result = review_integrated_story_run(
            repository=DataRepository(ROOT),
            store=LocalRunStore(_store_root()),
            run_id=record["run_id"],
            reviews=selections,
        )
        st.session_state.run_record = result["run"]
        if result["publishable"]:
            st.success("모든 이미지 검수를 통과했습니다. 게시 가능 HTML을 확인할 수 있습니다.")
        else:
            st.success("검수 결과를 저장했습니다. 모든 필수 항목을 통과해야 게시할 수 있습니다.")
        st.rerun()
    except Exception:
        st.error("검수 결과를 저장하지 못했습니다. 각 장면의 상태와 항목을 확인해 주세요.")


def _artifacts_for_record(record: dict[str, Any]) -> dict[str, Any]:
    remote_artifacts = record.get("artifacts")
    if isinstance(remote_artifacts, dict) and {
        "story",
        "manifest",
        "draft_html",
        "image_data",
        "source_files",
    }.issubset(remote_artifacts):
        return remote_artifacts
    return load_run_artifacts(_store_root(), record)


def _render_run_result() -> None:
    response = st.session_state.generation_response
    if not isinstance(response, dict):
        return
    result_uri = response.get("result_uri")
    if not result_uri:
        st.error("생성 결과를 확인할 수 없습니다.")
        return
    try:
        record = asyncio.run(read_run_resource(SERVER_URL, result_uri))
    except Exception:
        st.info("생성 결과를 아직 불러오지 못했습니다. 잠시 후 새로고침해 주세요.")
        if st.button("생성 상태 새로고침", type="primary"):
            st.rerun()
        return
    st.session_state.run_record = record
    if record["status"] == "running":
        st.info("스토리와 이미지를 생성하고 있습니다. 완료되면 결과가 표시됩니다.")
        if st.button("생성 상태 새로고침", type="primary"):
            st.rerun()
        return
    if record["status"] == "failed":
        st.error("생성에 실패했습니다. 입력을 확인한 뒤 새 대화에서 다시 시도해 주세요.")
        return
    try:
        artifacts = _artifacts_for_record(record)
    except Exception:
        st.error("완료된 산출물을 읽지 못했습니다. STORY_MCP_RUN_STORE 경로를 확인해 주세요.")
        return
    st.session_state.stage = "completed"
    result = record.get("result", {})
    if result.get("warning_count"):
        st.warning("원고에 추가 검토가 필요한 표현이 있습니다. 게시 전에 원고를 확인해 주세요.")
    if result.get("publishable_html"):
        st.success("게시 가능 HTML이 준비되었습니다.")
    else:
        st.info("초안 HTML을 확인하고 이미지 검수를 완료하면 게시 가능 HTML을 만들 수 있습니다.")

    draft_tab, publishable_tab, source_tab = st.tabs(
        ["완성 페이지 초안", "게시 가능 페이지", "원본 및 산출물"]
    )
    with draft_tab:
        st.html(artifacts["draft_html"])
    with publishable_tab:
        if artifacts["publishable_html"]:
            st.html(artifacts["publishable_html"])
        else:
            st.info("모든 필수 이미지 검수와 원고 검토를 통과하면 이 페이지가 준비됩니다.")
    with source_tab:
        for filename, content in artifacts["source_files"].items():
            st.download_button(
                f"{filename} 다운로드",
                data=content,
                file_name=f"{record['run_id']}-{Path(filename).name}",
                mime="text/html" if filename.endswith(".html") else "application/json",
                key=f"download-{record['run_id']}-{filename}",
            )
        with st.expander("원본 HTML 확인"):
            selected = (
                "publishable.html"
                if "publishable.html" in artifacts["source_files"]
                else "draft.html"
            )
            st.code(artifacts["source_files"][selected], language="html")
    _render_review_form(record, artifacts)


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
    .block-container {max-width: 1220px; padding-top: 2rem;}
    [data-testid="stChatMessage"] {border: 1px solid #e5e7eb; border-radius: 16px; padding: .25rem;}
    .demo-kicker {color:#0f766e; font-weight:800; letter-spacing:.08em; text-transform:uppercase;}
    .demo-copy {max-width:760px; color:#475569; font-size:1.05rem;}
    </style>
    <div class="demo-kicker">Review-first creation</div>
    """,
    unsafe_allow_html=True,
)
header, reset = st.columns([5, 1])
header.title("Funding Story AI")
if reset.button("새로 시작", use_container_width=True):
    _reset()
    st.rerun()
st.markdown(
    '<p class="demo-copy">제품 설명과 참조 이미지를 대화로 입력하면 필요한 정보를 묻고, '
    "요약 승인 후 검토 가능한 펀딩 스토리와 HTML을 생성합니다.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 1], gap="large")
with left:
    _render_conversation()
    if not st.session_state.generation_response:
        _render_summary_and_actions()
        _render_chat_input()
    else:
        _render_run_result()
with right:
    _render_progress_panel()
