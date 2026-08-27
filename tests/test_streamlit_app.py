from pathlib import Path

import pytest


def test_streamlit_app_renders_initial_screen_without_runtime_errors() -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app = streamlit_testing.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
    ).run(timeout=20)
    assert not app.exception
    assert [title.value for title in app.title] == ["Funding Story AI"]


def test_streamlit_app_exposes_only_explicit_approval_action() -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app = streamlit_testing.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
    ).run(timeout=20)
    app.session_state["stage"] = "collecting"
    app.run(timeout=20)
    assert "남은 질문을 건너뛰고 생성" not in [button.label for button in app.button]

    app.session_state["stage"] = "awaiting-approval"
    app.run(timeout=20)
    assert "이 정보로 스토리 생성" in [button.label for button in app.button]
