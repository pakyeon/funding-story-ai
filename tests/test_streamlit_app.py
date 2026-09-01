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
    app.session_state["summary"] = {
        "summary_text": "테스트 요약",
    }
    app.run(timeout=20)
    assert "이 요약으로 생성 준비" in [button.label for button in app.button]


def test_streamlit_app_keeps_generation_as_a_separate_action() -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app = streamlit_testing.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
    ).run(timeout=20)
    app.session_state["stage"] = "generation-ready"
    app.run(timeout=20)
    labels = [button.label for button in app.button]
    assert "승인된 정보로 생성 실행" in labels
    assert "이 요약으로 생성 준비" not in labels
