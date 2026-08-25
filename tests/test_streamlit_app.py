from pathlib import Path

import pytest


def test_streamlit_app_renders_initial_screen_without_runtime_errors() -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app = streamlit_testing.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
    ).run(timeout=20)
    assert not app.exception
    assert [title.value for title in app.title] == ["Funding Story AI"]
