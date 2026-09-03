import pytest

from funding_story_ai.config import RuntimeSettings


def test_settings_require_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        RuntimeSettings.from_env()


def test_settings_allow_missing_project_for_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert RuntimeSettings.from_env(require_project=False).project_id == ""


def test_settings_read_model_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_PRIMARY_MODEL", "custom-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "custom-fallback")
    settings = RuntimeSettings.from_env()
    assert settings.primary_model == "custom-primary"
    assert settings.fallback_model == "custom-fallback"


def test_settings_use_default_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("GEMINI_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    settings = RuntimeSettings.from_env()
    assert settings.primary_model == "gemini-3.8-flash"
    assert settings.fallback_model == "gemini-3.6-flash"
    assert settings.thinking_level == "MEDIUM"
    assert settings.max_output_tokens == 24576


@pytest.mark.parametrize("level", ["LOW", "MEDIUM", "HIGH"])
def test_settings_accept_supported_thinking_levels(
    monkeypatch: pytest.MonkeyPatch,
    level: str,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", level.lower())
    assert RuntimeSettings.from_env().thinking_level == level


def test_settings_reject_minimal_for_gemini_38_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "MINIMAL")
    with pytest.raises(ValueError, match="LOW, MEDIUM, or HIGH"):
        RuntimeSettings.from_env()
