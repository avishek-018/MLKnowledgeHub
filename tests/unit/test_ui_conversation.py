"""Headless Streamlit tests for scoped conversation rendering."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "ml_knowledge_hub"
    / "ui"
    / "app.py"
)


def _start_app() -> AppTest:
    # Skip the optional APP_PASSWORD gate a local secrets.toml may enable.
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.session_state["authenticated"] = True
    return app.run()


def test_greeting_renders_without_retrieval_panels():
    app = _start_app()

    app.chat_input(key="chat_prompt").set_value("Hello!").run()

    assert not app.exception
    assert len(app.session_state["chat_messages"]) == 2
    assert app.session_state["chat_messages"][0] == {
        "role": "user",
        "content": "Hello!",
    }
    result = app.session_state["chat_messages"][1]["result"]
    assert result["type"] == "conversation"
    assert result["operation"] == "greeting"
    assert "Sources" not in [item.value for item in app.subheader]
    assert not app.expander


def test_chat_history_persists_across_multiple_messages():
    app = _start_app()

    app.chat_input(key="chat_prompt").set_value("Hello!").run()
    app.chat_input(key="chat_prompt").set_value("Thank you").run()

    assert not app.exception
    assert len(app.session_state["chat_messages"]) == 4
    assert [
        message["role"] for message in app.session_state["chat_messages"]
    ] == ["user", "assistant", "user", "assistant"]


def test_clear_button_starts_a_new_conversation():
    app = _start_app()
    app.chat_input(key="chat_prompt").set_value("Hello!").run()

    app.button(key="clear_chat").click().run()

    assert not app.exception
    assert app.session_state["chat_messages"] == []
    assert len(app.chat_message) == 1
