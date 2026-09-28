"""AppTest smoke tests for the unified src/app.py. These exercise real Groq
(and, for test_full_pipeline_run_via_ui, real Anthropic) calls — slower than
the pure-logic tests, but they're what caught a real relative-path crash
during manual testing before, so they stay in the suite."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "app.py"


def _at():
    return AppTest.from_file(str(APP), default_timeout=180)


def test_loads_without_exception_default_mode():
    at = _at().run()
    assert not at.exception
    assert at.session_state["mode"] == "Multi-Agent Pipeline"


def test_ifc_studio_scenario_button_generates_design():
    at = _at().run()
    at.sidebar.radio[0].set_value("IFC Design Studio").run()
    assert not at.exception
    btn = next(b for b in at.sidebar.button if "Suburban" in b.label)
    btn.click().run()
    assert not at.exception
    assert at.session_state["units"] == 60
    assert at.session_state["ifc_layout"] is not None
    assert at.metric[0].value == "60"


def test_chat_panel_replies():
    at = _at().run()
    assert not at.exception
    at.chat_input[0].set_value("What does FAR mean?").run()
    assert not at.exception
    roles = [m["role"] for m in at.session_state["chat_history"]]
    assert roles == ["user", "assistant"]
    assert len(at.session_state["chat_history"][-1]["content"]) > 0


def test_open_ifc_file_explorer():
    from rngd.ifc import explorer

    if not explorer.list_available_files():
        pytest.skip("no .ifc files available under data/ifc_samples/ or outputs/ifc_design/")
    at = _at().run()
    at.sidebar.radio[0].set_value("Open IFC File").run()
    assert not at.exception
    btn = next(b for b in at.sidebar.button if "Load" in b.label)
    btn.click().run()
    assert not at.exception
    ss_ready = at.session_state["explorer_summary"]
    assert ss_ready and ss_ready["total_elements"] > 0
    assert at.session_state["explorer_meshes"]


@pytest.mark.slow
def test_full_pipeline_run_via_ui():
    at = _at().run()
    btn = next(b for b in at.sidebar.button if b.label == "Run full pipeline")
    btn.click().run()
    assert not at.exception
    state = at.session_state["pipeline_state"]
    assert state is not None and state.final_decision is not None
    assert len(state.trace) >= 10
    # a proactive suggestion should have been posted to chat
    assert any(m["role"] == "assistant" for m in at.session_state["chat_history"])
