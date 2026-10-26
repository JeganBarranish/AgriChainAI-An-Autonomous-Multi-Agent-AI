"""Starts app.py headlessly with Streamlit's AppTest and clicks through the UI."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_runs_and_flows():
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=240).run()
    assert not at.exception
    assert any("Ready" in m.value for m in at.sidebar.markdown)
    at.button[1].click().run()                      # Check Paneer Spoilage
    assert not at.exception
    for v in ("8", "4.2", "37", "skip", "skip"):
        at.chat_input[0].set_value(v).run()
    assert not at.exception
    assert any("Spoilage Agent" in str(m.value) for m in at.markdown)
