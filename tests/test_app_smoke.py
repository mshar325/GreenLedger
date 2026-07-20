"""End-to-end app smoke test via Streamlit's AppTest -- executes the ENTIRE script,
every tab, and the form-submit + simulator paths. This is the test that would have caught
the `MOSS_SOFT` NameError that a plain `curl` HTTP-200 boot check missed: curl gets a
response from the server without triggering full script execution, so a name error inside
a tab body slips through. AppTest actually runs the script.

Skips if the exported artifacts aren't present (CI without a trained model)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "model_meta.json").exists()
    or not (ROOT / "app_data" / "dashboard_data.csv.gz").exists(),
    reason="app artifacts not exported (run export_artifacts.py + analysis scripts)")


def _apptest():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def test_all_tabs_render_without_exception():
    at = _apptest()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.tabs) == 5


def test_form_submit_and_simulator_paths():
    at = _apptest()
    at.run()
    at.button[0].click()          # "Assess risk"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    toggles = list(at.toggle)
    if toggles:
        toggles[0].set_value(True)  # "Simulate retrofits"
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
