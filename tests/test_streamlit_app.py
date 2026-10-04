import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None, reason="streamlit not installed (--extra web)"
)

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_streamlit_app_end_to_end(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from stringart.models import models_dir

    # Use the repo's already-downloaded face models instead of fetching into a temp dir.
    monkeypatch.setenv("STRINGART_MODELS", os.fspath(models_dir()))
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception
    assert "Pick a sample" in at.info[0].value

    sliders = {s.label: s for s in at.sidebar.slider}
    sliders["Pins"].set_value(128)
    sliders["Thread colours"].set_value(2)
    at.sidebar.button[0].click().run()
    assert not at.exception
    res = at.session_state["result"]
    assert res["kit"][:2] == b"PK" and res["gif"][:3] == b"GIF"
    assert res["render"].shape[:2] == (600, 600)
    assert len(at.get("download_button")) == 3
