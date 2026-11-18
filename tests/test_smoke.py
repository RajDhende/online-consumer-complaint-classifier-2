"""Smoke tests: every page must render and a prediction must run end to end."""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


PAGES = [
    "🏠  Home Dashboard",
    "🔎  Complaint Classification",
    "📈  Complaint Analytics",
    "🗃️  Dataset Explorer",
    "🎯  Model Performance",
    "ℹ️  About Project",
]


@pytest.fixture(scope="module")
def app() -> AppTest:
    instance = AppTest.from_file(str(APP_PATH), default_timeout=120)
    instance.run()
    assert not instance.exception, instance.exception
    return instance


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app: AppTest, page: str) -> None:
    app.radio[0].set_value(page).run()
    assert not app.exception, f"{page}: {app.exception}"


def test_prediction_end_to_end(app: AppTest) -> None:
    app.radio[0].set_value("🔎  Complaint Classification").run()
    app.text_area[0].set_value(
        "I disputed an incorrect collection account on my credit report, but the "
        "bureau verified it without investigating and it is lowering my score."
    ).run()

    predict = [button for button in app.button if "Predict" in button.label]
    assert predict, "Predict button not found"
    predict[0].click().run()

    assert not app.exception, app.exception
    prediction = app.session_state["prediction"]
    assert prediction is not None
    assert prediction["category"]
    assert 0.0 <= prediction["confidence"] <= 1.0


def test_batch_classifier() -> None:
    import joblib

    import config
    import preprocessing
    from app import classify_frame

    artifacts = {
        "model": joblib.load(config.MODEL_PATH),
        "vectorizer": joblib.load(config.VECTORIZER_PATH),
        "metadata": {"best_model": "test"},
    }
    frame = pd.DataFrame(
        {
            "narrative": [
                "My mortgage servicer applied my payment to the wrong account and reported me late.",
                "A debt collector keeps calling me about a debt I already paid.",
            ]
        }
    )
    result = classify_frame(frame, "narrative", artifacts)
    assert result["predicted_category"].notna().all()
    assert result["confidence"].between(0, 1).all()
    assert (
        preprocessing.clean_text("Hello, WORLD! Visit https://example.com")
        == "hello world visit"
    )
