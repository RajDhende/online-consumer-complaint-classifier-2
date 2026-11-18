"""
Online Consumer Complaint Classification System
===============================================

A Streamlit dashboard for exploring real CFPB consumer complaints and
classifying new complaint narratives with TF-IDF + machine learning.

Run locally:
    uv run streamlit run app.py

The six pages are:
    1. Home Dashboard
    2. Complaint Classification
    3. Complaint Analytics
    4. Dataset Explorer
    5. Model Performance
    6. About Project
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics
import config
import data_loader
import preprocessing

# --------------------------------------------------------------------------- #
# Page setup and visual theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Consumer Complaint Intelligence",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #0F172A;
    --muted: #64748B;
    --line: #E2E8F0;
    --brand: #4F46E5;
    --brand-2: #06B6D4;
    --surface: #FFFFFF;
}

html, body {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp, .stMarkdown, .stText, .stCaption,
[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"],
[data-testid="stText"], [data-testid="stCaptionContainer"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stButton button p, .stDownloadButton button p, textarea, input, select {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

.stApp {
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(79,70,229,.10), transparent 60%),
        radial-gradient(900px 420px at 5% 0%, rgba(6,182,212,.08), transparent 55%),
        #F6F7FB;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3.5rem;
    max-width: 1420px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #141A38 55%, #1E1B4B 100%);
    border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] * { color: #E2E8F0; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: .35rem; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: .55rem .7rem;
    border-radius: 10px;
    transition: background .15s ease;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,.06);
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.10); }
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.20);
    color: #E2E8F0;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,.15);
    border-color: rgba(255,255,255,.34);
    color: #FFFFFF;
}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 30px 34px;
    background:
        radial-gradient(700px 260px at 92% 0%, rgba(6,182,212,.35), transparent 60%),
        linear-gradient(120deg, #1E1B4B 0%, #312E81 52%, #4338CA 100%);
    box-shadow: 0 18px 40px -24px rgba(49,46,129,.75);
    margin-bottom: 1.3rem;
}
.hero::after {
    content: "";
    position: absolute;
    right: -70px; top: -90px;
    width: 280px; height: 280px;
    border-radius: 50%;
    border: 34px solid rgba(255,255,255,.06);
}
.hero-kicker {
    display: inline-block;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #A5B4FC;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.14);
    padding: .34rem .7rem;
    border-radius: 999px;
    margin-bottom: .8rem;
}
.hero-title {
    font-size: clamp(1.65rem, 2.6vw, 2.5rem);
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -.02em;
    color: #FFFFFF;
    margin: 0 0 .55rem 0;
}
.hero-sub {
    color: #C7D2FE;
    font-size: .98rem;
    line-height: 1.55;
    max-width: 820px;
    margin: 0;
}
.hero-meta {
    display: flex; flex-wrap: wrap; gap: .5rem;
    margin-top: 1.1rem;
}
.hero-chip {
    font-size: .74rem;
    font-weight: 600;
    color: #E0E7FF;
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.14);
    padding: .3rem .65rem;
    border-radius: 999px;
}

/* ---------- Section headings ---------- */
.section-title {
    font-size: 1.06rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -.01em;
    margin: .1rem 0 .1rem 0;
}
.section-sub {
    font-size: .82rem;
    color: var(--muted);
    margin: 0 0 .6rem 0;
}

/* ---------- KPI cards ---------- */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 18px 15px 18px;
    box-shadow: 0 1px 2px rgba(15,23,42,.05);
    position: relative;
    overflow: hidden;
    min-height: 118px;
}
.kpi-card::before {
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--accent, var(--brand));
}
.kpi-top {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: .45rem;
}
.kpi-label {
    font-size: .74rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted);
}
.kpi-icon { font-size: 1.05rem; }
.kpi-value {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -.02em;
    color: var(--ink);
    line-height: 1.12;
}
.kpi-value.small { font-size: 1.16rem; }
.kpi-value.tiny { font-size: 0.92rem; }
.kpi-hint {
    font-size: .76rem;
    color: var(--muted);
    margin-top: .28rem;
    line-height: 1.35;
}

/* ---------- Prediction result ---------- */
.result-card {
    background: linear-gradient(160deg, #FFFFFF 0%, #F5F3FF 100%);
    border: 1px solid #DDD6FE;
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 10px 30px -22px rgba(79,70,229,.65);
}
.result-kicker {
    font-size: .72rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #6D28D9;
}
.result-title {
    font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em;
    color: #1E1B4B; margin: .3rem 0 .1rem 0;
}
.result-official {
    font-size: .82rem; color: #6B7280; margin-bottom: 1rem;
}
.conf-row {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: .8rem; font-weight: 600; color: #4C1D95; margin-bottom: .35rem;
}
.conf-track {
    height: 10px; border-radius: 999px; background: #EDE9FE; overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6366F1, #06B6D4);
}

/* ---------- Insight panels ---------- */
.insight {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-left: 4px solid var(--brand-2);
    border-radius: 12px;
    padding: 13px 15px;
    margin-bottom: .65rem;
}
.insight-title {
    font-size: .78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: #0E7490; margin-bottom: .22rem;
}
.insight-body { font-size: .87rem; color: #334155; line-height: 1.5; }

/* ---------- Pills ---------- */
.pill-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 .6rem 0; }
.pill {
    font-size: .74rem; font-weight: 600;
    color: #3730A3; background: #EEF2FF;
    border: 1px solid #C7D2FE;
    padding: .26rem .6rem; border-radius: 999px;
}

/* ---------- Architecture flow ---------- */
.flow-row {
    display: flex; align-items: stretch; gap: .45rem;
    flex-wrap: wrap; margin: .4rem 0 .8rem 0;
}
.flow-box {
    flex: 1 1 150px;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-top: 3px solid var(--accent, var(--brand));
    border-radius: 12px;
    padding: 12px 13px;
    min-width: 140px;
}
.flow-step {
    font-size: .66rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted);
}
.flow-name { font-size: .88rem; font-weight: 700; color: var(--ink); margin: .18rem 0; }
.flow-desc { font-size: .76rem; color: var(--muted); line-height: 1.4; }
.flow-arrow { align-self: center; color: #94A3B8; font-size: 1.1rem; }

.footer {
    margin-top: 2.2rem; padding-top: 1rem;
    border-top: 1px solid var(--line);
    color: var(--muted); font-size: .76rem; line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_dataset_cached(path_str: str, modified_at: float) -> pd.DataFrame:
    """Load the project dataset; the mtime keeps the cache honest."""
    return data_loader.load_dataset(Path(path_str))


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict | None:
    """Load the trained model and every evaluation artefact once per process."""
    required = [config.MODEL_PATH, config.VECTORIZER_PATH, config.METADATA_PATH]
    if not all(path.exists() for path in required):
        return None

    artifacts: dict = {
        "model": joblib.load(config.MODEL_PATH),
        "vectorizer": joblib.load(config.VECTORIZER_PATH),
        "metadata": json.loads(config.METADATA_PATH.read_text(encoding="utf-8")),
    }

    optional = {
        "metrics": config.METRICS_PATH,
        "comparison": config.COMPARISON_PATH,
        "confusion": config.CONFUSION_PATH,
        "report": config.REPORT_PATH,
        "class_distribution": config.CLASS_DISTRIBUTION_PATH,
        "top_features": config.MODEL_DIR / "top_features.json",
        "sample_predictions": config.MODEL_DIR / "sample_predictions.csv",
    }
    for key, path in optional.items():
        if not path.exists():
            artifacts[key] = None
            continue
        if path.suffix == ".json":
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".csv":
            if key == "confusion":
                artifacts[key] = pd.read_csv(path, index_col=0)
            elif key == "report":
                artifacts[key] = pd.read_csv(path, index_col=0)
            else:
                artifacts[key] = pd.read_csv(path)
        else:
            artifacts[key] = None

    return artifacts


def get_dataset() -> pd.DataFrame | None:
    """Return the dataset or render a friendly setup message."""
    if not config.DATASET_PATH.exists() and not config.DATASET_GZIP_PATH.exists():
        st.error(
            "The dataset is missing. Run `uv run python fetch_data.py` "
            "(or `python fetch_data.py`) in the project folder, then refresh this page."
        )
        return None
    dataset_path = (
        config.DATASET_PATH
        if config.DATASET_PATH.exists()
        else config.DATASET_GZIP_PATH
    )
    try:
        return load_dataset_cached(str(dataset_path), dataset_path.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the dataset: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Small presentation helpers
# --------------------------------------------------------------------------- #
def esc(value: object) -> str:
    return html.escape(str(value))


def fmt(number: float | int | None, decimals: int = 0) -> str:
    if number is None:
        return "n/a"
    if isinstance(number, float) and pd.isna(number):
        return "n/a"
    if decimals:
        return f"{number:,.{decimals}f}"
    return f"{int(number):,}"


def pct(value: float | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def hero(
    kicker: str, title: str, subtitle: str, chips: list[str] | None = None
) -> None:
    chip_html = ""
    if chips:
        chip_html = (
            '<div class="hero-meta">'
            + "".join(f'<span class="hero-chip">{esc(chip)}</span>' for chip in chips)
            + "</div>"
        )
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{esc(kicker)}</div>
            <div class="hero-title">{esc(title)}</div>
            <p class="hero-sub">{esc(subtitle)}</p>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    sub = f'<div class="section-sub">{esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-title">{esc(title)}</div>{sub}',
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str, value: str, hint: str = "", icon: str = "📊", accent: str = "#4F46E5"
) -> None:
    value_class = (
        "kpi-value tiny"
        if len(value) > 17
        else "kpi-value small"
        if len(value) > 13
        else "kpi-value"
    )
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent}">
            <div class="kpi-top">
                <span class="kpi-label">{esc(label)}</span>
                <span class="kpi-icon">{icon}</span>
            </div>
            <div class="{value_class}">{esc(value)}</div>
            <div class="kpi-hint">{esc(hint)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(title: str, body: str) -> None:
    st.markdown(
        f'<div class="insight"><div class="insight-title">{esc(title)}</div>'
        f'<div class="insight-body">{esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


def chart_card(title: str, subtitle: str = "") -> st.delta_generator.DeltaGenerator:
    """Open a bordered card that holds a chart."""
    container = st.container(border=True)
    with container:
        section(title, subtitle)
    return container


def render_chart(fig: go.Figure, height: int | None = None) -> None:
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(df: pd.DataFrame | None, artifacts: dict | None) -> str:
    logo = config.ASSETS_DIR / "logo.png"
    with st.sidebar:
        if logo.exists():
            st.image(str(logo), width=64)
        st.markdown(
            "<div style='font-size:1.02rem;font-weight:800;letter-spacing:-.01em;"
            "line-height:1.2;margin-top:.35rem;'>Consumer Complaint<br>Intelligence</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:.74rem;color:#94A3B8;margin:.3rem 0 .9rem 0;'>"
            "Web Data Mining mini-project</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "🏠  Home Dashboard",
                "🔎  Complaint Classification",
                "📈  Complaint Analytics",
                "🗃️  Dataset Explorer",
                "🎯  Model Performance",
                "ℹ️  About Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        if df is not None:
            st.markdown(
                f"<div style='font-size:.72rem;color:#94A3B8;line-height:1.7;'>"
                f"DATASET<br><b style='color:#E2E8F0;'>{len(df):,}</b> complaints<br>"
                f"<b style='color:#E2E8F0;'>{df[config.TARGET_COLUMN].nunique()}</b> categories"
                f"</div>",
                unsafe_allow_html=True,
            )
        if artifacts:
            st.markdown(
                f"<div style='font-size:.72rem;color:#94A3B8;line-height:1.7;margin-top:.6rem;'>"
                f"MODEL<br><b style='color:#E2E8F0;'>{esc(artifacts['metadata']['best_model'])}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:.72rem;color:#FCA5A5;line-height:1.6;margin-top:.6rem;'>"
                "Model artefacts not found.<br>Run <code>train_model.py</code>.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("↻  Reload model & data", width="stretch"):
            load_artifacts.clear()
            load_dataset_cached.clear()
            st.rerun()

        st.markdown(
            "<div style='font-size:.68rem;color:#64748B;line-height:1.5;margin-top:1.2rem;'>"
            "Source: CFPB Consumer Complaint Database.<br>"
            "For academic use. Predictions are model estimates, not legal advice.</div>",
            unsafe_allow_html=True,
        )
    return page


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def predict_one(text: str, artifacts: dict) -> dict | None:
    """Clean, vectorise and classify a single complaint."""
    cleaned = preprocessing.clean_text(text)
    if not cleaned:
        return None

    matrix = artifacts["vectorizer"].transform([cleaned])
    model = artifacts["model"]
    prediction = str(model.predict(matrix)[0])

    probabilities: list[tuple[str, float]] = []
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(matrix)[0]
        probabilities = sorted(
            [
                (str(label), float(value))
                for label, value in zip(model.classes_, values)
            ],
            key=lambda item: item[1],
            reverse=True,
        )

    confidence = probabilities[0][1] if probabilities else None
    return {
        "category": prediction,
        "display": config.display_name(prediction),
        "confidence": confidence,
        "probabilities": probabilities,
        "cleaned": cleaned,
        "model": artifacts["metadata"]["best_model"],
    }


def classify_frame(
    frame: pd.DataFrame,
    text_column: str,
    artifacts: dict,
    max_rows: int = 10_000,
) -> pd.DataFrame:
    """Batch-classify a dataframe and add category/confidence columns."""
    work = frame.head(max_rows).copy()
    raw_texts = work[text_column].fillna("").astype(str).tolist()
    cleaned = [preprocessing.clean_text(text) for text in raw_texts]

    valid_mask = [bool(text) for text in cleaned]
    predictions: list[str | None] = [None] * len(cleaned)
    confidences: list[float | None] = [None] * len(cleaned)

    valid_indices = [index for index, is_valid in enumerate(valid_mask) if is_valid]
    if valid_indices:
        matrix = artifacts["vectorizer"].transform(
            [cleaned[index] for index in valid_indices]
        )
        model = artifacts["model"]
        predicted = model.predict(matrix)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(matrix)
            confidence_values = probabilities.max(axis=1)
        else:
            confidence_values = [None] * len(predicted)

        for cursor, index in enumerate(valid_indices):
            predictions[index] = str(predicted[cursor])
            value = confidence_values[cursor]
            confidences[index] = float(value) if value is not None else None

    work["predicted_category"] = predictions
    work["predicted_display"] = [
        config.display_name(value) if isinstance(value, str) else None
        for value in predictions
    ]
    work["confidence"] = confidences
    return work


SAMPLE_COMPLAINTS = {
    "Credit report item": (
        "I disputed an incorrect collection account on my credit report months ago, "
        "but the credit bureau keeps verifying it without a real investigation. The "
        "account is not mine, and it is lowering my credit score and blocking a mortgage "
        "application."
    ),
    "Collector harassment": (
        "A debt collector keeps calling me several times a day about a debt I already "
        "paid. They refuse to send written validation of the debt and continue to call "
        "my family and my workplace even after I asked them to stop."
    ),
    "Mortgage misapplied": (
        "My mortgage servicer applied my monthly payment to the wrong account and now "
        "reports me as late to the credit bureaus. I have bank records proving the "
        "payment was on time, but customer service has not corrected the error."
    ),
    "Bank overdraft fees": (
        "My bank charged me three overdraft fees for a single debit card transaction "
        "and refused to reverse them. My account balance was positive when the "
        "transaction started, and the fees pushed me into a negative balance."
    ),
    "Fraudulent new card": (
        "Someone opened a credit card in my name and the issuer is demanding payment. "
        "I never authorized this account. I filed a fraud report and sent an identity "
        "theft affidavit, but the bank closed my case without removing the balance."
    ),
    "Wire not reversed": (
        "I sent a wire transfer to the wrong account and my bank has not reversed it "
        "after two weeks. Customer service keeps transferring me between departments "
        "and no one will provide a trace number or a written explanation."
    ),
    "Refund refused": (
        "I ordered a product online, but the company delivered a damaged item and "
        "refused to refund my money. I paid with my credit card and the dispute was "
        "closed without a proper explanation."
    ),
}


# --------------------------------------------------------------------------- #
# Page 1 - Home dashboard
# --------------------------------------------------------------------------- #
def page_home(df: pd.DataFrame) -> None:
    hero(
        "Web Data Mining · Consumer Finance",
        "Consumer Complaint Intelligence",
        "A machine-learning dashboard that reads real consumer complaint narratives, "
        "classifies them into financial product categories, and surfaces the patterns "
        "behind thousands of complaints.",
        [
            "CFPB public data",
            f"{len(df):,} narratives",
            f"{df[config.TARGET_COLUMN].nunique()} categories",
            "TF-IDF + supervised learning",
        ],
    )

    stats = analytics.kpis(df)
    counts = analytics.category_counts(df)
    companies = analytics.company_counts(df, 1)
    word_stats = analytics.narrative_word_stats(df)

    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    with col1:
        kpi_card(
            "Total complaints",
            fmt(stats["total_complaints"]),
            "Narrative-bearing records in the sample",
            "🗂️",
            "#4F46E5",
        )
    with col2:
        kpi_card(
            "Complaint categories",
            fmt(stats["total_categories"]),
            "Distinct CFPB product families",
            "🏷️",
            "#06B6D4",
        )
    with col3:
        kpi_card(
            "Top category",
            config.display_name(stats["top_category"])
            if stats["top_category"] != "n/a"
            else "n/a",
            f"{pct(stats['top_category_share'])} of all complaints",
            "🥇",
            "#F59E0B",
        )
    with col4:
        kpi_card(
            "Companies involved",
            fmt(stats["total_companies"]),
            "Unique companies named in the sample",
            "🏢",
            "#10B981",
        )
    with col5:
        kpi_card(
            "Timely responses",
            pct(stats["timely_rate"]),
            "Share of complaints answered on time",
            "⏱️",
            "#8B5CF6",
        )

    st.write("")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        with chart_card(
            "Complaint distribution by category",
            "Which financial products generate the most complaints?",
        ):
            render_chart(analytics.category_bar(counts, height=380))
    with right:
        with chart_card(
            "Share of complaints",
            "The same distribution as a proportion of the sample.",
        ):
            render_chart(analytics.category_pie(counts, height=380))

    with chart_card(
        "Complaint volume over time",
        "Monthly volume in this random CFPB sample (2015-2024).",
    ):
        monthly = analytics.monthly_counts(df)
        render_chart(analytics.monthly_line(monthly, height=340))

    st.write("")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        with chart_card(
            "Leading categories over time", "Top six product families, monthly."
        ):
            render_chart(
                analytics.category_month_line(
                    analytics.category_by_month(df), height=380
                )
            )
    with right:
        with st.container(border=True):
            section(
                "What the data says",
                "Automatically generated insights from the current sample.",
            )
            if not counts.empty:
                top = counts.iloc[0]
                insight(
                    "Dominant category",
                    f"{config.display_name(top['product'])} accounts for "
                    f"{pct(top['share'])} of the sample ({int(top['complaints']):,} complaints).",
                )
            if not companies.empty:
                insight(
                    "Most mentioned company",
                    f"{companies.iloc[0]['company']} appears in "
                    f"{int(companies.iloc[0]['complaints']):,} complaints.",
                )
            if not word_stats.empty:
                longest = word_stats.iloc[0]
                insight(
                    "Richest narratives",
                    f"{config.display_name(longest['product'])} complaints are the longest, "
                    f"averaging {longest['avg_words']:.0f} words each.",
                )
            insight(
                "Response behaviour",
                f"{pct(stats['timely_rate'])} of complaints in the sample received a timely "
                f"company response; the average narrative is {stats['avg_words']:.0f} words long.",
            )
            insight(
                "Why it matters",
                "Automatic routing lets a complaint team send each case to the right "
                "specialist in seconds instead of reading every narrative by hand.",
            )

    st.write("")
    with st.container(border=True):
        section(
            "Recent complaints",
            "The newest records in the sample, with narrative excerpts.",
        )
        recent = df.sort_values("date_received", ascending=False).head(6).copy()
        recent["date"] = pd.to_datetime(
            recent["date_received"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        recent["category"] = recent[config.TARGET_COLUMN].map(config.display_name)
        recent["excerpt"] = (
            recent[config.TEXT_COLUMN].astype(str).str.slice(0, 190) + "…"
        )
        recent["company_short"] = recent["company"].astype(str).str.slice(0, 38)
        display = recent[["date", "category", "company_short", "excerpt"]].rename(
            columns={
                "date": "Date",
                "category": "Category",
                "company_short": "Company",
                "excerpt": "Narrative excerpt",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True, height=300)

    st.markdown(
        '<div class="footer">The sample is a random subset of the public CFPB database, '
        "so absolute counts describe this sample, not the full population of US consumers. "
        "The full database and field definitions are published at "
        f'<a href="{config.CFPB_HOME}" target="_blank">consumerfinance.gov</a>.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Page 2 - Complaint classification
# --------------------------------------------------------------------------- #
def page_classify(artifacts: dict | None) -> None:
    hero(
        "Live prediction",
        "Classify a consumer complaint",
        "Paste a complaint narrative. The text is cleaned, converted to TF-IDF features, "
        "and passed to the trained classifier. You get a category, a confidence score, "
        "and the model's top alternatives.",
        ["TF-IDF", "Logistic Regression vs Naive Bayes", "Explainable output"],
    )

    if not artifacts:
        st.warning(
            "No trained model was found. Run `uv run python train_model.py` in the project "
            "folder, then press **Reload model & data** in the sidebar."
        )
        return

    if "complaint_text" not in st.session_state:
        st.session_state.complaint_text = ""
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    def load_sample(value: str) -> None:
        st.session_state.complaint_text = value
        st.session_state.prediction = None

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        with st.container(border=True):
            section("Complaint narrative", "Try one of the examples or write your own.")
            sample_cols = st.columns(2)
            for index, (label, text) in enumerate(SAMPLE_COMPLAINTS.items()):
                with sample_cols[index % 2]:
                    st.button(
                        label,
                        key=f"sample_{index}",
                        width="stretch",
                        on_click=load_sample,
                        args=(text,),
                    )

            text = st.text_area(
                "Complaint text",
                key="complaint_text",
                height=190,
                placeholder=(
                    "Example: I disputed an incorrect account on my credit report, "
                    "but the bureau verified it without investigating…"
                ),
                label_visibility="collapsed",
            )
            words = len(str(text).split())
            st.caption(f"{words} words · {len(str(text))} characters")

            button_col, clear_col = st.columns([1, 1])
            with button_col:
                predict = st.button(
                    "⚡  Predict category", type="primary", width="stretch"
                )
            with clear_col:
                if st.button("Clear", width="stretch"):
                    st.session_state.complaint_text = ""
                    st.session_state.prediction = None
                    st.rerun()

            if predict:
                if words < 4:
                    st.warning("Please enter a longer complaint (at least four words).")
                else:
                    with st.spinner("Cleaning text and running the classifier…"):
                        result = predict_one(text, artifacts)
                    if result is None:
                        st.error(
                            "The text contained no usable words after cleaning. "
                            "Try a longer, more descriptive complaint."
                        )
                    else:
                        st.session_state.prediction = result

    with right:
        result = st.session_state.prediction
        if not result:
            with st.container(border=True):
                section("Prediction", "The result appears here.")
                st.markdown(
                    "<div style='padding:1.6rem .4rem;color:#64748B;font-size:.9rem;line-height:1.6;'>"
                    "Enter a complaint and press <b>Predict category</b>. The dashboard will show "
                    "the predicted product family, a confidence score, and the strongest "
                    "alternatives.</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="pill-row">'
                    '<span class="pill">Text cleaning</span>'
                    '<span class="pill">TF-IDF</span>'
                    '<span class="pill">Multi-class model</span>'
                    '<span class="pill">Probability</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            confidence = result["confidence"]
            bar_width = f"{max(2.0, (confidence or 0) * 100):.1f}%"
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-kicker">Predicted category</div>
                    <div class="result-title">{esc(result["display"])}</div>
                    <div class="result-official">Official CFPB product label: {esc(result["category"])}</div>
                    <div class="conf-row">
                        <span>Model confidence</span>
                        <span>{pct(confidence)}</span>
                    </div>
                    <div class="conf-track"><div class="conf-fill" style="width:{bar_width};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Model: {result['model']}. Confidence is the model's estimated probability "
                "for the winning class, not a guarantee of correctness."
            )

            if result["probabilities"]:
                top = result["probabilities"][:5]
                with st.container(border=True):
                    section(
                        "Top alternatives", "How the model distributed its probability."
                    )
                    render_chart(
                        analytics.probability_bar(
                            [config.display_name(label) for label, _ in top],
                            [value for _, value in top],
                            height=280,
                        )
                    )

            with st.expander("See the cleaned text used by the model"):
                st.code(result["cleaned"] or "(empty)", language="text")
                st.caption(
                    "Lower-casing, contraction expansion, URL/redaction removal, punctuation "
                    "removal, tokenisation and stop-word removal all happen before TF-IDF."
                )

            with st.expander("How should I read this prediction?"):
                st.markdown(
                    "- **Category** is the highest-probability product family for this narrative.\n"
                    "- **Confidence** is the model's probability for that class. Scores below "
                    "about 60% mean the text overlaps several categories.\n"
                    "- The model was trained on real CFPB financial complaints. A complaint about "
                    "a non-financial retail purchase may be mapped to the closest financial "
                    "category (often Credit Card).\n"
                    "- Predictions are an aid for triage, not a legal or regulatory decision."
                )


# --------------------------------------------------------------------------- #
# Page 3 - Analytics
# --------------------------------------------------------------------------- #
def page_analytics(df: pd.DataFrame) -> None:
    hero(
        "Exploratory analytics",
        "Complaint analytics",
        "Filter the complaint population by date, product and free text, then explore "
        "category mix, monthly trends, company concentration, response behaviour and "
        "geography.",
        ["Interactive filters", "Plotly charts", "Pattern discovery"],
    )

    dates = pd.to_datetime(df["date_received"], errors="coerce").dropna()
    min_date, max_date = (
        (dates.min().date(), dates.max().date()) if not dates.empty else (None, None)
    )
    categories = sorted(df[config.TARGET_COLUMN].dropna().unique().tolist())

    with st.container(border=True):
        section("Filters", "Every chart below responds to these controls.")
        col1, col2, col3 = st.columns([1.15, 1.4, 1])
        with col1:
            if min_date and max_date:
                date_range = st.date_input(
                    "Date received",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
            else:
                date_range = None
        with col2:
            selected = st.multiselect(
                "Product categories",
                options=categories,
                default=categories,
                format_func=config.display_name,
            )
        with col3:
            search = st.text_input("Search narrative", placeholder="e.g. overdraft")

    filtered = analytics.apply_filters(
        df, date_range=date_range, products=selected, search=search
    )

    if filtered.empty:
        st.info(
            "No complaints match the current filters. Widen the date range or clear the search."
        )
        return

    stats = analytics.kpis(filtered)
    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} complaints.")

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        kpi_card(
            "Filtered complaints",
            fmt(len(filtered)),
            "Rows matching the filters",
            "🔎",
            "#4F46E5",
        )
    with col2:
        kpi_card(
            "Categories present",
            fmt(stats["total_categories"]),
            "Product families in view",
            "🏷️",
            "#06B6D4",
        )
    with col3:
        kpi_card(
            "Top category",
            config.display_name(stats["top_category"]),
            f"{pct(stats['top_category_share'])} of the filtered set",
            "🥇",
            "#F59E0B",
        )
    with col4:
        kpi_card(
            "Timely responses",
            pct(stats["timely_rate"]),
            "Company response timeliness",
            "⏱️",
            "#10B981",
        )

    counts = analytics.category_counts(filtered)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        with chart_card(
            "Category-wise distribution", "Complaint counts after filtering."
        ):
            render_chart(analytics.category_bar(counts, height=380))
    with right:
        with chart_card("Category share", "Proportion of the filtered population."):
            render_chart(analytics.category_pie(counts, height=380))

    with chart_card(
        "Monthly complaint trend", "Volume by month for the current selection."
    ):
        render_chart(
            analytics.monthly_line(analytics.monthly_counts(filtered), height=320)
        )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with chart_card(
            "Top companies", "Companies named most often in the filtered complaints."
        ):
            render_chart(
                analytics.company_bar(
                    analytics.company_counts(filtered, 10), height=400
                )
            )
    with right:
        with chart_card(
            "Company response outcomes", "How companies closed the complaints."
        ):
            response = analytics.response_counts(filtered)
            if response.empty:
                st.caption("No company-response field in this dataset.")
            else:
                render_chart(analytics.response_pie(response, height=400))

    left, right = st.columns(2, gap="large")
    with left:
        with chart_card("Complaints by state", "Top states in the current selection."):
            states = analytics.state_counts(filtered, 12)
            if states.empty:
                st.caption("No state field in this dataset.")
            else:
                render_chart(analytics.state_bar(states, height=340))
    with right:
        with chart_card("Narrative length by category", "Average words per complaint."):
            stats_frame = analytics.narrative_word_stats(filtered)
            if stats_frame.empty:
                st.caption("No narrative-length data available.")
            else:
                render_chart(analytics.word_stats_bar(stats_frame, height=340))

    st.write("")
    with st.container(border=True):
        section("Pattern notes", "Plain-language observations from the current filter.")
        if not counts.empty:
            leading = counts.head(3)
            names = ", ".join(
                f"{config.display_name(row['product'])} ({int(row['complaints']):,})"
                for _, row in leading.iterrows()
            )
            insight("Leading categories", f"The top three categories are {names}.")
        if not response.empty:
            top_response = response.iloc[0]
            insight(
                "Most common outcome",
                f"“{top_response['company_response']}” is the most frequent company response, "
                f"covering {int(top_response['complaints']):,} complaints.",
            )
        timely = analytics.timely_counts(filtered)
        if not timely.empty:
            yes = timely[timely["timely_response"].str.lower() == "yes"][
                "complaints"
            ].sum()
            total = timely["complaints"].sum()
            insight(
                "Response timeliness",
                f"{yes / total:.1%} of answered complaints received a timely response."
                if total
                else "No timeliness data in this selection.",
            )


# --------------------------------------------------------------------------- #
# Page 4 - Dataset explorer
# --------------------------------------------------------------------------- #
def page_explorer(df: pd.DataFrame, artifacts: dict | None) -> None:
    hero(
        "Data workbench",
        "Dataset explorer",
        "Inspect the real complaint data, check its quality, search individual records, "
        "and upload your own CSV to classify a batch of complaints.",
        ["Preview", "Quality report", "CSV upload", "Download results"],
    )

    browse_tab, upload_tab, quality_tab = st.tabs(
        ["📋  Browse dataset", "⬆️  Upload & classify", "🧪  Data quality"]
    )

    # ---------------------------- Browse ---------------------------- #
    with browse_tab:
        summary = data_loader.dataset_summary(df)
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        with col1:
            kpi_card("Rows", fmt(summary["rows"]), "Complaint records", "📄", "#4F46E5")
        with col2:
            kpi_card(
                "Columns",
                fmt(summary["columns"]),
                "Fields in the dataset",
                "🧱",
                "#06B6D4",
            )
        with col3:
            memory = f"{summary['memory_mb']:,.1f} MB"
            kpi_card("Memory", memory, "In-memory size", "💾", "#10B981")
        with col4:
            span = (
                f"{summary.get('date_min', 'n/a')} → {summary.get('date_max', 'n/a')}"
            )
            kpi_card("Date span", span, "Received-date coverage", "📅", "#F59E0B")

        st.write("")
        with st.container(border=True):
            section("Search and filter", "Narrow the table before downloading it.")
            col1, col2, col3 = st.columns([1.4, 1, 1])
            with col1:
                search = st.text_input(
                    "Text search",
                    placeholder="Search complaint narratives…",
                    key="explore_search",
                )
            with col2:
                products = st.multiselect(
                    "Category",
                    options=sorted(df[config.TARGET_COLUMN].dropna().unique().tolist()),
                    default=[],
                    format_func=config.display_name,
                    key="explore_products",
                )
            with col3:
                limit = st.slider("Rows to display", 10, 500, 50, step=10)

        view = df
        if products:
            view = view[view[config.TARGET_COLUMN].isin(products)]
        if search:
            view = view[
                view[config.TEXT_COLUMN]
                .astype("string")
                .str.contains(search, case=False, na=False, regex=False)
            ]
        view = analytics.ensure_datetime(view).sort_values(
            "date_received", ascending=False
        )

        st.caption(
            f"{len(view):,} matching records. Showing the first {min(limit, len(view)):,}."
        )
        display_columns = [
            column
            for column in [
                "date_received",
                "product",
                "issue",
                "company",
                "state",
                "company_response",
                "timely_response",
                "narrative",
            ]
            if column in view.columns
        ]
        preview = view[display_columns].head(limit).copy()
        if "narrative" in preview.columns:
            preview["narrative"] = preview["narrative"].astype(str).str.slice(0, 240)
        st.dataframe(preview, width="stretch", hide_index=True, height=460)

        download_col, info_col = st.columns([1, 2])
        with download_col:
            st.download_button(
                "⬇️  Download filtered CSV",
                data=view[display_columns].to_csv(index=False).encode("utf-8"),
                file_name="filtered_complaints.csv",
                mime="text/csv",
                width="stretch",
            )
        with info_col:
            st.caption(
                "Downloads respect the search, category and sort controls above."
            )

    # ---------------------------- Upload ---------------------------- #
    with upload_tab:
        with st.container(border=True):
            section(
                "Upload a CSV",
                "The file needs a complaint-text column. A category column is optional.",
            )
            uploaded = st.file_uploader(
                "Choose a CSV file", type=["csv"], key="uploader"
            )

            if uploaded is not None:
                try:
                    upload_df = pd.read_csv(uploaded, low_memory=False)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read the CSV: {exc}")
                    upload_df = None

                if upload_df is not None and not upload_df.empty:
                    upload_df = data_loader.normalize_columns(upload_df)
                    ok, messages = data_loader.validate_upload(upload_df)
                    for message in messages:
                        (st.warning if ok else st.error)(message)

                    text_candidates = [
                        column
                        for column in upload_df.columns
                        if pd.api.types.is_object_dtype(upload_df[column])
                        or pd.api.types.is_string_dtype(upload_df[column])
                    ]
                    default_index = 0
                    if config.TEXT_COLUMN in text_candidates:
                        default_index = text_candidates.index(config.TEXT_COLUMN)

                    text_column = st.selectbox(
                        "Complaint-text column",
                        options=text_candidates,
                        index=default_index,
                        key="upload_text_column",
                    )

                    st.caption(
                        f"{len(upload_df):,} rows · {upload_df.shape[1]} columns"
                    )
                    st.dataframe(
                        upload_df.head(10), width="stretch", hide_index=True, height=260
                    )

                    n_rows = len(upload_df)
                    slider_max = max(2, min(10_000, n_rows))
                    slider_min = 1
                    default_rows = min(slider_max, max(slider_min, min(1_000, n_rows)))
                    if slider_max > slider_min:
                        max_rows = st.slider(
                            "Maximum rows to classify",
                            min_value=slider_min,
                            max_value=slider_max,
                            value=default_rows,
                            step=10 if slider_max - slider_min >= 10 else 1,
                        )
                    else:
                        max_rows = n_rows
                        st.caption(f"Classifying all {n_rows:,} uploaded rows.")

                    if st.button("⚡  Classify uploaded complaints", type="primary"):
                        if not artifacts:
                            st.error(
                                "Train the model first (`uv run python train_model.py`)."
                            )
                        else:
                            with st.spinner("Classifying the uploaded complaints…"):
                                results = classify_frame(
                                    upload_df, text_column, artifacts, max_rows=max_rows
                                )
                            st.session_state["upload_results"] = results
                            st.session_state["upload_name"] = uploaded.name

                    results = st.session_state.get("upload_results")
                    if (
                        results is not None
                        and st.session_state.get("upload_name") == uploaded.name
                    ):
                        st.success(
                            f"Classified {results['predicted_category'].notna().sum():,} complaints."
                        )
                        result_columns = [
                            column
                            for column in [
                                "predicted_display",
                                "predicted_category",
                                "confidence",
                                text_column,
                            ]
                            if column in results.columns
                        ]
                        st.dataframe(
                            results[result_columns].head(200),
                            width="stretch",
                            hide_index=True,
                            height=360,
                        )
                        st.download_button(
                            "⬇️  Download classified results",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name=f"classified_{uploaded.name}",
                            mime="text/csv",
                        )
                        distribution = (
                            results["predicted_display"]
                            .dropna()
                            .value_counts()
                            .rename_axis("category")
                            .reset_index(name="complaints")
                        )
                        if not distribution.empty:
                            figure = analytics.category_bar(
                                distribution.rename(columns={"category": "label"}),
                                height=320,
                            )
                            render_chart(figure)

    # ---------------------------- Quality ---------------------------- #
    with quality_tab:
        with st.container(border=True):
            section("Missing-value analysis", "Null and blank counts per column.")
            quality = pd.DataFrame(
                {
                    "column": df.columns,
                    "missing": [int(df[column].isna().sum()) for column in df.columns],
                    "missing_pct": [df[column].isna().mean() for column in df.columns],
                    "dtype": [str(df[column].dtype) for column in df.columns],
                }
            ).sort_values("missing", ascending=False)
            quality["missing_pct"] = quality["missing_pct"].map(
                lambda value: f"{value:.1%}"
            )
            st.dataframe(quality, width="stretch", hide_index=True, height=320)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            with st.container(border=True):
                section("Text quality", "Narrative length and duplication.")
                lengths = (
                    df[config.TEXT_COLUMN]
                    .astype("string")
                    .fillna("")
                    .str.findall(r"[A-Za-z]+")
                    .map(len)
                )
                duplicates = int(
                    df[config.TEXT_COLUMN].astype("string").duplicated().sum()
                )
                st.markdown(
                    f"- Narratives: **{len(df):,}**\n"
                    f"- Duplicate narratives: **{duplicates:,}**\n"
                    f"- Shortest: **{int(lengths.min()):,} words**\n"
                    f"- Median: **{int(lengths.median()):,} words**\n"
                    f"- Average: **{lengths.mean():.0f} words**\n"
                    f"- Longest: **{int(lengths.max()):,} words**"
                )
        with col2:
            with st.container(border=True):
                section("Category counts", "Records per product family.")
                counts = analytics.category_counts(df)
                counts["share"] = counts["share"].map(lambda value: f"{value:.1%}")
                st.dataframe(
                    counts[["label", "complaints", "share"]].rename(
                        columns={
                            "label": "Category",
                            "complaints": "Complaints",
                            "share": "Share",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    height=320,
                )

        with st.container(border=True):
            section(
                "Field reference", "How the raw CFPB fields are used by this project."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        (
                            "date_received",
                            "Date the CFPB received the complaint",
                            "Trends, date filters",
                        ),
                        (
                            "product",
                            "Financial product family",
                            "Classification target",
                        ),
                        ("sub_product", "More specific product", "Exploration"),
                        ("issue", "The consumer's stated issue", "Exploration"),
                        (
                            "narrative",
                            "Free-text complaint story",
                            "TF-IDF model input",
                        ),
                        (
                            "company",
                            "Company named in the complaint",
                            "Company analysis",
                        ),
                        ("state", "Consumer's state", "Geography chart"),
                        (
                            "company_response",
                            "How the company closed the case",
                            "Outcome analysis",
                        ),
                        (
                            "timely_response",
                            "Whether the response was timely",
                            "Service KPI",
                        ),
                        (
                            "consumer_disputed",
                            "Whether the consumer disputed the response",
                            "Quality signal",
                        ),
                    ],
                    columns=["Field", "Meaning", "Used for"],
                ),
                width="stretch",
                hide_index=True,
            )


# --------------------------------------------------------------------------- #
# Page 5 - Model performance
# --------------------------------------------------------------------------- #
def page_performance(artifacts: dict | None) -> None:
    hero(
        "Evaluation",
        "Model performance",
        "Every number on this page comes from the held-out test split. The TF-IDF "
        "vectorizer was fitted on training text only, so no test information leaks "
        "into the model.",
        ["Stratified 80/20 split", "Held-out test set", "Two models compared"],
    )

    if not artifacts:
        st.warning(
            "No evaluation artefacts found. Run `uv run python train_model.py`, then press "
            "**Reload model & data** in the sidebar."
        )
        return

    metadata = artifacts["metadata"]
    metrics = artifacts["metrics"]
    comparison = artifacts["comparison"]
    confusion = artifacts["confusion"]
    report = artifacts["report"]
    best = next(
        (item for item in metrics["models"] if item["model"] == metrics["best_model"]),
        metrics["models"][0],
    )

    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    with col1:
        kpi_card(
            "Selected model",
            metadata["best_model"],
            "Chosen by weighted F1",
            "🏆",
            "#4F46E5",
        )
    with col2:
        kpi_card(
            "Accuracy",
            pct(best["accuracy"]),
            "Correct on held-out data",
            "🎯",
            "#10B981",
        )
    with col3:
        kpi_card(
            "Weighted F1",
            pct(best["f1_weighted"]),
            "Balances precision and recall",
            "⚖️",
            "#06B6D4",
        )
    with col4:
        kpi_card(
            "Test rows",
            fmt(metadata["test_rows"]),
            f"of {metadata['dataset_rows']:,} total",
            "🧪",
            "#F59E0B",
        )
    with col5:
        kpi_card(
            "TF-IDF features",
            fmt(metadata["n_features"]),
            "Unigrams + bigrams",
            "🧮",
            "#8B5CF6",
        )

    st.write("")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        with chart_card(
            "Model comparison",
            "Accuracy, precision, recall and F1 on the same test split.",
        ):
            render_chart(analytics.comparison_bar(comparison, height=340))
    with right:
        with st.container(border=True):
            section("Comparison table", "Higher is better for every metric.")
            formatted = comparison.copy()
            for column in ["accuracy", "precision", "recall", "f1_score", "macro_f1"]:
                if column in formatted.columns:
                    formatted[column] = formatted[column].map(
                        lambda value: f"{value:.4f}"
                    )
            st.dataframe(
                formatted.rename(
                    columns={
                        "model": "Model",
                        "accuracy": "Accuracy",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1_score": "Weighted F1",
                        "macro_f1": "Macro F1",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            winner = comparison.iloc[0]
            loser = comparison.iloc[1] if len(comparison) > 1 else None
            if loser is not None:
                insight(
                    "Why this model won",
                    f"{winner['model']} reached {winner['f1_score']:.4f} weighted F1 versus "
                    f"{loser['f1_score']:.4f} for {loser['model']}. Logistic Regression learns "
                    "a weight per word and class, which suits sparse TF-IDF text; Naive Bayes "
                    "is faster but assumes independent features.",
                )
            st.caption(
                "Precision: of the complaints predicted in a class, how many were correct. "
                "Recall: of the complaints truly in a class, how many were found. "
                "F1: the harmonic mean of the two."
            )

    with chart_card(
        "Confusion matrix", "Rows are actual categories, columns are predictions."
    ):
        if confusion is not None:
            render_chart(analytics.confusion_heatmap(confusion, height=640))
            st.caption(
                "A strong model has large values on the diagonal. Off-diagonal cells show which "
                "categories the model confuses, usually because the narratives share vocabulary."
            )
        else:
            st.caption("Confusion matrix not found.")

    with chart_card(
        "Per-category F1 score",
        "Where the classifier is strong and where it struggles.",
    ):
        if report is not None:
            render_chart(analytics.per_class_f1_bar(report, height=400))
        else:
            st.caption("Classification report not found.")

    with st.container(border=True):
        section(
            "Classification report",
            "Precision, recall, F1 and support for every class.",
        )
        if report is not None:
            display = report.copy()
            display.index = [
                "Weighted average"
                if name == "weighted avg"
                else "Macro average"
                if name == "macro avg"
                else "Accuracy"
                if name == "accuracy"
                else config.display_name(name)
                for name in display.index
            ]
            for column in ["precision", "recall", "f1-score"]:
                if column in display.columns:
                    display[column] = display[column].map(
                        lambda value: "—" if pd.isna(value) else f"{value:.4f}"
                    )
            if "support" in display.columns:
                display["support"] = display["support"].map(
                    lambda value: f"{int(value):,}"
                )
            st.dataframe(display, width="stretch")
        else:
            st.caption("Classification report not found.")

    if artifacts.get("top_features"):
        with st.container(border=True):
            section(
                "What the model learned",
                "The strongest TF-IDF terms for each category (Logistic Regression coefficients).",
            )
            selected_class = st.selectbox(
                "Category",
                options=list(artifacts["top_features"].keys()),
                format_func=config.display_name,
            )
            terms = artifacts["top_features"].get(selected_class, [])
            pills = "".join(
                f'<span class="pill">{esc(item["term"])} · {item["weight"]:+.3f}</span>'
                for item in terms
            )
            st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)
            st.caption(
                "Positive weights push a complaint toward this category. These terms are a useful "
                "sanity check that the model learned meaningful vocabulary, not noise."
            )

    with st.expander("Model card and reproducibility details"):
        st.markdown(
            f"- **Dataset source:** CFPB Consumer Complaint Database (public mirror with narratives)\n"
            f"- **Rows used:** {metadata['dataset_rows']:,}\n"
            f"- **Train / test:** {metadata['train_rows']:,} / {metadata['test_rows']:,}\n"
            f"- **Split:** stratified, test size {metadata['test_size']:.0%}, random state {metadata['random_state']}\n"
            f"- **TF-IDF:** max {metadata['tfidf']['max_features']:,} features, "
            f"ngram range {tuple(metadata['tfidf']['ngram_range'])}, "
            f"min_df {metadata['tfidf']['min_df']}, max_df {metadata['tfidf']['max_df']}\n"
            f"- **Selected model:** {metadata['best_model']}\n"
            f"- **Trained at:** {metadata['trained_at_utc']} UTC\n"
            f"- **Python:** {metadata['python']}"
        )
        st.caption(
            "No test document is used to fit the vectorizer or the classifier. Duplicate "
            "narratives are removed before the split to avoid near-duplicate leakage."
        )


# --------------------------------------------------------------------------- #
# Page 6 - About
# --------------------------------------------------------------------------- #
def page_about(df: pd.DataFrame, artifacts: dict | None) -> None:
    hero(
        "Documentation",
        "About this project",
        "An individual Web Data Mining mini-project that turns public consumer "
        "complaints into a searchable, classifiable intelligence dashboard.",
        ["Python", "Streamlit", "scikit-learn", "Plotly"],
    )

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        with st.container(border=True):
            section(
                "What the system does",
                "From raw complaint text to an actionable category.",
            )
            st.markdown(
                "Consumers send free-text complaints to companies and regulators. Reading "
                "them one by one is slow and inconsistent. This system:\n\n"
                "1. **Collects** a real sample of complaints from the public CFPB database.\n"
                "2. **Cleans** every narrative: lower-casing, URL and redaction removal, "
                "tokenisation, stop-word removal.\n"
                "3. **Represents** each complaint as a TF-IDF vector of unigrams and bigrams.\n"
                "4. **Classifies** the complaint into a financial product family with "
                "Logistic Regression or Multinomial Naive Bayes.\n"
                "5. **Explains** the result with a confidence score, top alternatives and the "
                "terms that drove the model.\n"
                "6. **Explores** the population with interactive category, trend, company and "
                "response charts.\n"
            )
        with st.container(border=True):
            section("System architecture", "The flow implemented in this repository.")
            steps = [
                (
                    "01",
                    "Collect",
                    "CFPB public complaint data + user CSV upload",
                    "#4F46E5",
                ),
                (
                    "02",
                    "Clean",
                    "Missing values, duplicates, short narratives",
                    "#06B6D4",
                ),
                (
                    "03",
                    "Preprocess",
                    "Lower-case, tokenise, remove stop-words",
                    "#8B5CF6",
                ),
                ("04", "Vectorise", "TF-IDF unigrams + bigrams", "#F59E0B"),
                ("05", "Classify", "Logistic Regression / Naive Bayes", "#10B981"),
                ("06", "Explain", "Confidence, alternatives, top terms", "#EF4444"),
            ]
            boxes = []
            for index, (number, name, description, accent) in enumerate(steps):
                boxes.append(
                    f'<div class="flow-box" style="--accent:{accent};">'
                    f'<div class="flow-step">Step {number}</div>'
                    f'<div class="flow-name">{name}</div>'
                    f'<div class="flow-desc">{description}</div></div>'
                )
                if index < len(steps) - 1:
                    boxes.append('<div class="flow-arrow">→</div>')
            st.markdown(
                f'<div class="flow-row">{"".join(boxes)}</div>', unsafe_allow_html=True
            )

        with st.container(border=True):
            section("Dataset provenance", "Honest sourcing, including a limitation.")
            st.markdown(
                f"**Official source:** [CFPB Consumer Complaint Database]"
                f"({config.CFPB_HOME}). The CFPB publishes complaints about financial "
                "products and services after the company responds or after 15 days.\n\n"
                f"**The limitation:** the current official bulk export "
                f"(`complaints.csv.zip`) contains complaint metadata but no free-text "
                "narrative column. A text classifier needs the narrative, so this project "
                f"samples a public mirror of the same CFPB database that preserves it: "
                f"[{config.CFPB_MIRROR_URL}]({config.CFPB_MIRROR_URL}).\n\n"
                f"**Sample used:** {len(df):,} randomly selected narrative complaints, "
                f"{df['date_received'].min():%b %Y} to {df['date_received'].max():%b %Y}, "
                f"across {df[config.TARGET_COLUMN].nunique()} product families. "
                "No records are fabricated. Absolute counts describe this sample only.\n\n"
                "**Uploads:** the Dataset Explorer accepts any CSV with a text column, so the "
                "same pipeline can be demonstrated on other complaint collections."
            )

    with right:
        with st.container(border=True):
            section(
                "Technology stack",
                "Chosen to stay explainable for an engineering student.",
            )
            stack = [
                ("Python", "Application language"),
                ("Streamlit", "Interactive web dashboard"),
                ("Pandas + NumPy", "Data processing"),
                ("scikit-learn", "TF-IDF, models, metrics"),
                ("Plotly", "Interactive charts"),
                ("Matplotlib / Seaborn", "Report figures"),
                ("Joblib", "Model persistence"),
                ("GitHub + Streamlit Cloud", "Source control and deployment"),
            ]
            for name, purpose in stack:
                st.markdown(f"- **{name}** — {purpose}")

        with st.container(border=True):
            section(
                "Web Data Mining concepts", "Where each concept appears in the project."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Web data collection", "CFPB public data + CSV upload"),
                        ("Text mining", "40k complaint narratives"),
                        ("Text preprocessing", "preprocessing.py"),
                        ("Information extraction", "TF-IDF features"),
                        ("Feature extraction", "unigrams + bigrams"),
                        ("Text classification", "Logistic Regression / Naive Bayes"),
                        ("Pattern identification", "Analytics + top model terms"),
                        ("Visualisation", "Six-page Plotly dashboard"),
                    ],
                    columns=["Concept", "Implementation"],
                ),
                width="stretch",
                hide_index=True,
            )

        with st.container(border=True):
            section("Ethics and limitations", "What the system should not be used for.")
            st.markdown(
                "- Complaints are **not** a statistical sample of all consumers, so counts do "
                "not measure a company's overall service quality.\n"
                "- The classifier predicts a **product family**, not fault, legality or outcome.\n"
                "- Confidence is a model probability and can be wrong on short, ambiguous or "
                "non-financial text.\n"
                "- The model reflects the vocabulary of its training snapshot; new products and "
                "slang need retraining.\n"
                "- Do not use predictions as the sole basis for a legal or regulatory decision."
            )

        with st.container(border=True):
            section("Project files", "What each module owns.")
            st.markdown(
                "- `app.py` — Streamlit UI and all six pages\n"
                "- `train_model.py` — training, comparison, artefact export\n"
                "- `preprocessing.py` — text cleaning and tokenisation\n"
                "- `data_loader.py` — download, sampling, column normalisation\n"
                "- `analytics.py` — aggregations and Plotly figures\n"
                "- `fetch_data.py` — one-command dataset builder\n"
                "- `config.py` — paths, constants, labels\n"
                "- `model/` — trained model, vectorizer, metrics\n"
                "- `docs/` — report, viva guide, deployment guide"
            )

        if artifacts:
            metadata = artifacts["metadata"]
            with st.container(border=True):
                section("Current model", "Loaded from the saved artefacts.")
                st.markdown(
                    f"- Best model: **{metadata['best_model']}**\n"
                    f"- Trained: **{metadata['trained_at_utc']} UTC**\n"
                    f"- Classes: **{len(metadata['labels'])}**\n"
                    f"- Features: **{metadata['n_features']:,}**"
                )

    st.markdown(
        '<div class="footer">Built as an individual Web Data Mining laboratory mini-project. '
        "Dataset: Consumer Financial Protection Bureau, public domain. "
        "Code: Python, scikit-learn and Streamlit. "
        "Predictions are for academic demonstration.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def main() -> None:
    config.ensure_directories()
    df = get_dataset()
    artifacts = load_artifacts()
    page = sidebar(df, artifacts)

    if "Home Dashboard" in page:
        if df is not None:
            page_home(df)
    elif "Complaint Classification" in page:
        page_classify(artifacts)
    elif "Complaint Analytics" in page:
        if df is not None:
            page_analytics(df)
    elif "Dataset Explorer" in page:
        if df is not None:
            page_explorer(df, artifacts)
    elif "Model Performance" in page:
        page_performance(artifacts)
    elif "About Project" in page:
        if df is not None:
            page_about(df, artifacts)


if __name__ == "__main__":
    main()
