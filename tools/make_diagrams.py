"""Generate the architecture diagram and project flowchart used in the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#0F172A"
MUTED = "#475569"
COLORS = {
    "data": "#4F46E5",
    "process": "#0891B2",
    "model": "#7C3AED",
    "artifact": "#D97706",
    "app": "#059669",
    "decision": "#DC2626",
}


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    return fig, ax


def box(ax, x, y, w, h, title, subtitle="", kind="process", fontsize=9.5):
    color = COLORS[kind]
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=1.6",
        linewidth=1.6,
        edgecolor=color,
        facecolor=color + "14",
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x,
            y + h * 0.14,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            x,
            y - h * 0.20,
            subtitle,
            ha="center",
            va="center",
            fontsize=fontsize - 2.2,
            color=MUTED,
            linespacing=1.25,
        )
    else:
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=INK,
        )


def arrow(ax, start, end, color="#64748B", style="-|>", width=1.6, curve=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=width,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
        )
    )


def architecture() -> None:
    fig, ax = canvas(12.5, 10.5)
    ax.set_ylim(-6, 100)
    ax.text(
        50,
        97.5,
        "System architecture",
        ha="center",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        50,
        93.8,
        "Online Consumer Complaint Classification System",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    # Data sources
    box(
        ax,
        15,
        87,
        25,
        7,
        "Official CFPB database",
        "consumerfinance.gov",
        "data",
        fontsize=9,
    )
    box(
        ax,
        50,
        87,
        25,
        7,
        "Public CFPB mirror",
        "preserves narrative text",
        "data",
        fontsize=9,
    )
    box(
        ax,
        85,
        87,
        25,
        7,
        "User CSV upload",
        "any complaint-text column",
        "data",
        fontsize=9,
    )

    # Vertical training pipeline
    box(
        ax,
        50,
        76,
        72,
        7.2,
        "Data ingestion and sampling",
        "download -> keep narrative rows -> reproducible 40,000-row reservoir sample",
        "process",
    )
    box(
        ax,
        50,
        65.5,
        72,
        7.2,
        "Cleaning and preprocessing",
        "missing values, duplicates, short texts -> lower-case, URLs, redactions, tokens, stop-words",
        "process",
    )
    box(
        ax,
        50,
        55,
        72,
        7.2,
        "Stratified train / test split",
        "80% training, 20% held-out test, fixed random state for reproducibility",
        "model",
    )
    box(
        ax,
        50,
        44.5,
        72,
        7.2,
        "TF-IDF feature extraction",
        "unigrams + bigrams, 60,000 features, fitted on the training split only",
        "model",
    )
    box(ax, 31, 32.5, 30, 7.2, "Logistic Regression", "balanced class weights", "model")
    box(ax, 69, 32.5, 30, 7.2, "Multinomial Naive Bayes", "alpha = 0.1", "model")
    box(
        ax,
        50,
        20.5,
        72,
        7.2,
        "Evaluation and model selection",
        "accuracy, precision, recall, F1, confusion matrix -> best weighted F1",
        "artifact",
    )
    box(
        ax,
        50,
        11.0,
        72,
        6.4,
        "Saved model artefacts",
        "trained_model.joblib | tfidf_vectorizer.joblib | metrics.json | reports",
        "artifact",
    )
    box(
        ax,
        50,
        2.0,
        86,
        6.2,
        "Streamlit dashboard",
        "Home | Classification | Analytics | Dataset Explorer | Model Performance | About",
        "app",
    )

    # Pipeline arrows
    arrow(ax, (15, 83.3), (32, 80.0))
    arrow(ax, (50, 83.3), (50, 80.0))
    arrow(ax, (85, 83.3), (68, 80.0))
    arrow(ax, (50, 72.3), (50, 69.3))
    arrow(ax, (50, 61.8), (50, 58.8))
    arrow(ax, (50, 51.3), (50, 48.3))
    arrow(ax, (40, 40.8), (33, 36.3))
    arrow(ax, (60, 40.8), (67, 36.3))
    arrow(ax, (31, 28.8), (42, 24.3))
    arrow(ax, (69, 28.8), (58, 24.3))
    arrow(ax, (50, 16.8), (50, 14.2))
    arrow(ax, (50, 7.8), (50, 5.1))

    ax.text(
        50,
        -4.6,
        "Prediction path: user text -> same cleaning -> saved vectorizer -> selected model -> category + confidence",
        ha="center",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    fig.savefig(
        OUT / "architecture.png", dpi=220, bbox_inches="tight", facecolor="#FFFFFF"
    )
    plt.close(fig)
    print("saved docs/architecture.png")


def flowchart() -> None:
    fig, ax = canvas(10.5, 15.5)
    ax.text(
        50,
        98.5,
        "Project flowchart",
        ha="center",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )

    box(ax, 50, 94, 34, 4.6, "Start", "", "app", fontsize=10)
    steps = [
        (87, "Load CFPB complaint dataset", "validate text + product columns", "data"),
        (
            79,
            "Clean records",
            "missing values, duplicates, short narratives",
            "process",
        ),
        (71, "Preprocess text", "lower-case, tokenise, remove stop-words", "process"),
        (
            63,
            "Select target categories",
            "minimum 150 records, cap dominant class",
            "process",
        ),
        (55, "Stratified train / test split", "80% / 20%, fixed random state", "model"),
        (
            47,
            "Fit TF-IDF on training text",
            "transform the test set separately",
            "model",
        ),
        (39, "Train both classifiers", "Logistic Regression + Multinomial NB", "model"),
        (
            31,
            "Evaluate on held-out test set",
            "accuracy, precision, recall, F1",
            "artifact",
        ),
    ]
    previous = (50, 91.7)
    for y, title, subtitle, kind in steps:
        box(ax, 50, y, 62, 5.6, title, subtitle, kind)
        arrow(ax, previous, (50, y + 2.9))
        previous = (50, y - 2.9)

    diamond = plt.Polygon(
        [(50, 25.6), (66, 22.7), (50, 19.8), (34, 22.7)],
        closed=True,
        facecolor=COLORS["decision"] + "14",
        edgecolor=COLORS["decision"],
        linewidth=1.6,
    )
    ax.add_patch(diamond)
    ax.text(
        50,
        22.7,
        "Best model\nweighted F1?",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )
    arrow(ax, previous, (50, 25.8))

    box(
        ax,
        18,
        15.5,
        28,
        5.6,
        "Save artefacts",
        "model, vectorizer, metrics",
        "artifact",
    )
    box(ax, 50, 12.2, 34, 4.8, "Launch Streamlit app", "cloud or local", "app")
    box(
        ax,
        50,
        4.8,
        58,
        6.2,
        "User enters a complaint -> predict",
        "same cleaning + vectorizer -> category + confidence + alternatives",
        "app",
    )

    arrow(ax, (34, 22.7), (18, 18.4), color=COLORS["decision"])
    ax.text(
        24, 22.4, "select", fontsize=8.5, color=COLORS["decision"], fontweight="bold"
    )
    arrow(ax, (18, 12.7), (33, 12.2))
    arrow(ax, (67, 12.2), (78, 18.4), color="#94A3B8", style="-|>", curve=-0.2)
    ax.text(76, 17.2, "retrain / tune", fontsize=8.5, color="#64748B", rotation=62)
    arrow(ax, (50, 9.8), (50, 8.0))

    fig.savefig(
        OUT / "flowchart.png", dpi=220, bbox_inches="tight", facecolor="#FFFFFF"
    )
    plt.close(fig)
    print("saved docs/flowchart.png")


if __name__ == "__main__":
    architecture()
    flowchart()
