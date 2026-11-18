"""
Analytics helpers and Plotly chart builders.

The Streamlit pages stay thin: they call these functions, pass user filters,
and render the returned dataframes/figures. Keeping the calculations here also
makes the analytics testable outside Streamlit.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config

FONT_FAMILY = "Inter, Segoe UI, system-ui, -apple-system, sans-serif"


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a parsed ``date_received`` column when present."""
    work = df.copy()
    if "date_received" in work.columns:
        work["date_received"] = pd.to_datetime(work["date_received"], errors="coerce")
    return work


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple | None = None,
    products: list[str] | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Apply the sidebar/page filters consistently."""
    work = ensure_datetime(df)

    if date_range and len(date_range) == 2 and "date_received" in work.columns:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        work = work[work["date_received"].between(start, end)]

    if products and config.TARGET_COLUMN in work.columns:
        work = work[work[config.TARGET_COLUMN].isin(products)]

    if search and config.TEXT_COLUMN in work.columns:
        work = work[
            work[config.TEXT_COLUMN]
            .astype("string")
            .str.contains(search, case=False, na=False, regex=False)
        ]

    return work


def kpis(df: pd.DataFrame) -> dict:
    """Headline numbers for the Home dashboard."""
    total = int(len(df))
    result = {
        "total_complaints": total,
        "total_categories": 0,
        "top_category": "n/a",
        "top_category_count": 0,
        "top_category_share": 0.0,
        "total_companies": 0,
        "timely_rate": None,
        "avg_words": 0.0,
        "date_min": None,
        "date_max": None,
    }
    if total == 0:
        return result

    if config.TARGET_COLUMN in df.columns:
        counts = df[config.TARGET_COLUMN].dropna().value_counts()
        result["total_categories"] = int(len(counts))
        if not counts.empty:
            result["top_category"] = str(counts.index[0])
            result["top_category_count"] = int(counts.iloc[0])
            result["top_category_share"] = float(counts.iloc[0] / total)

    if "company" in df.columns:
        result["total_companies"] = int(df["company"].dropna().nunique())

    if "timely_response" in df.columns:
        timely = df["timely_response"].astype("string").str.strip().str.lower()
        answered = timely.isin(["yes", "no"])
        if answered.any():
            result["timely_rate"] = float((timely == "yes").sum() / answered.sum())

    if config.TEXT_COLUMN in df.columns:
        words = (
            df[config.TEXT_COLUMN]
            .astype("string")
            .fillna("")
            .str.findall(r"[A-Za-z]+")
            .map(len)
        )
        result["avg_words"] = float(words.mean()) if len(words) else 0.0

    if "date_received" in df.columns:
        dates = pd.to_datetime(df["date_received"], errors="coerce").dropna()
        if not dates.empty:
            result["date_min"] = dates.min()
            result["date_max"] = dates.max()

    return result


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Complaints per product category, with display names and share."""
    if config.TARGET_COLUMN not in df.columns or df.empty:
        return pd.DataFrame(columns=["product", "label", "complaints", "share"])
    counts = (
        df[config.TARGET_COLUMN]
        .dropna()
        .value_counts()
        .rename_axis("product")
        .reset_index(name="complaints")
    )
    counts["label"] = counts["product"].map(config.display_name)
    total = counts["complaints"].sum()
    counts["share"] = counts["complaints"] / total if total else 0.0
    return counts


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Complaints per calendar month."""
    if "date_received" not in df.columns or df.empty:
        return pd.DataFrame(columns=["month", "complaints"])
    dates = pd.to_datetime(df["date_received"], errors="coerce").dropna()
    if dates.empty:
        return pd.DataFrame(columns=["month", "complaints"])
    months = dates.dt.to_period("M").dt.to_timestamp()
    counts = (
        months.value_counts()
        .sort_index()
        .rename_axis("month")
        .reset_index(name="complaints")
    )
    return counts


def company_counts(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Companies with the most complaints in the current selection."""
    if "company" not in df.columns or df.empty:
        return pd.DataFrame(columns=["company", "complaints"])
    counts = (
        df["company"]
        .dropna()
        .value_counts()
        .head(top_n)
        .rename_axis("company")
        .reset_index(name="complaints")
    )
    counts["short_company"] = counts["company"].str.slice(0, 34)
    return counts


def response_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of company responses."""
    if "company_response" not in df.columns or df.empty:
        return pd.DataFrame(columns=["company_response", "complaints"])
    return (
        df["company_response"]
        .dropna()
        .value_counts()
        .rename_axis("company_response")
        .reset_index(name="complaints")
    )


def timely_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Timely vs late company responses."""
    if "timely_response" not in df.columns or df.empty:
        return pd.DataFrame(columns=["timely_response", "complaints"])
    return (
        df["timely_response"]
        .dropna()
        .str.strip()
        .str.title()
        .value_counts()
        .rename_axis("timely_response")
        .reset_index(name="complaints")
    )


def state_counts(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Complaints per US state."""
    if "state" not in df.columns or df.empty:
        return pd.DataFrame(columns=["state", "complaints"])
    return (
        df["state"]
        .dropna()
        .str.upper()
        .value_counts()
        .head(top_n)
        .rename_axis("state")
        .reset_index(name="complaints")
    )


def category_by_month(df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Monthly counts for the leading categories (stacked trend chart)."""
    if "date_received" not in df.columns or config.TARGET_COLUMN not in df.columns:
        return pd.DataFrame(columns=["month", "label", "complaints"])
    work = ensure_datetime(df).dropna(subset=["date_received", config.TARGET_COLUMN])
    if work.empty:
        return pd.DataFrame(columns=["month", "label", "complaints"])
    leaders = work[config.TARGET_COLUMN].value_counts().head(top_n).index
    work = work[work[config.TARGET_COLUMN].isin(leaders)].copy()
    work["month"] = work["date_received"].dt.to_period("M").dt.to_timestamp()
    work["label"] = work[config.TARGET_COLUMN].map(config.display_name)
    grouped = (
        work.groupby(["month", "label"], observed=True)
        .size()
        .reset_index(name="complaints")
    )
    return grouped.sort_values("month")


def narrative_word_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Average narrative length per category (text-mining insight)."""
    if config.TEXT_COLUMN not in df.columns or config.TARGET_COLUMN not in df.columns:
        return pd.DataFrame(columns=["label", "avg_words"])
    work = df.dropna(subset=[config.TARGET_COLUMN]).copy()
    work["words"] = (
        work[config.TEXT_COLUMN]
        .astype("string")
        .fillna("")
        .str.findall(r"[A-Za-z]+")
        .map(len)
    )
    grouped = (
        work.groupby(config.TARGET_COLUMN, observed=True)["words"]
        .mean()
        .reset_index()
        .rename(columns={config.TARGET_COLUMN: "product", "words": "avg_words"})
    )
    grouped["label"] = grouped["product"].map(config.display_name)
    return grouped.sort_values("avg_words", ascending=False)


# --------------------------------------------------------------------------- #
# Chart styling
# --------------------------------------------------------------------------- #
def style_figure(
    fig: go.Figure, height: int = 360, show_legend: bool = False
) -> go.Figure:
    """Apply the dashboard's visual language to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        colorway=config.PALETTE,
        height=height,
        margin=dict(l=12, r=72, t=24, b=12),
        font=dict(family=FONT_FAMILY, size=12, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_family=FONT_FAMILY, font_size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None, layer="below traces")
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        title=None,
        layer="below traces",
    )
    return fig


def category_bar(df: pd.DataFrame, height: int = 360) -> go.Figure:
    data = df.sort_values("complaints", ascending=True)
    fig = px.bar(
        data,
        x="complaints",
        y="label",
        orientation="h",
        color="complaints",
        color_continuous_scale=["#C7D2FE", "#4F46E5"],
        text="complaints",
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="outside", cliponaxis=False
    )
    fig.update_layout(coloraxis_showscale=False)
    return style_figure(fig, height=height)


def category_pie(df: pd.DataFrame, height: int = 380) -> go.Figure:
    fig = px.pie(
        df,
        names="label",
        values="complaints",
        hole=0.55,
        color_discrete_sequence=config.PALETTE,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>%{value:,} complaints<br>%{percent}<extra></extra>",
    )
    return style_figure(fig, height=height, show_legend=True)


def monthly_line(df: pd.DataFrame, height: int = 360) -> go.Figure:
    fig = px.area(
        df,
        x="month",
        y="complaints",
        markers=True,
        color_discrete_sequence=[config.PALETTE[0]],
    )
    fig.update_traces(
        line=dict(width=2.5),
        fillcolor="rgba(79,70,229,0.12)",
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} complaints<extra></extra>",
    )
    return style_figure(fig, height=height)


def category_month_line(df: pd.DataFrame, height: int = 380) -> go.Figure:
    fig = px.line(
        df,
        x="month",
        y="complaints",
        color="label",
        markers=True,
        color_discrete_sequence=config.PALETTE,
    )
    fig.update_traces(line=dict(width=2))
    return style_figure(fig, height=height, show_legend=True)


def company_bar(df: pd.DataFrame, height: int = 400) -> go.Figure:
    data = df.sort_values("complaints", ascending=True)
    fig = px.bar(
        data,
        x="complaints",
        y="short_company",
        orientation="h",
        color_discrete_sequence=[config.PALETTE[1]],
        text="complaints",
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="outside", cliponaxis=False
    )
    fig.update_yaxes(tickfont=dict(size=11))
    return style_figure(fig, height=height)


def response_pie(df: pd.DataFrame, height: int = 360) -> go.Figure:
    fig = px.pie(
        df,
        names="company_response",
        values="complaints",
        hole=0.5,
        color_discrete_sequence=config.PALETTE[2:],
    )
    fig.update_traces(
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
    )
    return style_figure(fig, height=height, show_legend=True)


def state_bar(df: pd.DataFrame, height: int = 340) -> go.Figure:
    fig = px.bar(
        df,
        x="state",
        y="complaints",
        color="complaints",
        color_continuous_scale=["#A5F3FC", "#0891B2"],
        text="complaints",
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="outside", cliponaxis=False
    )
    fig.update_layout(coloraxis_showscale=False)
    return style_figure(fig, height=height)


def word_stats_bar(df: pd.DataFrame, height: int = 340) -> go.Figure:
    data = df.sort_values("avg_words", ascending=True)
    fig = px.bar(
        data,
        x="avg_words",
        y="label",
        orientation="h",
        color_discrete_sequence=[config.PALETTE[3]],
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x:.0f} words on average<extra></extra>"
    )
    return style_figure(fig, height=height)


def confusion_heatmap(matrix: pd.DataFrame, height: int = 620) -> go.Figure:
    labels = [label.replace("actual: ", "") for label in matrix.index]
    short = [config.display_name(label) for label in labels]
    values = matrix.to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=short,
            y=short,
            colorscale=[[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#312E81"]],
            text=values,
            texttemplate="%{text:,}",
            hovertemplate="Actual <b>%{y}</b><br>Predicted <b>%{x}</b><br>%{z:,} complaints<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_xaxes(tickangle=-35, side="bottom")
    return style_figure(fig, height=height)


def comparison_bar(df: pd.DataFrame, height: int = 340) -> go.Figure:
    melted = df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="Metric",
        value_name="Score",
    )
    melted["Metric"] = melted["Metric"].str.replace("_", " ").str.title()
    fig = px.bar(
        melted,
        x="model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=config.PALETTE[:4],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0, 1.08])
    return style_figure(fig, height=height, show_legend=True)


def per_class_f1_bar(report: pd.DataFrame, height: int = 380) -> go.Figure:
    rows = report[~report.index.isin(["accuracy", "macro avg", "weighted avg"])].copy()
    rows = rows.reset_index().rename(
        columns={"class_or_average": "product", "f1-score": "f1"}
    )
    rows["label"] = rows["product"].map(config.display_name)
    rows = rows.sort_values("f1", ascending=True)
    fig = px.bar(
        rows,
        x="f1",
        y="label",
        orientation="h",
        color="f1",
        color_continuous_scale=["#FCA5A5", "#EF4444", "#991B1B"],
        text="f1",
    )
    fig.update_traces(
        texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(range=[0, 1.08])
    return style_figure(fig, height=height)


def probability_bar(
    labels: list[str], probabilities: list[float], height: int = 300
) -> go.Figure:
    """Horizontal bar chart of the model's top class probabilities."""
    frame = pd.DataFrame({"label": labels, "probability": probabilities}).sort_values(
        "probability", ascending=True
    )
    fig = px.bar(
        frame,
        x="probability",
        y="label",
        orientation="h",
        color="probability",
        color_continuous_scale=["#E0E7FF", "#4F46E5"],
        text="probability",
    )
    fig.update_traces(
        texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(range=[0, 1.05], tickformat=".0%")
    fig = style_figure(fig, height=height)
    fig.update_yaxes(showgrid=False)
    return fig
