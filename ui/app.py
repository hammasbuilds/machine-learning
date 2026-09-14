"""Interactive explorer for the twenty projects.

Three tabs. The first browses every project's real measured results and figures. The
other two are live: you supply the input, the real code recomputes on the real data,
and the chart moves. Nothing here is a mock-up of a result - every number comes from
the same functions `run.py` uses.

Run: streamlit run ui/app.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "01-backtest-overfitting"))
sys.path.insert(0, str(ROOT / "projects" / "03-imbalanced-maintenance"))

PROJECTS_DIR = ROOT / "projects"

st.set_page_config(page_title="machine-learning · 20 projects", layout="wide")

# One palette for the whole app, so the charts read as a set rather than as twenty
# unrelated notebooks.
BLUE, RED, GREY, AMBER = "#2563eb", "#dc2626", "#94a3b8", "#f59e0b"


# --- loading -----------------------------------------------------------------------


@st.cache_data
def load_projects() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for folder in sorted(PROJECTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        readme = folder / "README.md"
        result = folder / "result.json"
        text = readme.read_text(encoding="utf-8", errors="replace") if readme.exists() else ""
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), folder.name)
        # The headline is the first bold paragraph - every README leads with its finding.
        headline = ""
        match = re.search(r"^\*\*(.+?)\*\*", text, re.S | re.M)
        if match:
            headline = " ".join(match.group(1).split())
        out[folder.name] = {
            "title": title,
            "headline": headline,
            "result": json.loads(result.read_text(encoding="utf-8")) if result.exists() else {},
            "figures": sorted((folder / "figures").glob("*.png")) if (folder / "figures").exists() else [],
            "readme": text,
        }
    return out


PROJECTS = load_projects()


def scalars(result: dict) -> dict:
    return {k: v for k, v in result.items() if isinstance(v, (int, float, str))}


def tables(result: dict) -> dict[str, list[dict]]:
    return {
        k: v
        for k, v in result.items()
        if isinstance(v, list) and v and all(isinstance(row, dict) for row in v)
    }


def grouped_bar(rows: list[dict], title: str) -> go.Figure | None:
    """Render a result table as a grouped bar chart, one series per numeric column."""
    label_key = next((k for k, v in rows[0].items() if isinstance(v, str)), None)
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not label_key or not numeric:
        return None
    labels = [str(r.get(label_key, i)) for i, r in enumerate(rows)]
    figure = go.Figure()
    palette = [BLUE, AMBER, GREY, RED, "#16a34a"]
    for i, column in enumerate(numeric):
        figure.add_bar(
            name=column,
            x=labels,
            y=[r.get(column) for r in rows],
            marker_color=palette[i % len(palette)],
        )
    figure.update_layout(
        title=title, barmode="group", height=380, margin=dict(t=48, b=40, l=8, r=8)
    )
    return figure


tab_browse, tab_search, tab_cost = st.tabs(
    ["Explore the 20 projects", "Lab · backtest overfitting", "Lab · cost-sensitive threshold"]
)

# --- tab 1: browse -------------------------------------------------------------------

with tab_browse:
    st.title("Twenty machine-learning projects")
    st.caption(
        "Each one built around the mistake that makes its answer wrong. Every number "
        "below was produced on real public data - sources and SHA-256 in "
        "`data/sources.json`."
    )

    name = st.selectbox("Project", list(PROJECTS), format_func=lambda k: PROJECTS[k]["title"])
    project = PROJECTS[name]

    if project["headline"]:
        st.info(project["headline"])

    numbers = scalars(project["result"])
    if numbers:
        st.subheader("Measured result")
        items = list(numbers.items())
        for start in range(0, len(items), 4):
            for column, (key, value) in zip(st.columns(4), items[start : start + 4]):
                shown = f"{value:,.4g}" if isinstance(value, (int, float)) else str(value)
                column.metric(key.replace("_", " "), shown)

    for table_name, rows in tables(project["result"]).items():
        st.subheader(table_name.replace("_", " "))
        figure = grouped_bar(rows, table_name.replace("_", " "))
        left, right = st.columns([3, 2])
        if figure is not None:
            left.plotly_chart(figure, use_container_width=True)
        right.dataframe(rows, hide_index=True, use_container_width=True)

    if project["figures"]:
        st.subheader(f"Figures ({len(project['figures'])})")
        for start in range(0, len(project["figures"]), 2):
            for column, path in zip(st.columns(2), project["figures"][start : start + 2]):
                column.image(str(path), caption=path.name, use_container_width=True)

# --- tab 2: live deflated Sharpe -----------------------------------------------------

with tab_search:
    from deflated import expected_max_sharpe  # noqa: E402

    st.title("How many strategies did you try?")
    st.caption(
        "Searching more strategies raises the Sharpe ratio the *best* one reaches even "
        "when none of them has an edge. This is the real `expected_max_sharpe` from "
        "`projects/01-backtest-overfitting/deflated.py` - move the inputs and watch the "
        "bar a strategy has to clear."
    )

    stored = PROJECTS["01-backtest-overfitting"]["result"]
    left, right = st.columns(2)
    with left:
        trials = st.slider("Strategies searched", 1, 5000, int(stored.get("trials", 846)), 1)
        observed = st.slider(
            "Your best strategy's Sharpe (annualised)",
            0.0, 3.0, float(stored.get("observed_sharpe", 0.62)), 0.01,
        )
    with right:
        sd = st.slider("Spread of Sharpe across the strategies tried (sd)", 0.05, 1.5, 0.5, 0.05)
        periods = st.number_input("Periods per year", 12, 365, 252)

    variance_per_period = (sd / np.sqrt(periods)) ** 2
    bar = expected_max_sharpe(trials, variance_per_period) * np.sqrt(periods)
    survives = observed > bar

    a, b, c = st.columns(3)
    a.metric("Strategies searched", f"{trials:,}")
    b.metric("Sharpe noise alone reaches", f"{bar:.3f}")
    c.metric("Your Sharpe", f"{observed:.3f}", delta=f"{observed - bar:+.3f} vs noise")

    if survives:
        st.success("Clears the bar - the edge is not explained by the size of the search alone.")
    else:
        st.error("Below the bar - a search this large produces this Sharpe from noise.")

    grid = np.unique(np.round(np.logspace(0, np.log10(5000), 120)).astype(int))
    curve = [expected_max_sharpe(int(n), variance_per_period) * np.sqrt(periods) for n in grid]
    figure = go.Figure()
    figure.add_scatter(x=grid, y=curve, name="Sharpe from noise alone", line=dict(color=RED, width=3))
    figure.add_hline(
        y=observed, line=dict(color=BLUE, dash="dash"),
        annotation_text="your strategy", annotation_position="top left",
    )
    figure.add_scatter(
        x=[trials], y=[bar], mode="markers", name="your search size",
        marker=dict(color=AMBER, size=14, line=dict(color="white", width=2)),
    )
    figure.update_layout(
        xaxis_type="log", xaxis_title="strategies searched (log)",
        yaxis_title="annualised Sharpe", height=420, margin=dict(t=30, b=40, l=8, r=8),
    )
    st.plotly_chart(figure, use_container_width=True)

# --- tab 3: live cost-sensitive threshold --------------------------------------------

with tab_cost:
    st.title("What does a missed failure cost you?")
    st.caption(
        "Real AI4I 2020 maintenance data, real gradient-boosting scores. 0.5 is a "
        "default, not a decision - the threshold that minimises cost depends on what a "
        "miss costs relative to a false alarm, and you set that here."
    )

    @st.cache_resource(show_spinner="Training on the real AI4I data (once)...")
    def fit_scores():
        import pandas as pd
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        sys.path.insert(0, str(ROOT / "projects" / "03-imbalanced-maintenance"))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "imb_run", ROOT / "projects" / "03-imbalanced-maintenance" / "run.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # defines load(); main() is __main__-guarded
        frame: pd.DataFrame = module.load()

        target = next(c for c in frame.columns if "Machine failure" in c or c == "failure")
        drop = [c for c in frame.columns if frame[c].dtype == object or c == target]
        X = frame.drop(columns=drop).select_dtypes("number")
        y = frame[target].astype(int).to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
        model.fit(Xtr, ytr)
        return yte, model.predict_proba(Xte)[:, 1]

    try:
        y_true, scores = fit_scores()
    except Exception as exc:  # the data file may be absent on a fresh clone
        st.warning(f"Could not load the maintenance dataset: {exc}")
        st.stop()

    from cost import evaluate_at  # noqa: E402

    left, right = st.columns(2)
    with left:
        cost_miss = st.number_input("Cost of a missed failure", 100, 100_000, 5000, step=100)
        cost_alarm = st.number_input("Cost of a false alarm", 10, 10_000, 100, step=10)
    with right:
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.0025)

    decision = evaluate_at(y_true, scores, threshold, cost_miss=cost_miss, cost_alarm=cost_alarm)
    grid = np.linspace(0.001, 0.999, 400)
    costs = [
        evaluate_at(y_true, scores, t, cost_miss=cost_miss, cost_alarm=cost_alarm).cost for t in grid
    ]
    best_index = int(np.argmin(costs))
    best_threshold, best_cost = float(grid[best_index]), float(costs[best_index])

    a, b, c, d = st.columns(4)
    a.metric("Cost at your threshold", f"{decision.cost:,.0f}")
    b.metric("Cheapest possible", f"{best_cost:,.0f}", delta=f"{best_cost - decision.cost:,.0f}")
    b.caption(f"at threshold {best_threshold:.4f}")
    recall = decision.true_positives / max(decision.true_positives + decision.false_negatives, 1)
    precision = decision.true_positives / max(decision.true_positives + decision.false_positives, 1)
    c.metric("Recall", f"{recall:.1%}")
    d.metric("Precision", f"{precision:.1%}")

    figure = go.Figure()
    figure.add_scatter(x=grid, y=costs, name="cost", line=dict(color=BLUE, width=3))
    figure.add_vline(
        x=threshold, line=dict(color=AMBER, dash="dash"), annotation_text="yours"
    )
    figure.add_scatter(
        x=[best_threshold], y=[best_cost], mode="markers", name="cheapest",
        marker=dict(color="#16a34a", size=14, line=dict(color="white", width=2)),
    )
    figure.update_layout(
        xaxis_title="threshold", yaxis_title="total cost", height=400,
        margin=dict(t=30, b=40, l=8, r=8),
    )
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Confusion at your threshold")
    matrix = [
        [decision.true_negatives, decision.false_positives],
        [decision.false_negatives, decision.true_positives],
    ]
    heat = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["predicted healthy", "predicted failure"],
            y=["actually healthy", "actually failed"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    heat.update_layout(height=300, margin=dict(t=10, b=10, l=8, r=8))
    st.plotly_chart(heat, use_container_width=True)
