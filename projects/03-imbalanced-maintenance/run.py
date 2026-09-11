"""Predicting machine failure on 10,000 real cycles, where 96.6% accuracy is worthless.

uv run python projects/03-imbalanced-maintenance/run.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from cost import (  # noqa: E402
    auc,
    average_precision,
    best_threshold,
    cost_curve,
    evaluate_at,
    precision_recall_curve,
    roc_curve,
)

from shared.data import AI4I_MAINTENANCE, fetch  # noqa: E402
from shared.plotting import (  # noqa: E402
    PALETTE,
    annotate,
    caption,
    money_axis,
    save,
    use_house_style,
)

FIGURES = HERE / "figures"

# What the two mistakes actually cost. Stated here, in money, at the top of the file,
# because every threshold decision below follows from this ratio and nothing else.
COST_MISS = 5000.0  # an unplanned stoppage: a lost production shift
COST_ALARM = 100.0  # an unnecessary inspection: an engineer, an hour
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load() -> pd.DataFrame:
    path = fetch(AI4I_MAINTENANCE, "ai4i-2020-maintenance.zip")
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, encoding="utf-8-sig")
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = load()
    n = len(frame)
    failures = int(frame["Machine failure"].sum())
    print(f"\n{n:,} machine cycles, {failures} failures ({failures / n:.2%})\n")

    # --- the label is not quite what it says ------------------------------
    modes = frame[FAILURE_MODES].sum(axis=1) > 0
    labelled = frame["Machine failure"].astype(bool)
    mode_no_label = int((modes & ~labelled).sum())
    label_no_mode = int((~modes & labelled).sum())
    print(f"  rows with a failure mode flagged but 'Machine failure' = 0 : {mode_no_label}")
    print(f"  rows labelled failed with no mode flagged                 : {label_no_mode}")
    print(f"  of those, random failures (RNF) = {int(frame.loc[modes & ~labelled, 'RNF'].sum())}\n")

    # --- features ---------------------------------------------------------
    numeric = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    X = frame[numeric].copy()
    # Machine quality grade L/M/H is ordinal, not nominal - H is the better casting.
    X["quality"] = frame["Type"].map({"L": 0, "M": 1, "H": 2}).astype(float)
    # Power is torque x angular velocity. The physics is known, so the feature is given
    # rather than left for a tree to rediscover from two columns.
    X["power_w"] = frame["Torque [Nm]"] * frame["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["temp_gap"] = frame["Process temperature [K]"] - frame["Air temperature [K]"]
    y = labelled.to_numpy()

    # Split by position, not at random: these are sequential cycles of a process, and a
    # shuffled split lets the model see the future of a wear curve it is meant to predict.
    cut = int(n * 0.7)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y[:cut], y[cut:]
    print(
        f"  train {len(Xtr):,} cycles ({ytr.sum()} failures) | test {len(Xte):,} ({yte.sum()} failures)"
    )

    scaler = StandardScaler().fit(Xtr)
    linear = LogisticRegression(max_iter=2000, class_weight="balanced")
    linear.fit(scaler.transform(Xtr), ytr)
    trees = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
    trees.fit(Xtr, ytr)

    models = {
        "Never predict failure": np.zeros(len(yte)),
        "Logistic regression": linear.predict_proba(scaler.transform(Xte))[:, 1],
        "Gradient boosting": trees.predict_proba(Xte)[:, 1],
    }

    # --- the table that makes the point -----------------------------------
    rows = []
    for name, scores in models.items():
        at_half = evaluate_at(yte, scores, 0.5, cost_miss=COST_MISS, cost_alarm=COST_ALARM)
        fpr, tpr = roc_curve(yte, scores)
        rows.append(
            {
                "model": name,
                "accuracy@0.5": at_half.accuracy,
                "recall@0.5": at_half.recall,
                "roc_auc": auc(fpr, tpr) if scores.std() else 0.5,
                "avg_precision": average_precision(yte, scores)
                if scores.std()
                else float(yte.mean()),
                "cost@0.5": at_half.cost,
            }
        )
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    best_name = "Gradient boosting"
    scores = models[best_name]
    default = evaluate_at(yte, scores, 0.5, cost_miss=COST_MISS, cost_alarm=COST_ALARM)
    tuned = best_threshold(yte, scores, cost_miss=COST_MISS, cost_alarm=COST_ALARM)

    print(
        f"\n  Cost of a miss ${COST_MISS:,.0f} vs a false alarm ${COST_ALARM:,.0f} (ratio {COST_MISS / COST_ALARM:.0f}:1)"
    )
    print(f"    threshold 0.50 (default) : {json.dumps(default.summary())}")
    print(f"    threshold {tuned.threshold:.3f} (chosen)  : {json.dumps(tuned.summary())}")
    saved = default.cost - tuned.cost
    print(
        f"    moving the threshold saves ${saved:,.0f} on {len(yte):,} cycles "
        f"({saved / default.cost:.1%} of the cost)"
    )

    # --- 1. accuracy is a lie ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4))
    order = table.sort_values("accuracy@0.5")
    bars = ax.barh(order["model"], order["accuracy@0.5"], color=PALETTE["sky"])
    for bar, (_, r) in zip(bars, order.iterrows(), strict=True):
        ax.text(
            r["accuracy@0.5"] - 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{r['accuracy@0.5']:.1%}   recall {r['recall@0.5']:.0%}",
            va="center",
            fontweight="bold",
            fontsize=9,
            color="white",
        )
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Accuracy")
    ax.set_title("All three look excellent. One of them never predicts a failure.")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        f"UCI AI4I 2020, held-out {len(yte):,} cycles with {int(yte.sum())} failures "
        f"({yte.mean():.1%}). Accuracy's floor here is 96.6%.",
    )
    print(
        "\n  "
        + str(save(fig, FIGURES / "01-accuracy-is-a-lie.png").relative_to(HERE.parent.parent))
    )

    # --- 2. ROC flatters, PR tells the truth ------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for name, colour in (("Logistic regression", "blue"), ("Gradient boosting", "orange")):
        fpr, tpr = roc_curve(yte, models[name])
        left.plot(fpr, tpr, color=PALETTE[colour], lw=1.9, label=f"{name} ({auc(fpr, tpr):.3f})")
        precision, recall, _ = precision_recall_curve(yte, models[name])
        right.plot(
            recall,
            precision,
            color=PALETTE[colour],
            lw=1.9,
            label=f"{name} ({average_precision(yte, models[name]):.3f})",
        )
    left.plot([0, 1], [0, 1], color="#bbbbbb", ls="--", lw=1.2)
    left.set_xlabel("False positive rate")
    left.set_ylabel("True positive rate")
    left.set_title("ROC — floor is 0.500")
    left.legend(loc="lower right", fontsize=8)
    right.axhline(yte.mean(), color="#bbbbbb", ls="--", lw=1.2)
    right.text(
        0.52,
        yte.mean() + 0.02,
        f"floor = base rate = {yte.mean():.3f}",
        fontsize=8,
        color="#888888",
    )
    right.set_xlabel("Recall")
    right.set_ylabel("Precision")
    right.set_ylim(0, 1.02)
    right.set_title("Precision-recall — floor is 0.034")
    right.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "The same two models, judged by two curves",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        "ROC divides false positives by 2,900 true negatives, so the curve barely moves. "
        "Precision divides them by the alarms raised, which is what an engineer experiences.",
    )
    print("  " + str(save(fig, FIGURES / "02-roc-vs-pr.png").relative_to(HERE.parent.parent)))

    # --- 3. the cost curve -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.4))
    thresholds, costs = cost_curve(yte, scores, cost_miss=COST_MISS, cost_alarm=COST_ALARM)
    ax.plot(thresholds, costs, color=PALETTE["blue"], lw=2)
    ax.axvline(0.5, color="#999999", ls="--", lw=1.4)
    ax.axvline(tuned.threshold, color=PALETTE["green"], lw=2)
    annotate(ax, 0.5, default.cost, f"  default 0.50\n  ${default.cost:,.0f}", color="#666666")
    annotate(
        ax,
        tuned.threshold,
        tuned.cost,
        f"  chosen {tuned.threshold:.3f}\n  ${tuned.cost:,.0f}",
        dx=-96,
        dy=-34,
        color=PALETTE["green"],
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Total cost on {len(yte):,} cycles")
    money_axis(ax)
    ax.set_title(
        f"A miss costs {COST_MISS / COST_ALARM:.0f} false alarms, so the threshold belongs nowhere near 0.5"
    )
    caption(
        fig,
        f"Miss ${COST_MISS:,.0f} (lost production shift), false alarm ${COST_ALARM:,.0f} "
        f"(one engineer-hour). Change the ratio and the optimum moves.",
    )
    print("  " + str(save(fig, FIGURES / "03-cost-curve.png").relative_to(HERE.parent.parent)))

    # --- 4. what the move buys, in units an engineer recognises -----------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = ["Missed failures", "False alarms"]
    width = 0.36
    x = np.arange(2)
    ax.bar(
        x - width / 2,
        [default.false_negatives, default.false_positives],
        width,
        label="threshold 0.50",
        color="#9a9a9a",
    )
    ax.bar(
        x + width / 2,
        [tuned.false_negatives, tuned.false_positives],
        width,
        label=f"threshold {tuned.threshold:.3f}",
        color=PALETTE["green"],
    )
    for i, (a, b) in enumerate(
        [
            (default.false_negatives, tuned.false_negatives),
            (default.false_positives, tuned.false_positives),
        ]
    ):
        ax.text(i - width / 2, a + 0.5, str(a), ha="center", fontweight="bold", fontsize=9)
        ax.text(i + width / 2, b + 0.5, str(b), ha="center", fontweight="bold", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Trading false alarms for caught failures, on purpose")
    caption(
        fig,
        f"Held-out {len(yte):,} cycles. The trade is only correct because the cost ratio was stated first.",
    )
    print("  " + str(save(fig, FIGURES / "04-what-it-buys.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "cycles": n,
                "failures": failures,
                "failure_rate": round(failures / n, 4),
                "label_inconsistencies": {
                    "mode_flagged_not_labelled": mode_no_label,
                    "labelled_no_mode": label_no_mode,
                },
                "cost_miss": COST_MISS,
                "cost_alarm": COST_ALARM,
                "models": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows
                ],
                "default_threshold": default.summary(),
                "chosen_threshold": tuned.summary(),
                "cost_saved": round(saved, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
