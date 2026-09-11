"""The `duration` leak, on 45,211 real phone calls.

uv run python projects/05-leakage/run.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent / "03-imbalanced-maintenance"))

from cost import auc, roc_curve  # noqa: E402
from leak import LEAKY, audit_columns, expected_campaign_value, lift_at_k  # noqa: E402

from shared.data import BANK_MARKETING, fetch  # noqa: E402
from shared.plotting import (  # noqa: E402
    PALETTE,
    annotate,
    caption,
    money_axis,
    save,
    use_house_style,
)

FIGURES = HERE / "figures"

# A term deposit is worth this much to the bank; a call costs this much to make.
VALUE_PER_SUBSCRIPTION = 80.0
COST_PER_CALL = 2.0
CALL_BUDGET = 0.10  # the campaign can phone the top 10% of the list

# Everything a call-centre system knows *before* dialling: who the customer is, what the
# bank already knows about them, and what happened in previous campaigns.
KNOWN_BEFORE_DIALLING = {
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
}


def load() -> pd.DataFrame:
    path = fetch(BANK_MARKETING, "uci-bank-marketing.zip")
    with zipfile.ZipFile(path) as outer:
        inner_name = next(
            n for n in outer.namelist() if n.endswith("bank.zip") or n.endswith(".csv")
        )
        if inner_name.endswith(".zip"):
            import io

            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
                target = next(n for n in inner.namelist() if n.endswith("bank-full.csv"))
                with inner.open(target) as handle:
                    return pd.read_csv(handle, sep=";")
        with outer.open(inner_name) as handle:
            return pd.read_csv(handle, sep=";")


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = load()
    frame.columns = [c.strip().strip('"') for c in frame.columns]
    y = (frame["y"].astype(str).str.strip().str.lower() == "yes").to_numpy()
    n = len(frame)
    print(f"\n{n:,} phone calls, {y.sum():,} subscriptions ({y.mean():.2%})\n")

    features = frame.drop(columns=["y"])
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            features[column] = features[column].astype("category").cat.codes

    # --- the audit ---------------------------------------------------------
    report = audit_columns(features, y, known_at_decision_time=KNOWN_BEFORE_DIALLING)
    print(report.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    flagged = report[report["verdict"].str.startswith("LEAK")]
    print(f"\n  columns unavailable at decision time: {list(flagged['column'])}")

    # --- two models, identical except for one column ----------------------
    cut = int(n * 0.7)
    honest_columns = [c for c in features.columns if c not in LEAKY]

    def fit_score(columns: list[str]) -> np.ndarray:
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=0)
        model.fit(features.iloc[:cut][columns], y[:cut])
        return model.predict_proba(features.iloc[cut:][columns])[:, 1]

    scores = {
        "With `duration` (leaky)": fit_score(list(features.columns)),
        "Without `duration` (honest)": fit_score(honest_columns),
    }
    yte = y[cut:]

    print(f"\n  train {cut:,} calls | test {n - cut:,} calls ({yte.mean():.2%} subscribe)\n")
    rows = []
    for name, s in scores.items():
        fpr, tpr = roc_curve(yte, s)
        rows.append(
            {
                "model": name,
                "roc_auc": auc(fpr, tpr),
                "lift@10%": lift_at_k(yte, s, k=CALL_BUDGET),
                **{
                    f"campaign_{k}": v
                    for k, v in expected_campaign_value(
                        yte,
                        s,
                        k=CALL_BUDGET,
                        value_per_subscription=VALUE_PER_SUBSCRIPTION,
                        cost_per_call=COST_PER_CALL,
                    ).items()
                },
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    gap = table.loc[0, "roc_auc"] - table.loc[1, "roc_auc"]
    print(
        f"\n  One column is worth {gap:+.3f} ROC-AUC - and cannot be used, because to know"
        f"\n  the call lasted eleven minutes you must first have made an eleven-minute call."
    )

    # --- the second leak, found while checking the first ------------------
    #
    # The held-out slice subscribes at 25.4% against 11.7% overall. The rows are in
    # chronological order and the campaign changed, so a *random* split mixes those two
    # regimes on both sides and every fold gets to see the future. That is why the number
    # usually quoted for this dataset is around 0.93 and a time-ordered split gives 0.73.
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(n)
    rand_tr, rand_te = shuffled[:cut], shuffled[cut:]

    def fit_score_indices(columns: list[str], train_idx, test_idx) -> np.ndarray:
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=0)
        model.fit(features.iloc[train_idx][columns], y[train_idx])
        return model.predict_proba(features.iloc[test_idx][columns])[:, 1]

    split_comparison = []
    for split_name, (tr, te) in (
        ("random split", (rand_tr, rand_te)),
        ("time-ordered split", (np.arange(cut), np.arange(cut, n))),
    ):
        for label, columns in (
            ("with duration", list(features.columns)),
            ("without duration", honest_columns),
        ):
            s = fit_score_indices(columns, tr, te)
            fpr, tpr = roc_curve(y[te], s)
            split_comparison.append(
                {
                    "split": split_name,
                    "features": label,
                    "roc_auc": auc(fpr, tpr),
                    "test_positive_rate": float(y[te].mean()),
                }
            )

    splits = pd.DataFrame(split_comparison)
    print("\n  The same two models under two ways of splitting the data:\n")
    print(splits.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"\n  Overall subscription rate {y.mean():.2%}; the last 30% of rows subscribe at "
        f"{y[cut:].mean():.2%}.\n  A random split puts both regimes on both sides, which is a "
        f"leak of its own."
    )

    # --- 1. the two ROC curves --------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for (name, s), colour in zip(scores.items(), (PALETTE["red"], PALETTE["blue"]), strict=True):
        fpr, tpr = roc_curve(yte, s)
        ax.plot(fpr, tpr, lw=2.1, color=colour, label=f"{name} — AUC {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", ls="--", lw=1.2)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("The same model, one column apart")
    ax.legend(loc="lower right")
    caption(
        fig,
        f"UCI Bank Marketing, held-out {n - cut:,} calls. `duration` is flagged as "
        "unusable by the dataset's own documentation.",
    )
    print(
        "\n  " + str(save(fig, FIGURES / "01-one-column-apart.png").relative_to(HERE.parent.parent))
    )

    # --- 2. the audit table as a chart ------------------------------------
    view = report.head(10).iloc[::-1]
    colours = [
        PALETTE["red"]
        if v.startswith("LEAK")
        else (PALETTE["orange"] if v.startswith("check") else PALETTE["sky"])
        for v in view["verdict"]
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(view["column"], view["solo_auc"], color=colours)
    ax.axvline(0.5, color="#999999", ls="--", lw=1.2)
    for yy, (_, r) in enumerate(view.iterrows()):
        ax.text(
            r["solo_auc"] + 0.008,
            yy,
            f"{r['solo_auc']:.3f}",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlabel("AUC of that column alone")
    ax.set_xlim(0, 0.95)
    ax.set_title("One column separates the classes almost by itself. Ask when it is recorded.")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        "Red = not knowable at decision time. The colour cannot be computed from the "
        "data; it comes from knowing how the data was collected.",
    )
    print("  " + str(save(fig, FIGURES / "02-column-audit.png").relative_to(HERE.parent.parent)))

    # --- 3. the money -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    labels = list(scores.keys())
    profits = [r["campaign_profit"] for r in rows]
    bars = ax.bar(labels, profits, color=[PALETTE["red"], PALETTE["blue"]], width=0.5)
    for bar, r in zip(bars, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r["campaign_profit"],
            f"  GBP {r['campaign_profit']:,.0f}\n  {r['campaign_subscriptions']} sales "
            f"from {r['campaign_calls']:,} calls",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    ax.set_ylabel(f"Campaign profit, top {CALL_BUDGET:.0%} of the list")
    money_axis(ax, symbol="GBP ")
    ax.set_title("What the leak promises, and what it can deliver: nothing")
    caption(
        fig,
        f"Deposit worth GBP {VALUE_PER_SUBSCRIPTION:.0f}, call costs GBP {COST_PER_CALL:.0f}. "
        "The leaky model cannot rank a list of people who have not been called yet.",
    )
    print("  " + str(save(fig, FIGURES / "03-campaign-value.png").relative_to(HERE.parent.parent)))

    # --- 4. why it leaks, visibly -----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.3))
    bins = np.linspace(0, 1200, 70)
    ax.hist(
        frame.loc[~y, "duration"],
        bins=bins,
        color=PALETTE["sky"],
        alpha=0.75,
        label="did not subscribe",
        density=True,
    )
    ax.hist(
        frame.loc[y, "duration"],
        bins=bins,
        color=PALETTE["orange"],
        alpha=0.7,
        label="subscribed",
        density=True,
    )
    ax.axvline(float(frame.loc[~y, "duration"].median()), color=PALETTE["blue"], lw=2)
    ax.axvline(float(frame.loc[y, "duration"].median()), color=PALETTE["red"], lw=2)
    annotate(
        ax,
        float(frame.loc[y, "duration"].median()),
        ax.get_ylim()[1] * 0.7,
        f"  median {frame.loc[y, 'duration'].median():.0f}s",
        color=PALETTE["red"],
    )
    annotate(
        ax,
        float(frame.loc[~y, "duration"].median()),
        ax.get_ylim()[1] * 0.9,
        f"  median {frame.loc[~y, 'duration'].median():.0f}s",
        color=PALETTE["blue"],
    )
    ax.set_xlabel("Call duration (seconds)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("People who say yes stay on the phone. That is the outcome, not a predictor.")
    caption(fig, "A long call and a subscription are the same event observed twice.")
    print("  " + str(save(fig, FIGURES / "04-why-it-leaks.png").relative_to(HERE.parent.parent)))

    # --- 5. two leaks, side by side ---------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.36
    x = np.arange(2)
    for i, label in enumerate(("with duration", "without duration")):
        vals = [
            float(splits[(splits["split"] == s) & (splits["features"] == label)]["roc_auc"].iloc[0])
            for s in ("random split", "time-ordered split")
        ]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            color=(PALETTE["red"] if i == 0 else PALETTE["blue"]),
            label=label,
        )
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.006,
                f"{v:.3f}",
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
    ax.set_xticks(x, ["Random split\n(mixes past and future)", "Time-ordered split\n(honest)"])
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.set_title("Two leaks compound: a column you will not have, and a split that shuffles time")
    caption(
        fig,
        f"UCI Bank Marketing, n={n:,}. Rows are chronological; the final 30% subscribe "
        f"at {y[cut:].mean():.1%} against {y.mean():.1%} overall.",
    )
    print("  " + str(save(fig, FIGURES / "05-two-leaks.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "calls": int(n),
                "subscriptions": int(y.sum()),
                "subscription_rate": round(float(y.mean()), 4),
                "flagged_columns": list(flagged["column"]),
                "auc_gap_from_one_column": round(float(gap), 4),
                "overall_positive_rate": round(float(y.mean()), 4),
                "holdout_positive_rate": round(float(y[cut:].mean()), 4),
                "split_comparison": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in split_comparison
                ],
                "models": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
