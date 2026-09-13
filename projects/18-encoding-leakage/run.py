"""Target encoding on a real high-cardinality column, and on a column of pure noise.

uv run python projects/18-encoding-leakage/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from encoding import ENCODERS, cardinality_profile  # noqa: E402

from shared.data import online_retail  # noqa: E402
from shared.plotting import PALETTE, caption, percent_axis, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
CUT = 0.75  # temporal split point
NUMERIC = ["log_price", "month", "weekday", "hour"]


def fit_and_score(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[float, float]:
    """Logistic regression, train AUC and test AUC.

    Both are reported deliberately. A leak that inflates only the training score produces a
    model that is genuinely worse; a leak that inflates both produces a number that is a lie.
    The gap between them is the diagnostic.
    """
    scaler = StandardScaler().fit(train[columns])
    model = LogisticRegression(max_iter=2000, C=1.0).fit(
        scaler.transform(train[columns]), train["y"]
    )
    return (
        float(
            roc_auc_score(train["y"], model.predict_proba(scaler.transform(train[columns]))[:, 1])
        ),
        float(roc_auc_score(test["y"], model.predict_proba(scaler.transform(test[columns]))[:, 1])),
    )


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = online_retail()
    frame = frame.sort_values("InvoiceDate").reset_index(drop=True)

    # Quantity itself is excluded: returns carry negative quantities, so it *is* the label.
    frame["y"] = frame["is_return"].astype(int)
    frame["log_price"] = np.log1p(frame["Price"])
    frame["month"] = frame["InvoiceDate"].dt.month
    frame["weekday"] = frame["InvoiceDate"].dt.dayofweek
    frame["hour"] = frame["InvoiceDate"].dt.hour
    frame["stock"] = frame["StockCode"]

    # A control column: random labels with the same number of distinct values as StockCode,
    # assigned with no relation to anything. Any predictive power it shows is pure leakage.
    rng = np.random.default_rng(0)
    n_codes = frame["stock"].nunique()
    frame["noise_id"] = rng.integers(0, n_codes, size=len(frame)).astype(str)

    cut = int(len(frame) * CUT)
    train, test = frame.iloc[:cut].copy(), frame.iloc[cut:].copy()

    print(f"\nOnline Retail II, {len(frame):,} invoice lines, {frame['y'].mean():.2%} returns")
    print(
        f"  temporal split: {len(train):,} train / {len(test):,} test at "
        f"{frame['InvoiceDate'].iloc[cut]:%Y-%m-%d}"
    )
    print(f"  StockCode has {n_codes:,} distinct values; noise_id has the same, at random\n")

    # --- the surface the leak acts on ---------------------------------------
    print("  === where target encoding is dangerous: the rare tail ===")
    profile = cardinality_profile(frame["stock"].to_numpy())
    print(f"  {'appearances':<14}{'stock codes':>13}{'rows':>10}{'share of rows':>15}")
    for _, row in profile.iterrows():
        print(
            f"  {row['appearances']:<14}{row['categories']:>13,}{row['rows']:>10,}"
            f"{row['share_of_rows']:>15.2%}"
        )
    rare = profile[profile["appearances"].isin(["1", "2-4"])]
    print(f"\n    {rare['categories'].sum():,} codes appear fewer than five times. For those, the")
    print("    'category mean' is mostly the row's own target with extra steps.\n")

    # --- baseline: no categorical at all ------------------------------------
    baseline_train, baseline_test = fit_and_score(train, test, NUMERIC)
    print("  === four ways to encode StockCode, and what each one claims ===")
    print(f"  {'encoding':<26}{'train AUC':>11}{'test AUC':>10}{'gap':>8}{'vs baseline':>13}")
    print(
        f"  {'(no StockCode at all)':<26}{baseline_train:>11.4f}{baseline_test:>10.4f}"
        f"{baseline_train - baseline_test:>8.4f}{'-':>13}"
    )

    rows = []
    for encoder in ENCODERS:
        train["te_stock"], test["te_stock"] = encoder.fit_transform(
            train["stock"].to_numpy(),
            train["y"].to_numpy(),
            test["stock"].to_numpy(),
            test["y"].to_numpy(),
        )
        a, b = fit_and_score(train, test, [*NUMERIC, "te_stock"])
        rows.append(
            {
                "encoding": encoder.name,
                "train_auc": a,
                "test_auc": b,
                "gap": a - b,
                "lift": b - baseline_test,
            }
        )
        print(f"  {encoder.name:<26}{a:>11.4f}{b:>10.4f}{a - b:>8.4f}{b - baseline_test:>+13.4f}")

    naive = next(r for r in rows if r["encoding"].startswith("Naive"))
    correct = next(r for r in rows if r["encoding"].startswith("Out-of-fold"))
    print(f"\n    The naive encoder reports a training AUC of {naive['train_auc']:.4f}.")
    print(f"    Done correctly the same column is worth {correct['lift']:+.4f} on the test set.\n")

    # --- the control: encoding pure noise -----------------------------------
    print("  === the same four encoders on a column that is definitionally worthless ===")
    print(f"  {'encoding':<26}{'train AUC':>11}{'test AUC':>10}{'gap':>8}")
    noise_rows = []
    for encoder in ENCODERS:
        train["te_noise"], test["te_noise"] = encoder.fit_transform(
            train["noise_id"].to_numpy(),
            train["y"].to_numpy(),
            test["noise_id"].to_numpy(),
            test["y"].to_numpy(),
        )
        a, b = fit_and_score(train, test, [*NUMERIC, "te_noise"])
        noise_rows.append({"encoding": encoder.name, "train_auc": a, "test_auc": b, "gap": a - b})
        print(f"  {encoder.name:<26}{a:>11.4f}{b:>10.4f}{a - b:>8.4f}")

    noise_naive = noise_rows[0]
    print(
        f"\n    A column of random integers, target-encoded naively, trains to "
        f"AUC {noise_naive['train_auc']:.4f}."
    )
    print(
        f"    It carries no information. Every point above {baseline_train:.4f} is copied labels.\n"
    )

    # --- how the leak scales with rarity ------------------------------------
    print("  === the leak is concentrated in rare categories ===")
    counts = train["stock"].value_counts()
    naive_encoder, oof_encoder = ENCODERS[0], ENCODERS[-1]
    naive_tr, _ = naive_encoder.fit_transform(
        train["stock"].to_numpy(),
        train["y"].to_numpy(),
        test["stock"].to_numpy(),
        test["y"].to_numpy(),
    )
    oof_tr, _ = oof_encoder.fit_transform(
        train["stock"].to_numpy(),
        train["y"].to_numpy(),
        test["stock"].to_numpy(),
        test["y"].to_numpy(),
    )

    train_counts = train["stock"].map(counts).to_numpy()
    buckets = [(1, 1), (2, 4), (5, 19), (20, 99), (100, 10**9)]
    labels = ["1", "2-4", "5-19", "20-99", "100+"]
    leak_rows = []
    print(f"  {'appearances':<14}{'naive corr':>13}{'out-of-fold corr':>19}{'rows':>10}")
    for (low, high), label in zip(buckets, labels, strict=True):
        mask = (train_counts >= low) & (train_counts <= high)
        if mask.sum() < 30 or train["y"].to_numpy()[mask].std() == 0:
            continue
        y = train["y"].to_numpy()[mask]
        naive_corr = float(np.corrcoef(naive_tr[mask], y)[0, 1])
        oof_corr = float(np.corrcoef(oof_tr[mask], y)[0, 1]) if np.std(oof_tr[mask]) > 0 else 0.0
        leak_rows.append(
            {
                "appearances": label,
                "naive": naive_corr,
                "out_of_fold": oof_corr,
                "rows": int(mask.sum()),
            }
        )
        print(f"  {label:<14}{naive_corr:>13.3f}{oof_corr:>19.3f}{mask.sum():>10,}")

    print("\n    Correlation between the encoded feature and the target, within each bucket.")
    print("    At one appearance the naive encoding *is* the target. At 100+ it is a real")
    print("    statistic and the two methods agree.")

    # --- figure 1: cardinality ----------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.bar(profile["appearances"], profile["categories"], color=PALETTE["blue"], width=0.6)
    left.set_ylabel("Distinct stock codes")
    left.set_xlabel("Times the code appears")
    left.set_title(f"{n_codes:,} categories, most of them rare")
    right.bar(profile["appearances"], profile["share_of_rows"], color=PALETTE["orange"], width=0.6)
    right.set_ylabel("Share of all rows")
    right.set_xlabel("Times the code appears")
    percent_axis(right)
    right.set_title("but the rare ones are a small slice of the data")
    fig.suptitle(
        "Why one-hot fails here, and why the alternative is dangerous",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        "A category seen once has a 'category mean' equal to its own target. That is "
        "the leak, and it lives entirely in the left-hand bars.",
    )
    print("\n  " + str(save(fig, FIGURES / "01-cardinality.png").relative_to(HERE.parent.parent)))

    # --- figure 2: train vs test -------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["train_auc"] for r in rows], 0.38, label="train AUC", color=PALETTE["red"])
    ax.bar(
        x + 0.2,
        [r["test_auc"] for r in rows],
        0.38,
        label="test AUC (the truth)",
        color=PALETTE["blue"],
    )
    ax.axhline(baseline_test, color="#333333", ls="--", lw=1.5)
    ax.text(
        len(rows) - 0.55,
        baseline_test + 0.008,
        "no StockCode at all",
        fontsize=8,
        ha="right",
        color="#333333",
    )
    for i, r in enumerate(rows):
        ax.text(
            i - 0.2,
            r["train_auc"] + 0.006,
            f"{r['train_auc']:.3f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(i + 0.2, r["test_auc"] + 0.006, f"{r['test_auc']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [r["encoding"].replace(" (", "\n(") for r in rows], fontsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Same column, same model, four encodings")
    caption(
        fig,
        f"Online Retail II, temporal split, {frame['y'].mean():.2%} return rate. The "
        f"red-blue gap is the part of the training signal that does not exist.",
    )
    print("  " + str(save(fig, FIGURES / "02-four-encodings.png").relative_to(HERE.parent.parent)))

    # --- figure 3: the noise control ---------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(noise_rows))
    ax.bar(
        x - 0.2, [r["train_auc"] for r in noise_rows], 0.38, label="train AUC", color=PALETTE["red"]
    )
    ax.bar(
        x + 0.2, [r["test_auc"] for r in noise_rows], 0.38, label="test AUC", color=PALETTE["blue"]
    )
    ax.axhline(baseline_train, color="#333333", ls="--", lw=1.5)
    for i, r in enumerate(noise_rows):
        ax.text(
            i - 0.2,
            r["train_auc"] + 0.006,
            f"{r['train_auc']:.3f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xticks(x, [r["encoding"].replace(" (", "\n(") for r in noise_rows], fontsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("A column of random integers, target-encoded")
    caption(
        fig,
        "The column is generated by a random number generator and has no relationship "
        "to anything. The dashed line is the model without it. Everything above the "
        "line is copied labels.",
    )
    print("  " + str(save(fig, FIGURES / "03-noise-control.png").relative_to(HERE.parent.parent)))

    # --- figure 4: leak by rarity ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(leak_rows))
    ax.bar(
        x - 0.2, [r["naive"] for r in leak_rows], 0.38, label="Naive encoding", color=PALETTE["red"]
    )
    ax.bar(
        x + 0.2,
        [r["out_of_fold"] for r in leak_rows],
        0.38,
        label="Out-of-fold",
        color=PALETTE["blue"],
    )
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x, [f"{r['appearances']}\n({r['rows']:,} rows)" for r in leak_rows], fontsize=8)
    ax.set_xlabel("Times the stock code appears in the training set")
    ax.set_ylabel("Correlation with the target")
    ax.legend(fontsize=8)
    ax.set_title("The leak is not uniform: it is entirely in the tail")
    caption(
        fig,
        "Within each rarity bucket, how strongly the encoded feature correlates with "
        "the label it was built from. Common categories are fine either way.",
    )
    print("  " + str(save(fig, FIGURES / "04-by-rarity.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "rows": int(len(frame)),
                "return_rate": round(float(frame["y"].mean()), 5),
                "distinct_stock_codes": int(n_codes),
                "baseline": {
                    "train_auc": round(baseline_train, 4),
                    "test_auc": round(baseline_test, 4),
                },
                "stock_code": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows
                ],
                "noise_control": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in noise_rows
                ],
                "leak_by_rarity": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in leak_rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
