"""Do the probabilities mean anything? 45,211 real phone calls, three models.

uv run python projects/12-calibration/run.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from calibrate import (  # noqa: E402
    Isotonic,
    Platt,
    Uncalibrated,
    auc,
    brier,
    evaluate,
    expected_calibration_error,
    money_at_stake,
    reliability,
)

from shared.data import BANK_MARKETING, fetch  # noqa: E402
from shared.plotting import (  # noqa: E402
    PALETTE,
    caption,
    money_axis,
    percent_axis,
    save,
    use_house_style,
)

FIGURES = HERE / "figures"
EXPOSURE = 8000.0  # what one account is worth, for the money calculation
LGD = 0.45  # loss given default


def load() -> pd.DataFrame:
    path = fetch(BANK_MARKETING, "uci-bank-marketing.zip")
    with zipfile.ZipFile(path) as outer:
        inner = next(n for n in outer.namelist() if n.endswith("bank.zip") or n.endswith(".csv"))
        if inner.endswith(".zip"):
            import io

            with zipfile.ZipFile(io.BytesIO(outer.read(inner))) as archive:
                target = next(n for n in archive.namelist() if n.endswith("bank-full.csv"))
                with archive.open(target) as handle:
                    return pd.read_csv(handle, sep=";")
        with outer.open(inner) as handle:
            return pd.read_csv(handle, sep=";")


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = load()
    frame.columns = [c.strip().strip('"') for c in frame.columns]
    y = (frame["y"].astype(str).str.strip().str.lower() == "yes").to_numpy()

    # `duration` is dropped: it does not exist at decision time. See project 05.
    features = frame.drop(columns=["y", "duration"])
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            features[column] = features[column].astype("category").cat.codes

    rng = np.random.default_rng(0)
    order = rng.permutation(len(frame))
    n = len(order)
    train, calib, test = (
        order[: int(n * 0.6)],
        order[int(n * 0.6) : int(n * 0.8)],
        order[int(n * 0.8) :],
    )
    print(f"\n{n:,} calls, {y.mean():.2%} subscribe")
    print(f"  train {len(train):,} | calibration {len(calib):,} | test {len(test):,}\n")

    X = features.to_numpy(dtype=float)
    scaler = StandardScaler().fit(X[train])

    models = {
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, random_state=0
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=4, random_state=0, n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
        "Logistic regression": LogisticRegression(max_iter=2000),
    }

    raw_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    print(f"  {'model':<22}{'AUC':>8}{'Brier':>10}{'ECE':>9}   probabilities are")
    for name, model in models.items():
        needs_scaling = name in {"Naive Bayes", "Logistic regression"}
        Xtr = scaler.transform(X[train]) if needs_scaling else X[train]
        model.fit(Xtr, y[train])

        def score(idx: np.ndarray, model=model, needs_scaling=needs_scaling) -> np.ndarray:
            Xi = scaler.transform(X[idx]) if needs_scaling else X[idx]
            return model.predict_proba(Xi)[:, 1]

        raw_scores[name] = (score(calib), score(test))
        s = raw_scores[name][1]
        ece = expected_calibration_error(y[test], s)
        # Direction, not just size: over-confident models sit below the diagonal.
        predicted, observed, counts = reliability(y[test], s)
        valid = counts > 0
        bias = float(np.average(predicted[valid] - observed[valid], weights=counts[valid]))
        direction = "too high" if bias > 0.005 else ("too low" if bias < -0.005 else "about right")
        print(
            f"  {name:<22}{auc(y[test], s):>8.4f}{brier(y[test], s):>10.5f}{ece:>9.4f}   {direction}"
        )

    worst = max(models, key=lambda m: expected_calibration_error(y[test], raw_scores[m][1]))
    print(f"\n  worst calibrated: {worst}")

    # --- the repair ---------------------------------------------------------
    scores_calib, scores_test = raw_scores[worst]
    reports, curves = [], {}
    for calibrator in (Uncalibrated(), Platt(), Isotonic()):
        report, calibrated = evaluate(calibrator, scores_calib, y[calib], scores_test, y[test])
        reports.append(report)
        curves[calibrator.name] = calibrated

    print(f"\n  === repairing {worst} ===")
    print(f"  {'method':<18}{'AUC':>8}{'Brier':>10}{'ECE':>9}{'worst bin':>11}")
    for report in reports:
        s = report.summary()
        print(
            f"  {s['method']:<18}{s['auc']:>8.4f}{s['brier']:>10.5f}"
            f"{s['ece']:>9.4f}{s['max_error']:>11.4f}"
        )

    unchanged = abs(reports[0].auc - reports[1].auc) < 0.02
    print(
        f"\n  AUC moves by {abs(reports[0].auc - reports[1].auc):.4f} — "
        f"{'unchanged, as it must be' if unchanged else 'note the change'}. "
        f"Calibration is monotone,\n  so it cannot reorder anything, and AUC only sees order."
    )

    # --- what it is worth ---------------------------------------------------
    print(f"\n  === at EUR {EXPOSURE:,.0f} exposure, {LGD:.0%} loss given default ===")
    money = {}
    for name, probabilities in curves.items():
        money[name] = money_at_stake(
            y[test], probabilities, exposure=EXPOSURE, loss_given_default=LGD
        )
        m = money[name]
        print(
            f"    {name:<18} provision EUR {m['predicted_loss']:>13,.0f}   "
            f"error {m['error_pct']:>+8.2%}"
        )

    # --- how much calibration data is enough? ------------------------------
    sizes = [100, 250, 500, 1000, 2500, 5000, len(calib)]
    curve_rows = []
    for size in sizes:
        take = calib[:size]
        for calibrator in (Platt(), Isotonic()):
            report, _ = evaluate(
                calibrator, raw_scores[worst][0][:size], y[take], scores_test, y[test]
            )
            curve_rows.append({"size": size, "method": calibrator.name, "ece": report.ece})
    sample_curve = pd.DataFrame(curve_rows)

    crossover = None
    for size in sizes:
        at = sample_curve[sample_curve["size"] == size].set_index("method")["ece"]
        if "Isotonic" in at and "Platt (sigmoid)" in at and at["Isotonic"] < at["Platt (sigmoid)"]:
            crossover = size
            break
    print(
        f"\n  Isotonic beats Platt from about {crossover or 'never in this range'} "
        f"calibration samples on."
    )

    # --- 1. the reliability diagram ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", ls="--", lw=1.3, label="perfect")
    for (name, probabilities), colour in zip(
        curves.items(), (PALETTE["red"], PALETTE["blue"], PALETTE["green"]), strict=True
    ):
        predicted, observed, counts = reliability(y[test], probabilities)
        valid = counts > 0
        ax.plot(
            predicted[valid],
            observed[valid],
            "o-",
            lw=1.9,
            ms=5,
            color=colour,
            label=f"{name} (ECE {expected_calibration_error(y[test], probabilities):.3f})",
        )
    ax.set_xlabel("Probability the model claimed")
    ax.set_ylabel("How often it actually happened")
    percent_axis(ax)
    percent_axis(ax, "x")
    ax.set_title(f"{worst}: what 0.70 actually means")
    ax.legend(loc="upper left", fontsize=8.5)
    caption(
        fig,
        f"UCI Bank Marketing, held-out {len(test):,} calls. Quantile bins, so each "
        "holds the same number of samples rather than the same width.",
    )
    print("\n  " + str(save(fig, FIGURES / "01-reliability.png").relative_to(HERE.parent.parent)))

    # --- 2. AUC is blind ----------------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    names = [r.method for r in reports]
    left.bar(names, [r.auc for r in reports], color=PALETTE["sky"], width=0.5)
    for i, r in enumerate(reports):
        left.text(i, r.auc + 0.004, f"{r.auc:.4f}", ha="center", fontweight="bold", fontsize=9)
    left.set_ylim(0, 1.0)
    left.set_title("AUC — identical")
    left.grid(axis="y")
    right.bar(names, [r.ece for r in reports], color=PALETTE["orange"], width=0.5)
    for i, r in enumerate(reports):
        right.text(i, r.ece + 0.001, f"{r.ece:.4f}", ha="center", fontweight="bold", fontsize=9)
    right.set_title("Calibration error — not")
    right.grid(axis="y")
    fig.suptitle(
        "Calibration is monotone, so it cannot change any ranking metric",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        "This is why a model can ship with a great AUC and probabilities nobody "
        "should multiply by money.",
    )
    print("  " + str(save(fig, FIGURES / "02-auc-is-blind.png").relative_to(HERE.parent.parent)))

    # --- 3. the money -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.3))
    labels = list(money.keys())
    predicted = [money[k]["predicted_loss"] for k in labels]
    actual = money[labels[0]]["actual_loss"]
    bars = ax.bar(
        labels, predicted, color=[PALETTE["red"], PALETTE["blue"], PALETTE["green"]], width=0.5
    )
    ax.axhline(actual, color="#444444", lw=2, ls="--")
    ax.text(
        len(labels) - 0.5,
        actual,
        f"  actual EUR {actual:,.0f}",
        va="bottom",
        ha="right",
        fontsize=9,
        fontweight="bold",
    )
    for bar, k in zip(bars, labels, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            money[k]["predicted_loss"],
            f"{money[k]['error_pct']:+.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    ax.set_ylabel(f"Provision on {len(test):,} accounts")
    money_axis(ax, symbol="EUR ")
    ax.set_title("Expected loss = P(default) x exposure x LGD. The P has to be real.")
    caption(
        fig,
        f"EUR {EXPOSURE:,.0f} exposure, {LGD:.0%} LGD. No accuracy metric moves when "
        "this number is wrong.",
    )
    print("  " + str(save(fig, FIGURES / "03-the-money.png").relative_to(HERE.parent.parent)))

    # --- 4. how much data does each method need? ---------------------------
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for method, colour in (("Platt (sigmoid)", PALETTE["blue"]), ("Isotonic", PALETTE["green"])):
        rows = sample_curve[sample_curve["method"] == method]
        ax.plot(rows["size"], rows["ece"], "o-", lw=2, color=colour, label=method)
    ax.axhline(reports[0].ece, color=PALETTE["red"], ls="--", lw=1.5, label="uncalibrated")
    ax.set_xscale("log")
    ax.set_xlabel("Calibration samples (log scale)")
    ax.set_ylabel("Expected calibration error on the test set")
    ax.legend()
    ax.set_title("Isotonic is more flexible, and needs more data to earn it")
    caption(
        fig,
        "Platt has two parameters and is hard to overfit. Isotonic can fit any "
        "monotone shape, including the noise in a small calibration set.",
    )
    print("  " + str(save(fig, FIGURES / "04-how-much-data.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "calls": int(n),
                "positive_rate": round(float(y.mean()), 4),
                "split": {"train": len(train), "calibration": len(calib), "test": len(test)},
                "raw_models": [
                    {
                        "model": name,
                        "auc": round(auc(y[test], s[1]), 4),
                        "brier": round(brier(y[test], s[1]), 5),
                        "ece": round(expected_calibration_error(y[test], s[1]), 5),
                    }
                    for name, s in raw_scores.items()
                ],
                "worst_calibrated": worst,
                "repairs": [r.summary() for r in reports],
                "money": money,
                "isotonic_beats_platt_from": crossover,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
