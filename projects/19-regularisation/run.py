"""Lasso, ridge and elastic net on correlated sensors: same score, different story.

    uv run python projects/19-regularisation/run.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from regularise import (  # noqa: E402
    bootstrap_selection,
    fit_penalised,
    pairwise_stability,
    regularisation_path,
)

from shared.data import AI4I_MAINTENANCE, fetch  # noqa: E402
from shared.plotting import PALETTE, caption, percent_axis, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
C_CHOSEN = 0.1  # penalty strength used for every stability experiment
REPLICATES = 200


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """The features a maintenance engineer would actually build - correlated on purpose.

    Nothing here is contrived. Power is torque times angular velocity, the temperature
    *difference* is what drives heat dissipation, and wear-rate proxies are standard. They
    are also near-duplicates of each other and of their inputs, which is exactly the
    situation every real feature set is in.
    """
    out = pd.DataFrame(index=frame.index)
    air = frame["Air temperature [K]"]
    process = frame["Process temperature [K]"]
    speed = frame["Rotational speed [rpm]"]
    torque = frame["Torque [Nm]"]
    wear = frame["Tool wear [min]"]

    out["air_temp"] = air
    out["process_temp"] = process
    out["temp_difference"] = process - air
    out["temp_ratio"] = process / air
    out["speed"] = speed
    out["torque"] = torque
    out["power"] = torque * speed * 2 * np.pi / 60
    out["log_power"] = np.log1p(out["power"])
    out["tool_wear"] = wear
    out["wear_x_torque"] = wear * torque
    out["wear_squared"] = wear**2
    out["torque_per_rpm"] = torque / speed
    out["is_low_quality"] = (frame["Type"] == "L").astype(float)
    out["is_high_quality"] = (frame["Type"] == "H").astype(float)
    return out


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    with zipfile.ZipFile(fetch(AI4I_MAINTENANCE, "ai4i-2020-maintenance.zip")) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, encoding="utf-8-sig")
    frame.columns = [c.strip() for c in frame.columns]

    features = engineer(frame)
    columns = list(features.columns)
    y = frame["Machine failure"].astype(int).to_numpy()

    cut = int(len(frame) * 0.7)
    scaler = StandardScaler().fit(features.iloc[:cut])
    X = scaler.transform(features)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    print(f"\nUCI AI4I 2020: {len(frame):,} cycles, {y.mean():.2%} failures")
    print(f"  {len(columns)} engineered features from 5 sensors, correlated by construction\n")

    # --- how correlated, exactly? -------------------------------------------
    correlation = features.corr().abs()
    correlation = correlation.mask(np.eye(len(columns), dtype=bool), 0.0)
    pairs = (
        correlation.where(np.triu(np.ones(correlation.shape), 1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    print("  === the correlated pairs the penalty has to choose between ===")
    for (a, b), value in pairs.head(6).items():
        print(f"    {a:<18} {b:<18} |r| = {value:.3f}")
    print()

    # --- three penalties, one score -----------------------------------------
    print("  === three penalties, and how little the score can tell them apart ===")
    fits = [
        fit_penalised(X_train, y_train, X_test, y_test, columns,
                      name="Ridge (L2)", penalty="l2", C=C_CHOSEN),
        fit_penalised(X_train, y_train, X_test, y_test, columns,
                      name="Lasso (L1)", penalty="l1", C=C_CHOSEN),
        fit_penalised(X_train, y_train, X_test, y_test, columns,
                      name="Elastic net (0.5)", penalty="elasticnet", C=C_CHOSEN, l1_ratio=0.5),
        fit_penalised(X_train, y_train, X_test, y_test, columns,
                      name="No penalty", penalty=None, C=1.0),
    ]
    print(f"  {'model':<20}{'train AUC':>11}{'test AUC':>10}{'features kept':>16}")
    for fit in fits:
        print(f"  {fit.name:<20}{fit.train_auc:>11.4f}{fit.test_auc:>10.4f}"
              f"{fit.n_selected:>12} / {len(columns)}")

    spread = max(f.test_auc for f in fits) - min(f.test_auc for f in fits)
    lasso = next(f for f in fits if f.name.startswith("Lasso"))
    print(f"\n    Test AUC spread across all four: {spread:.4f}. The lasso keeps "
          f"{lasso.n_selected} of {len(columns)}\n    features and scores the same as the "
          f"model that keeps every one.\n")

    # --- the path -----------------------------------------------------------
    strengths = np.logspace(-3, 1.5, 22)
    path = regularisation_path(X_train, y_train, X_test, y_test, columns, strengths=strengths)
    per_strength = path.groupby("C").first().reset_index()
    best = per_strength.loc[per_strength["test_auc"].idxmax()]
    print("  === the path: when does dropping features start to cost anything? ===")
    print(f"  {'C':>10}{'kept':>7}{'train AUC':>12}{'test AUC':>11}")
    for _, row in per_strength.iloc[::3].iterrows():
        print(f"  {row['C']:>10.4f}{int(row['n_selected']):>7}{row['train_auc']:>12.4f}"
              f"{row['test_auc']:>11.4f}")
    print(f"\n    best test AUC {best['test_auc']:.4f} at C={best['C']:.4f} with "
          f"{int(best['n_selected'])} features.\n")

    # --- stability ----------------------------------------------------------
    print(f"  === {REPLICATES} bootstrap refits: which features are actually there? ===")
    stability = bootstrap_selection(
        X_train, y_train, columns, C=C_CHOSEN, replicates=REPLICATES, seed=3
    )
    print(f"  {'feature':<20}{'selected':>10}{'sign agrees':>14}")
    for _, row in stability.iterrows():
        flag = "  <- coin flip" if row["coin_flip"] else ""
        print(f"  {row['column']:<20}{row['selected_share']:>10.1%}"
              f"{row['sign_consistency']:>14.1%}{flag}")

    coin_flips = stability[stability["coin_flip"]]
    unstable_sign = stability[
        (stability["sign_consistency"] < 0.95) & (stability["selected_share"] > 0.2)
    ]
    print(f"\n    {len(coin_flips)} of {len(columns)} features are selected between 25% and 75% "
          f"of the time.")
    if len(unstable_sign):
        print(f"    {len(unstable_sign)} flip sign across resamples: "
              f"{', '.join(unstable_sign['column'])}.")
        print("    A coefficient whose sign is not stable is not evidence about direction.")

    overlaps = pairwise_stability(X_train, y_train, C=C_CHOSEN, replicates=40, seed=11)
    print(f"\n    Jaccard overlap between feature sets chosen on different resamples: "
          f"{overlaps.mean():.2f}")
    print(f"    (min {overlaps.min():.2f}, max {overlaps.max():.2f}). Two analysts with two")
    print(f"    resamples of the same data agree on {overlaps.mean():.0%} of the selected set.\n")

    # --- scaling ------------------------------------------------------------
    print("  === the same lasso on unscaled features ===")
    raw = features.to_numpy(dtype=float)
    unscaled = fit_penalised(raw[:cut], y_train, raw[cut:], y_test, columns,
                             name="Lasso, unscaled", penalty="l1", C=C_CHOSEN)
    kept_scaled = {c for c, k in zip(columns, lasso.selected, strict=True) if k}
    kept_raw = {c for c, k in zip(columns, unscaled.selected, strict=True) if k}
    print(f"    scaled:    {lasso.n_selected} features, test AUC {lasso.test_auc:.4f}")
    print(f"    unscaled:  {unscaled.n_selected} features, test AUC {unscaled.test_auc:.4f}")
    print(f"    in common: {len(kept_scaled & kept_raw)}")
    print(f"    only when scaled:   {', '.join(sorted(kept_scaled - kept_raw)) or 'none'}")
    print(f"    only when unscaled: {', '.join(sorted(kept_raw - kept_scaled)) or 'none'}")
    print("\n    The penalty acts on coefficients, and coefficients carry the units of their")
    print("    features. Tool wear in minutes and temperature in kelvin are not comparable,")
    print("    so an unscaled penalty selects on measurement units.")

    # --- figure 1: correlation ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(features.corr().abs(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(columns)), columns, fontsize=7)
    ax.grid(visible=False)
    fig.colorbar(image, ax=ax, shrink=0.7, label="|correlation|")
    ax.set_title("The penalty has to choose between these")
    caption(fig, "Fourteen features engineered from five sensors. Every dark cell is a pair "
                 "the lasso is close to indifferent between.")
    print("\n  " + str(save(fig, FIGURES / "01-correlation.png").relative_to(HERE.parent.parent)))

    # --- figure 2: the path -------------------------------------------------
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    for column in columns:
        rows = path[path["column"] == column]
        top.plot(rows["C"], rows["coefficient"], lw=1.4, alpha=0.85)
        last = rows.iloc[-1]
        if abs(last["coefficient"]) > 0.15:
            top.text(last["C"] * 1.06, last["coefficient"], column, fontsize=7, va="center")
    top.set_xscale("log"); top.axhline(0, color="#333333", lw=0.8)
    top.set_ylabel("Coefficient")
    top.set_title("Coefficients along the L1 path")
    top.set_xlim(strengths[0], strengths[-1] * 3)

    bottom.plot(per_strength["C"], per_strength["test_auc"], "o-", color=PALETTE["blue"],
                lw=2, ms=3, label="test AUC")
    bottom.plot(per_strength["C"], per_strength["train_auc"], "o--", color=PALETTE["red"],
                lw=1.5, ms=3, label="train AUC")
    bottom.set_xscale("log"); bottom.set_xlabel("C (higher = weaker penalty)")
    bottom.set_ylabel("ROC-AUC"); bottom.legend(fontsize=8)
    bottom.set_title("and what it costs: almost nothing, over three orders of magnitude")
    caption(fig, f"AI4I 2020, 70/30 split. The score is flat from C={strengths[4]:.3f} upward "
                 f"while the selected set changes completely.")
    print("  " + str(save(fig, FIGURES / "02-path.png").relative_to(HERE.parent.parent)))

    # --- figure 3: stability ------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    order = stability.sort_values("selected_share")
    colours = [PALETTE["red"] if c else PALETTE["blue"] for c in order["coin_flip"]]
    bars = ax.barh(order["column"], order["selected_share"], color=colours)
    ax.axvspan(0.25, 0.75, color=PALETTE["red"], alpha=0.08)
    ax.axvline(0.5, color=PALETTE["red"], ls="--", lw=1.5)
    for bar, value in zip(bars, order["selected_share"], strict=True):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.0%}",
                va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel(f"Share of {REPLICATES} bootstrap refits that kept the feature")
    percent_axis(ax, "x")
    ax.set_xlim(0, 1.12)
    ax.set_title("Which features are actually there?")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    caption(fig, "Shaded band: selected between 25% and 75% of the time. A feature in that "
                 "band was chosen by resampling noise, not by the data.")
    print("  " + str(save(fig, FIGURES / "03-stability.png").relative_to(HERE.parent.parent)))

    # --- figure 4: overlap + scaling ----------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    left.hist(overlaps, bins=24, color=PALETTE["orange"])
    left.axvline(overlaps.mean(), color=PALETTE["red"], lw=2)
    left.text(overlaps.mean(), left.get_ylim()[1] * 0.92, f" mean {overlaps.mean():.2f}",
              fontsize=9, fontweight="bold", color=PALETTE["red"])
    left.set_xlabel("Jaccard overlap between two resamples' selected sets")
    left.set_ylabel("Pairs"); left.set_xlim(0, 1)
    left.set_title("Two runs, two answers")

    width = 0.38
    x = np.arange(len(columns))
    right.barh(x - width / 2, lasso.coefficients, width, label="scaled", color=PALETTE["blue"])
    right.barh(x + width / 2, unscaled.coefficients, width, label="unscaled",
               color=PALETTE["red"])
    right.set_yticks(x, columns, fontsize=7)
    right.axvline(0, color="#333333", lw=0.8)
    right.set_xlabel("Coefficient"); right.legend(fontsize=8)
    right.set_title("Same penalty, different units")
    caption(fig, "Left: 780 pairs of bootstrap refits. Right: identical model and penalty "
                 "strength, run before and after standardising.")
    print("  " + str(save(fig, FIGURES / "04-overlap-and-scaling.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "rows": int(len(frame)),
                "failure_rate": round(float(y.mean()), 5),
                "features": columns,
                "penalties": [
                    {"model": f.name, "train_auc": round(f.train_auc, 4),
                     "test_auc": round(f.test_auc, 4), "kept": f.n_selected}
                    for f in fits
                ],
                "test_auc_spread": round(float(spread), 5),
                "stability": [
                    {"column": r["column"], "selected_share": round(float(r["selected_share"]), 3),
                     "sign_consistency": round(float(r["sign_consistency"]), 3)}
                    for _, r in stability.iterrows()
                ],
                "jaccard": {"mean": round(float(overlaps.mean()), 4),
                            "min": round(float(overlaps.min()), 4),
                            "max": round(float(overlaps.max()), 4)},
                "scaling": {
                    "scaled_kept": sorted(kept_scaled),
                    "unscaled_kept": sorted(kept_raw),
                    "scaled_test_auc": round(lasso.test_auc, 4),
                    "unscaled_test_auc": round(unscaled.test_auc, 4),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
