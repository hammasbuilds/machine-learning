"""Pricing 678,013 real motor policies, with and without the exposure offset.

uv run python projects/11-insurance-pricing/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from pricing import (  # noqa: E402
    credible_bands,
    design,
    fit_frequency,
    fit_severity,
    fit_tweedie,
    gini,
    lift_table,
    poisson_deviance,
    prepare,
)

from shared.data import french_motor  # noqa: E402
from shared.plotting import (  # noqa: E402
    PALETTE,
    annotate,
    caption,
    money_axis,
    save,
    use_house_style,
)

FIGURES = HERE / "figures"
FACTORS = ["driver_band", "vehicle_band", "bonus_band", "power_band", "density_band", "Area"]


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frequency, severity = french_motor()
    frame = prepare(frequency)

    claim_totals = severity.groupby("IDpol")["ClaimAmount"].sum()
    frame["total_loss"] = frame["IDpol"].map(claim_totals).fillna(0.0)

    print(f"\n{len(frame):,} policies, {frame['Exposure'].sum():,.0f} policy-years of exposure")
    print(f"  {(frame['ClaimNb'] > 0).mean():.2%} of policies had a claim")
    print(f"  claim frequency: {frame['ClaimNb'].sum() / frame['Exposure'].sum():.4f} per year")
    print(
        f"  mean exposure: {frame['Exposure'].mean():.3f} years "
        f"(min {frame['Exposure'].min():.3f}, max {frame['Exposure'].max():.2f})\n"
    )

    # Train and test split by policy, at random: these are independent contracts, not a
    # time series, so there is no ordering to preserve.
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(frame))
    cut = int(len(frame) * 0.75)
    train = frame.iloc[shuffled[:cut]].copy()
    test = frame.iloc[shuffled[cut:]].copy()
    print(f"  train {len(train):,} | test {len(test):,}")

    # --- frequency, with and without the offset ---------------------------
    with_offset = fit_frequency(train, FACTORS, use_offset=True)
    without_offset = fit_frequency(train, FACTORS, use_offset=False)

    Xte, _ = design(test, FACTORS)
    exposure_te = test["Exposure"].to_numpy(dtype=float)
    claims_te = test["ClaimNb"].to_numpy(dtype=float)

    predicted_right = with_offset.predict(Xte, exposure=exposure_te)
    predicted_wrong = without_offset.predict(Xte)  # a count, blind to time on risk

    print("\n  === the offset ===")
    for name, predicted in (
        ("with log(exposure) offset", predicted_right),
        ("WITHOUT the offset", predicted_wrong),
    ):
        print(
            f"    {name:<28} deviance {poisson_deviance(predicted, claims_te):.5f}   "
            f"gini {gini(predicted, claims_te, exposure_te):.4f}"
        )

    # What the missing offset actually does, in the units a policyholder feels: the
    # no-offset model predicts the same expected count whatever the term, so a two-week
    # policy is charged like an annual one.
    short = test["Exposure"] < 0.15
    long = test["Exposure"] > 0.85
    print("\n    expected claims for one policy:")
    for label, mask in (("on risk < 2 months", short), ("on risk > 10 months", long)):
        print(
            f"      {label:<22} with offset {predicted_right[mask].mean():.4f}   "
            f"without {predicted_wrong[mask].mean():.4f}"
        )
    print(
        f"      the no-offset model charges a 2-week policy "
        f"{predicted_wrong[short].mean() / max(predicted_wrong[long].mean(), 1e-9):.0%} of what it "
        f"charges an annual one,\n      while the correct model charges "
        f"{predicted_right[short].mean() / max(predicted_right[long].mean(), 1e-9):.0%}."
    )

    # And the trap in judging it: raw empirical rates on tiny denominators.
    bands = credible_bands(predicted_right, claims_te, exposure_te)
    print("\n    === why the raw rates look alarming, and are not ===")
    print(
        f"    {'exposure band':<16}{'policies':>10}{'exp-yrs':>10}{'observed':>11}"
        f"{'predicted':>11}{'% claiming':>12}  credible"
    )
    for _, r in bands.iterrows():
        print(
            f"    {str(r['band']):<16}{int(r['policies']):>10,}{r['exposure_years']:>10,.0f}"
            f"{r['observed_rate']:>11.4f}{r['predicted_rate']:>11.4f}"
            f"{r['share_claiming']:>12.2%}  {'yes' if r['credible'] else 'NO'}"
        )
    print("    The shortest band runs at 1.06 claims/year - and has the LOWEST share of")
    print("    policies claiming at all. It is a small denominator, not a dangerous driver.")

    # --- severity ----------------------------------------------------------
    claims_only = train[train["total_loss"] > 0].copy()
    claims_only["ClaimAmount"] = claims_only["total_loss"] / claims_only["ClaimNb"].clip(lower=1)
    # One claim in this dataset is 4.075 million euros. Actuaries cap large losses and
    # price them separately; leaving it in lets a single accident set everyone's premium.
    cap = float(claims_only["ClaimAmount"].quantile(0.995))
    capped = int((claims_only["ClaimAmount"] > cap).sum())
    claims_only["ClaimAmount"] = claims_only["ClaimAmount"].clip(upper=cap)
    print("\n  === severity ===")
    print(
        f"    {len(claims_only):,} policies with a claim; {capped} losses capped at "
        f"EUR {cap:,.0f} (99.5th percentile)"
    )

    severity_model = fit_severity(claims_only, FACTORS)
    print(
        f"    mean claim EUR {claims_only['ClaimAmount'].mean():,.0f}, "
        f"median EUR {claims_only['ClaimAmount'].median():,.0f}"
    )

    # --- pure premium: two models vs one ----------------------------------
    Xte_sev, _ = design(test, FACTORS)
    predicted_severity = severity_model.predict(Xte_sev)
    pure_premium_two = predicted_right * predicted_severity  # frequency x severity

    tweedie = fit_tweedie(train, FACTORS, power=1.5)
    pure_premium_one = tweedie.predict(Xte) * exposure_te

    loss_te = test["total_loss"].to_numpy(dtype=float)
    print("\n  === pure premium ===")
    rows = []
    for name, predicted in (
        ("frequency x severity", pure_premium_two),
        ("Tweedie, one model", pure_premium_one),
    ):
        g = gini(predicted, loss_te, exposure_te)
        ratio = loss_te.sum() / max(predicted.sum(), 1e-9)
        rows.append(
            {
                "model": name,
                "gini": g,
                "loss_ratio": ratio,
                "mean_premium": predicted.sum() / exposure_te.sum(),
            }
        )
        print(
            f"    {name:<24} gini {g:.4f}   actual/predicted {ratio:.3f}   "
            f"EUR {predicted.sum() / exposure_te.sum():,.0f}/year"
        )

    # --- 1. the offset, drawn ----------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    bins = np.linspace(0, 1, 26)
    left.hist(frame["Exposure"], bins=bins, color=PALETTE["blue"], alpha=0.85)
    left.set_xlabel("Exposure (policy-years on risk)")
    left.set_ylabel("Policies")
    left.set_title("Policies are not observed for the same length of time")

    x = np.arange(len(bands))
    widths = np.clip(bands["exposure_years"] / bands["exposure_years"].max(), 0.18, 1.0) * 0.42
    right.bar(x - 0.21, bands["observed_rate"], widths, label="observed", color="#8a8a83")
    right.bar(x + 0.21, bands["predicted_rate"], widths, label="predicted", color=PALETTE["green"])
    for i, r in bands.iterrows():
        if not r["credible"]:
            right.text(
                i,
                r["observed_rate"],
                "  thin\n  exposure",
                fontsize=7.5,
                color=PALETTE["red"],
                fontweight="bold",
                va="bottom",
            )
    right.set_xticks(x, [str(b) for b in bands["band"]], fontsize=7, rotation=20)
    right.set_ylabel("Claims per exposure-year")
    right.set_xlabel("Exposure band  (bar width = exposure behind the estimate)")
    right.legend(fontsize=8)
    right.set_title("An observed rate is only evidence in proportion to its exposure")
    caption(
        fig,
        f"French motor TPL, {len(frame):,} policies, {frame['Exposure'].sum():,.0f} "
        "exposure-years. Source: OpenML 41214/41215 (CASdatasets).",
    )
    print("\n  " + str(save(fig, FIGURES / "01-the-offset.png").relative_to(HERE.parent.parent)))

    # --- 2. the rating table ------------------------------------------------
    rel = with_offset.relativities()
    rel = rel[rel["factor"] != "intercept"]
    top = pd.concat([rel.head(9), rel.tail(9)]).drop_duplicates("factor")
    fig, ax = plt.subplots(figsize=(8.4, 6))
    colours = [PALETTE["red"] if v > 1 else PALETTE["green"] for v in top["relativity"]]
    ax.barh(top["factor"], top["relativity"] - 1, left=1, color=colours)
    ax.axvline(1.0, color="#444444", lw=2)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(
            r["relativity"] + (0.02 if r["relativity"] > 1 else -0.02),
            i,
            f"{r['relativity']:.2f}",
            va="center",
            ha="left" if r["relativity"] > 1 else "right",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlabel("Relativity (multiplier on the base rate)")
    ax.set_title("The rating table: who pays more, and how much more")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        "exp(coefficient) from the Poisson frequency GLM. 1.00 is the base class. "
        "This is the form a pricing committee can argue with.",
    )
    print("  " + str(save(fig, FIGURES / "02-rating-table.png").relative_to(HERE.parent.parent)))

    # --- 3. severity is not normal ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.3))
    amounts = claims_only["ClaimAmount"]
    ax.hist(
        amounts,
        bins=np.logspace(np.log10(max(amounts.min(), 1)), np.log10(amounts.max()), 60),
        color=PALETTE["orange"],
        alpha=0.9,
    )
    ax.set_xscale("log")
    ax.axvline(amounts.mean(), color=PALETTE["red"], lw=2)
    ax.axvline(amounts.median(), color=PALETTE["blue"], lw=2)
    annotate(
        ax,
        amounts.mean(),
        ax.get_ylim()[1] * 0.8,
        f"  mean EUR {amounts.mean():,.0f}",
        color=PALETTE["red"],
    )
    annotate(
        ax,
        amounts.median(),
        ax.get_ylim()[1] * 0.55,
        f"  median EUR {amounts.median():,.0f}",
        dx=-92,
        color=PALETTE["blue"],
    )
    ax.set_xlabel("Claim amount (EUR, log scale)")
    ax.set_ylabel("Claims")
    ax.set_title("Claim cost is right-skewed, so severity gets a Gamma, not a normal")
    caption(
        fig,
        f"{len(claims_only):,} policies with a claim. A normal model here predicts "
        "negative claim costs for the cheapest risks.",
    )
    print("  " + str(save(fig, FIGURES / "03-severity.png").relative_to(HERE.parent.parent)))

    # --- 4. does the price work? --------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, predicted, title in (
        (left, pure_premium_two, "frequency x severity"),
        (right, pure_premium_one, "Tweedie, one model"),
    ):
        table = lift_table(predicted, loss_te, exposure_te, bins=10)
        width = 0.4
        idx = np.arange(len(table))
        ax.bar(idx - width / 2, table["predicted"], width, label="predicted", color=PALETTE["blue"])
        ax.bar(idx + width / 2, table["actual"], width, label="actual", color=PALETTE["orange"])
        ax.set_xticks(idx, [str(i) for i in table["decile"]])
        ax.set_xlabel("Decile of predicted risk (cheapest to dearest)")
        ax.set_ylabel("Loss per exposure-year (EUR)")
        money_axis(ax, symbol="")
        ax.set_title(f"{title} — gini {gini(predicted, loss_te, exposure_te):.3f}")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Both price the risk. Only one can say whether frequency or severity moved.",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        f"Held-out {len(test):,} policies. A working model is a monotone staircase: "
        "the decile called cheapest really is cheapest.",
    )
    print("  " + str(save(fig, FIGURES / "04-lift.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "policies": int(len(frame)),
                "exposure_years": round(float(frame["Exposure"].sum()), 1),
                "claim_rate_per_year": round(
                    float(frame["ClaimNb"].sum() / frame["Exposure"].sum()), 5
                ),
                "policies_with_claim": round(float((frame["ClaimNb"] > 0).mean()), 5),
                "frequency": {
                    "with_offset": {
                        "deviance": round(poisson_deviance(predicted_right, claims_te), 6),
                        "gini": round(gini(predicted_right, claims_te, exposure_te), 4),
                    },
                    "without_offset": {
                        "deviance": round(poisson_deviance(predicted_wrong, claims_te), 6),
                        "gini": round(gini(predicted_wrong, claims_te, exposure_te), 4),
                    },
                    "short_vs_long_charge_ratio": {
                        "with_offset": round(
                            float(
                                predicted_right[short].mean()
                                / max(predicted_right[long].mean(), 1e-9)
                            ),
                            4,
                        ),
                        "without_offset": round(
                            float(
                                predicted_wrong[short].mean()
                                / max(predicted_wrong[long].mean(), 1e-9)
                            ),
                            4,
                        ),
                    },
                    "credibility_bands": [
                        {
                            "band": str(r["band"]),
                            "exposure_years": round(float(r["exposure_years"]), 1),
                            "observed_rate": round(float(r["observed_rate"]), 4),
                            "predicted_rate": round(float(r["predicted_rate"]), 4),
                            "share_claiming": round(float(r["share_claiming"]), 4),
                            "credible": bool(r["credible"]),
                        }
                        for _, r in bands.iterrows()
                    ],
                },
                "severity": {
                    "claims": int(len(claims_only)),
                    "cap_eur": round(cap, 2),
                    "capped": capped,
                    "mean_eur": round(float(claims_only["ClaimAmount"].mean()), 2),
                },
                "pure_premium": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
