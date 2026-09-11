"""When does a customer come back? 5,876 real customers, half of them still out there.

uv run python projects/13-survival/run.py
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

from survival import (  # noqa: E402
    Cohort,
    concordance,
    fit_cox,
    kaplan_meier,
    log_rank,
    median_survival,
    survival_at,
)

from shared.data import online_retail  # noqa: E402
from shared.plotting import (  # noqa: E402
    PALETTE,
    annotate,
    caption,
    percent_axis,
    save,
    use_house_style,
)

FIGURES = HERE / "figures"


def build_cohort(ledger: pd.DataFrame) -> pd.DataFrame:
    """Time from a customer's first purchase to their next one — or to the end of the data.

    This is a genuine survival problem, not one dressed up as one. The "event" is a repeat
    purchase. A customer who has not repeated by 2011-12-09 has not been observed to fail;
    she has been observed **for as long as we looked**, and no longer.
    """
    attributed = ledger.dropna(subset=["CustomerID"]).copy()
    end_of_data = attributed["InvoiceDate"].max()

    orders = (
        attributed.groupby(["CustomerID", "Invoice"])
        .agg(date=("InvoiceDate", "min"), value=("revenue", "sum"))
        .reset_index()
        .sort_values(["CustomerID", "date"])
    )

    rows = []
    for customer, group in orders.groupby("CustomerID"):
        dates = group["date"].to_numpy()
        first = dates[0]
        first_value = float(group["value"].iloc[0])

        if len(dates) > 1:
            duration = (dates[1] - first) / np.timedelta64(1, "D")
            observed = True
        else:
            duration = (end_of_data - first) / np.timedelta64(1, "D")
            observed = False

        rows.append(
            {
                "CustomerID": customer,
                "duration": max(float(duration), 0.5),
                "observed": observed,
                "first_value": first_value,
                "first_items": int((attributed["Invoice"] == group["Invoice"].iloc[0]).sum()),
                "country": attributed.loc[attributed["CustomerID"] == customer, "Country"].iloc[0],
                "started": pd.Timestamp(first),
            }
        )

    frame = pd.DataFrame(rows)
    # Customers who arrive near the end of the window are censored almost immediately and
    # carry almost no information. They are kept - dropping them would bias the sample
    # toward early joiners - but the follow-up available to each is recorded.
    frame["follow_up_days"] = (end_of_data - frame["started"]).dt.days
    return frame


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    ledger = online_retail()
    frame = build_cohort(ledger)
    cohort = Cohort(frame["duration"].to_numpy(), frame["observed"].to_numpy())

    print(f"\n{len(frame):,} customers, first purchase to repeat purchase")
    print(f"  {cohort.observed.sum():,} came back (event observed)")
    print(
        f"  {(~cohort.observed).sum():,} had not come back when the data ends "
        f"({cohort.censoring_rate:.1%} censored)\n"
    )

    # --- what a classifier would have said --------------------------------
    curve = kaplan_meier(cohort)
    naive = cohort.naive_event_rate()
    print("  === the error a classifier makes here ===")
    print(f"    counting censored customers as 'did not return': {naive:.1%} return")
    print(
        f"    Kaplan-Meier at 90 days                        : {1 - survival_at(curve, 90):.1%} returned"
    )
    print(
        f"    Kaplan-Meier at 180 days                       : {1 - survival_at(curve, 180):.1%} returned"
    )
    print(
        f"    Kaplan-Meier at 365 days                       : {1 - survival_at(curve, 365):.1%} returned"
    )
    median = median_survival(curve)
    print(
        f"    median time to repeat purchase                 : {median:.0f} days"
        if np.isfinite(median)
        else "    median: never reached"
    )

    # --- does first-order size change the timing? -------------------------
    threshold = float(frame["first_value"].median())
    big = frame["first_value"] > threshold
    cohort_big = Cohort(
        frame.loc[big, "duration"].to_numpy(), frame.loc[big, "observed"].to_numpy()
    )
    cohort_small = Cohort(
        frame.loc[~big, "duration"].to_numpy(), frame.loc[~big, "observed"].to_numpy()
    )
    curve_big, curve_small = kaplan_meier(cohort_big), kaplan_meier(cohort_small)
    test = log_rank(cohort_big, cohort_small)

    print(f"\n  === split by first-order value (median GBP {threshold:,.0f}) ===")
    for label, c in (("large first order", curve_big), ("small first order", curve_small)):
        m = median_survival(c)
        print(
            f"    {label:<20} median {m:>6.0f} days"
            if np.isfinite(m)
            else f"    {label:<20} median never reached"
        )
    print(f"    log-rank chi2 {test['chi2']:.2f}, p = {test['p_value']:.2e}")

    # --- Cox ---------------------------------------------------------------
    honest_columns = {
        "log_first_value": np.log1p(frame["first_value"].clip(lower=0)),
        "log_first_items": np.log1p(frame["first_items"]),
        "uk": (frame["country"] == "United Kingdom").astype(float),
    }

    # `follow_up_days` looks like an ordinary covariate and is a leak. For every *censored*
    # customer it equals the duration exactly - it IS the outcome, renamed. It is fitted
    # here deliberately, alongside the honest model, because the size of the difference is
    # the argument: survival data has its own leakage mode and it does not look like one.
    leaky = pd.DataFrame({**honest_columns, "log_follow_up": np.log1p(frame["follow_up_days"])})
    honest = pd.DataFrame(honest_columns)

    identical = float(
        (
            frame.loc[~frame["observed"], "follow_up_days"]
            == frame.loc[~frame["observed"], "duration"].round()
        ).mean()
    )
    print("\n  === a leak that looks like a covariate ===")
    print(f"    for censored customers, follow_up_days == duration in {identical:.1%} of rows")

    def standardise(d: pd.DataFrame) -> np.ndarray:
        return ((d - d.mean()) / d.std().replace(0, 1)).to_numpy()

    fits = {}
    for label, d in (("with follow-up (leaky)", leaky), ("without it (honest)", honest)):
        fitted = fit_cox(standardise(d), cohort, list(d.columns))
        c = concordance(fitted.risk_score(standardise(d)), cohort)
        fits[label] = (fitted, c)
        print(f"    {label:<24} C-index {c:.4f}")

    model, c_index = fits["without it (honest)"]
    leaky_model, leaky_c = fits["with follow-up (leaky)"]

    print(
        f"\n  === Cox proportional hazards ({model.n_events:,} events, "
        f"{model.n_observations:,} customers, honest covariates only) ==="
    )
    ratios = model.hazard_ratios()
    print(f"    {'covariate':<18}{'hazard ratio':>14}{'95% CI':>22}{'p':>12}")
    for _, r in ratios.iterrows():
        ci = f"[{r['lower']:.3f}, {r['upper']:.3f}]"
        print(f"    {r['covariate']:<18}{r['hazard_ratio']:>14.4f}{ci:>22}{r['p_value']:>12.2e}")
    print(f"\n    concordance (C-index): {c_index:.4f}   (0.5 = chance)")

    standardised = pd.DataFrame(honest_columns)
    standardised = (standardised - standardised.mean()) / standardised.std().replace(0, 1)

    # --- 1. the censoring, and what ignoring it costs ----------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    left.fill_between(
        curve["time"], curve["lower"], curve["upper"], color=PALETTE["blue"], alpha=0.18
    )
    left.plot(
        curve["time"],
        curve["survival"],
        lw=2.2,
        color=PALETTE["blue"],
        label="Kaplan-Meier (censoring handled)",
    )
    left.axhline(
        1 - naive,
        color=PALETTE["red"],
        ls="--",
        lw=2,
        label=f"classifier's answer: {naive:.0%} return, full stop",
    )
    if np.isfinite(median):
        left.axvline(median, color=PALETTE["green"], lw=1.6)
        annotate(left, median, 0.52, f"  median {median:.0f}d", color=PALETTE["green"])
    left.set_xlabel("Days since first purchase")
    left.set_ylabel("Still not returned")
    percent_axis(left)
    left.legend(fontsize=8, loc="upper right")
    left.set_title("A survival curve answers 'when'. A classifier answers one point badly.")

    at_risk = curve.set_index("time")["at_risk"]
    right.plot(at_risk.index, at_risk.to_numpy(), lw=2, color=PALETTE["purple"])
    right.set_xlabel("Days since first purchase")
    right.set_ylabel("Customers still at risk")
    right.set_title("The denominator shrinks as customers leave the window")
    caption(
        fig,
        f"UCI Online Retail II, {len(frame):,} customers, "
        f"{cohort.censoring_rate:.0%} censored at 2011-12-09. Shaded band is Greenwood 95%.",
    )
    print("\n  " + str(save(fig, FIGURES / "01-censoring.png").relative_to(HERE.parent.parent)))

    # --- 2. two cohorts ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for c, colour, label in (
        (curve_big, PALETTE["green"], f"first order > GBP {threshold:,.0f}"),
        (curve_small, PALETTE["orange"], f"first order <= GBP {threshold:,.0f}"),
    ):
        ax.fill_between(c["time"], c["lower"], c["upper"], color=colour, alpha=0.15)
        ax.plot(c["time"], c["survival"], lw=2.2, color=colour, label=label)
    ax.set_xlabel("Days since first purchase")
    ax.set_ylabel("Still not returned")
    percent_axis(ax)
    ax.legend()
    ax.set_title(f"Log-rank chi2 {test['chi2']:.1f}, p = {test['p_value']:.1e}")
    caption(
        fig,
        "The log-rank test uses every event time, not just the medians - two curves "
        "can share a median and be completely different.",
    )
    print("  " + str(save(fig, FIGURES / "02-two-cohorts.png").relative_to(HERE.parent.parent)))

    # --- 3. hazard ratios --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = ratios.iloc[::-1]
    colours = [PALETTE["green"] if v > 1 else PALETTE["red"] for v in order["hazard_ratio"]]
    ax.errorbar(
        order["hazard_ratio"],
        range(len(order)),
        xerr=[order["hazard_ratio"] - order["lower"], order["upper"] - order["hazard_ratio"]],
        fmt="o",
        ms=8,
        capsize=4,
        color="#444444",
        ecolor="#999999",
        zorder=3,
    )
    for i, (_, r) in enumerate(order.iterrows()):
        ax.plot(r["hazard_ratio"], i, "o", ms=9, color=colours[i], zorder=4)
        ax.text(
            r["upper"] + 0.01,
            i,
            f"{r['hazard_ratio']:.2f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.axvline(1.0, color="#444444", lw=2)
    ax.set_yticks(range(len(order)), order["covariate"])
    ax.set_xlabel("Hazard ratio (per standard deviation)")
    ax.set_title("Above 1: comes back sooner. Below 1: takes longer.")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        "Cox proportional hazards, Breslow ties, Newton-Raphson. Bars are 95% CIs. "
        "The baseline hazard is never estimated and never needs to be.",
    )
    print("  " + str(save(fig, FIGURES / "03-hazard-ratios.png").relative_to(HERE.parent.parent)))

    # --- 4. risk tiers, which is what it is for ---------------------------
    risk = model.risk_score(standardised.to_numpy())
    tiers = pd.qcut(risk, 4, labels=["slowest", "slow", "fast", "fastest"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for tier, colour in zip(
        ["fastest", "fast", "slow", "slowest"],
        (PALETTE["green"], PALETTE["sky"], PALETTE["orange"], PALETTE["red"]),
        strict=True,
    ):
        mask = np.asarray(tiers == tier)
        c = kaplan_meier(Cohort(cohort.duration[mask], cohort.observed[mask]))
        ax.plot(
            c["time"],
            c["survival"],
            lw=2,
            color=colour,
            label=f"{tier} quartile (n={mask.sum():,})",
        )
    ax.set_xlabel("Days since first purchase")
    ax.set_ylabel("Still not returned")
    percent_axis(ax)
    ax.legend(fontsize=8.5)
    ax.set_title(f"The model sorts customers by when they will return — C-index {c_index:.3f}")
    caption(
        fig,
        "Four quartiles of Cox risk score. Separation between the curves is the "
        "model doing its job; the C-index is that separation as one number.",
    )
    print("  " + str(save(fig, FIGURES / "04-risk-tiers.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "customers": int(len(frame)),
                "events": int(cohort.observed.sum()),
                "censored": int((~cohort.observed).sum()),
                "censoring_rate": round(cohort.censoring_rate, 4),
                "naive_classifier_return_rate": round(naive, 4),
                "kaplan_meier_returned_by": {
                    "90d": round(1 - survival_at(curve, 90), 4),
                    "180d": round(1 - survival_at(curve, 180), 4),
                    "365d": round(1 - survival_at(curve, 365), 4),
                },
                "median_days_to_repeat": None if not np.isfinite(median) else round(median, 1),
                "log_rank_first_order_value": test,
                "leak_check": {
                    "follow_up_equals_duration_when_censored": round(identical, 4),
                    "c_index_with_leak": round(leaky_c, 4),
                    "c_index_honest": round(c_index, 4),
                },
                "cox": {
                    "c_index": round(c_index, 4),
                    "log_likelihood": round(model.log_likelihood, 2),
                    "hazard_ratios": [
                        {
                            "covariate": r["covariate"],
                            "hazard_ratio": round(float(r["hazard_ratio"]), 4),
                            "ci": [round(float(r["lower"]), 4), round(float(r["upper"]), 4)],
                            "p": float(f"{r['p_value']:.3e}"),
                        }
                        for _, r in ratios.iterrows()
                    ],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
