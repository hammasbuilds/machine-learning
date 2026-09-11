"""Does valuation predict returns? 145 years of Shiller data, counted three ways.

uv run python projects/08-overlapping-windows/run.py
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

from overlap import (  # noqa: E402
    Regression,
    disjoint_indices,
    effective_sample_size,
    forward_return,
    newey_west_se,
    ordinary_least_squares,
    r_squared,
)

from shared.data import SHILLER_SP500, fetch_csv  # noqa: E402
from shared.plotting import (  # noqa: E402
    PALETTE,
    annotate,
    caption,
    percent_axis,
    save,
    use_house_style,
)

FIGURES = HERE / "figures"
HORIZON = 120  # months: ten years


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    frame = fetch_csv(SHILLER_SP500, "shiller-sp500-monthly.csv", parse_dates=["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    frame = frame[
        (frame["PE10"] > 0) & frame["RealPrice" if "RealPrice" in frame else "Real Price"].notna()
    ]
    price_column = "Real Price" if "Real Price" in frame.columns else "RealPrice"

    frame["forward"] = forward_return(frame[price_column].to_numpy(), HORIZON)
    usable = frame.dropna(subset=["forward", "PE10"]).reset_index(drop=True)

    x = usable["PE10"].to_numpy()
    y = usable["forward"].to_numpy()
    n = len(usable)
    independent = effective_sample_size(n, HORIZON)

    print(f"\nShiller S&P 500, {usable['Date'].iloc[0].date()} to {usable['Date'].iloc[-1].date()}")
    print(f"  {n:,} monthly rows of (CAPE, next {HORIZON // 12}-year real return)")
    print(
        f"  but only {independent:.0f} non-overlapping {HORIZON // 12}-year windows fit in that span\n"
    )
    print(
        f"  R-squared {r_squared(x, y):.3f}  |  slope {ordinary_least_squares(x, y, method='x').slope:.5f}"
        f" per CAPE point\n"
    )

    # --- the same relationship, three ways --------------------------------
    naive = ordinary_least_squares(x, y, method="Naive OLS (every month)")

    slope_nw, se_nw = newey_west_se(x, y, lags=HORIZON - 1)
    newey = Regression("Newey-West (lags = horizon)", slope_nw, se_nw, n, independent)

    picked = disjoint_indices(n, HORIZON)
    disjoint = ordinary_least_squares(
        x[picked], y[picked], method=f"Disjoint windows only (n={len(picked)})"
    )

    results = [naive, newey, disjoint]
    print(
        f"  {'method':<34}{'slope':>10}{'std err':>10}{'t':>8}{'p':>9}{'indep n':>10}  significant"
    )
    for r in results:
        s = r.summary()
        print(
            f"  {s['method']:<34}{s['slope']:>10.5f}{s['standard_error']:>10.5f}"
            f"{s['t']:>8.2f}{s['p']:>9.4f}{s['independent_observations']:>10.1f}  "
            f"{'yes' if s['significant_at_5pct'] else 'no'}"
        )

    inflation = abs(naive.t_statistic) / abs(disjoint.t_statistic) if disjoint.t_statistic else 0
    print(
        f"\n  The naive t is {inflation:.1f}x the honest one. sqrt(horizon) = "
        f"{np.sqrt(HORIZON):.1f}, which is where that factor comes from."
    )

    # --- 1. the scatter that convinces everyone ---------------------------
    fig, ax = plt.subplots(figsize=(7.8, 5))
    scatter = ax.scatter(
        x, y, c=usable["Date"].dt.year, cmap="viridis", s=7, alpha=0.55, linewidths=0
    )
    line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        line, naive.slope * line + (y.mean() - naive.slope * x.mean()), color=PALETTE["red"], lw=2.4
    )
    ax.axhline(0, color="#bbbbbb", lw=1, ls="--")
    ax.set_xlabel("CAPE (cyclically-adjusted P/E)")
    ax.set_ylabel(f"Annualised real return over the next {HORIZON // 12} years")
    percent_axis(ax)
    ax.set_title(f"R-squared {r_squared(x, y):.2f}, and it is almost entirely real")
    fig.colorbar(scatter, ax=ax, shrink=0.85, label="year")
    caption(
        fig,
        f"Shiller monthly data, {n:,} rows. The relationship is genuine; what is "
        "wrong is the confidence attached to it.",
    )
    print("\n  " + str(save(fig, FIGURES / "01-the-scatter.png").relative_to(HERE.parent.parent)))

    # --- 2. the t-statistics ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    names = [r.method for r in results]
    ts = [abs(r.t_statistic) for r in results]
    colours = [PALETTE["red"], PALETTE["orange"], PALETTE["green"]]
    bars = ax.barh(names[::-1], ts[::-1], color=colours[::-1])
    ax.axvline(1.96, color="#666666", ls="--", lw=1.6)
    ax.text(2.05, -0.42, "1.96", fontsize=9, color="#666666", fontweight="bold")
    for bar, v in zip(bars, ts[::-1], strict=True):
        ax.text(
            v + 0.4, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontweight="bold"
        )
    ax.set_xlabel("|t| on the CAPE coefficient")
    ax.set_title("Same data, same slope, three honest-looking standard errors")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        f"Overlapping windows share {HORIZON - 1} of their {HORIZON} months. "
        f"Treating them as independent shrinks the standard error by about "
        f"sqrt({HORIZON}) = {np.sqrt(HORIZON):.0f}.",
    )
    print(
        "  " + str(save(fig, FIGURES / "02-three-t-statistics.png").relative_to(HERE.parent.parent))
    )

    # --- 3. what overlap looks like ---------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4))
    for i in range(6):
        ax.barh(i, HORIZON, left=i, height=0.6, color=PALETTE["sky"], alpha=0.85)
    for j, start in enumerate(range(0, 6 * HORIZON, HORIZON)):
        ax.barh(7 + j, HORIZON, left=start, height=0.6, color=PALETTE["green"], alpha=0.9)
    ax.set_yticks([2.5, 9.5], ["consecutive\nmonthly rows", "disjoint\nwindows"])
    ax.set_xlabel("Months")
    ax.set_xlim(0, 6 * HORIZON)
    ax.set_title("Six rows of the regression, drawn to scale")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    caption(
        fig,
        "Top: six consecutive rows, each sharing 119 of 120 months with its neighbour. "
        "Bottom: six rows that share nothing.",
    )
    print(
        "  "
        + str(save(fig, FIGURES / "03-what-overlap-looks-like.png").relative_to(HERE.parent.parent))
    )

    # --- 4. does the conclusion survive at every offset? ------------------
    #
    # There are 120 different ways to pick disjoint windows - start in month 0, or 1, or 2.
    # Reporting the one that happens to look best would be exactly the sin this repo is
    # about, so all of them are shown.
    offsets = []
    for offset in range(HORIZON):
        idx = disjoint_indices(n, HORIZON, offset=offset)
        if len(idx) < 8:
            continue
        fit = ordinary_least_squares(x[idx], y[idx], method="d")
        offsets.append(
            {
                "offset": offset,
                "t": abs(fit.t_statistic),
                "slope": fit.slope,
                "significant": fit.significant,
            }
        )

    spread = pd.DataFrame(offsets)
    share = float(spread["significant"].mean())
    print(
        f"\n  Across all {len(spread)} ways of choosing disjoint windows: "
        f"{share:.0%} are significant at 5%,"
    )
    print(
        f"  |t| ranges {spread['t'].min():.2f} to {spread['t'].max():.2f} "
        f"(median {spread['t'].median():.2f})."
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.hist(spread["t"], bins=26, color=PALETTE["purple"], alpha=0.9)
    ax.axvline(1.96, color=PALETTE["red"], lw=2)
    ax.axvline(abs(naive.t_statistic), color="#999999", lw=2, ls="--")
    annotate(ax, 1.96, ax.get_ylim()[1] * 0.85, "  1.96", color=PALETTE["red"])
    ax.set_xlabel("|t| from one disjoint sample")
    ax.set_ylabel("Starting offsets")
    ax.set_title(f"Every way of picking non-overlapping windows — {share:.0%} reach significance")
    caption(
        fig,
        f"All {len(spread)} possible starting months. Reporting only the best one "
        "would be the same error the rest of this repo is about.",
    )
    print("  " + str(save(fig, FIGURES / "04-every-offset.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "from": str(usable["Date"].iloc[0].date()),
                "to": str(usable["Date"].iloc[-1].date()),
                "monthly_rows": int(n),
                "independent_windows": round(independent, 1),
                "r_squared": round(r_squared(x, y), 4),
                "regressions": [r.summary() for r in results],
                "t_inflation_factor": round(float(inflation), 2),
                "sqrt_horizon": round(float(np.sqrt(HORIZON)), 2),
                "disjoint_offsets_significant": round(share, 4),
                "disjoint_t_range": [
                    round(float(spread["t"].min()), 2),
                    round(float(spread["t"].max()), 2),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
