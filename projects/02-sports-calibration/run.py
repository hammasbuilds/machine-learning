"""Nine Premier League seasons: Elo, a fitted model, and the closing odds.

uv run python projects/02-sports-calibration/run.py
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

from elo import (  # noqa: E402
    AWAY,
    DRAW,
    HOME,
    Elo,
    accuracy,
    implied_probabilities,
    log_loss,
    multiclass_brier,
    overround,
    reliability,
    three_way_probabilities,
)

from shared.data import Source, fetch_csv  # noqa: E402
from shared.plotting import PALETTE, caption, percent_axis, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]
ODDS = ["PSCH", "PSCD", "PSCA"]  # Pinnacle closing - the sharpest prices published


def season_source(code: str) -> Source:
    return Source(
        name=f"football-data-e0-{code}",
        url=f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv",
        note=f"English Premier League 20{code[:2]}/{code[2:]}: results and closing odds.",
        licence="Free to use with attribution (football-data.co.uk)",
    )


def load() -> pd.DataFrame:
    frames = []
    for code in SEASONS:
        frame = fetch_csv(season_source(code), f"football-data-e0-{code}.csv")
        keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", *ODDS]
        missing = [c for c in keep if c not in frame.columns]
        if missing:
            print(f"  skipping 20{code[:2]}/{code[2:]}: missing {missing}")
            continue
        frame = frame[keep].dropna().copy()
        frame["season"] = f"20{code[:2]}/{code[2:]}"
        frames.append(frame)

    matches = pd.concat(frames, ignore_index=True)
    matches["outcome"] = matches["FTR"].map({"H": HOME, "D": DRAW, "A": AWAY}).astype(int)
    return matches


def walk_forward(matches: pd.DataFrame, *, k: float, draw_rate: float) -> np.ndarray:
    """Predict every match using only what was known before it was played.

    The rating is updated *after* the prediction is recorded, every time. Fitting Elo on a
    whole season and then scoring that season is the same error as shuffling a time series,
    and it produces a model that appears to know results it could not have known.
    """
    model = Elo(k=k, mov_factor=0.35)
    predictions = np.zeros((len(matches), 3))
    season = None

    for i, row in enumerate(matches.itertuples(index=False)):
        if season is not None and row.season != season:
            model.regress_to_mean(0.25)
        season = row.season

        expected = model.expected_home_score(row.HomeTeam, row.AwayTeam)
        predictions[i] = three_way_probabilities(expected, draw_rate=draw_rate)
        model.update(row.HomeTeam, row.AwayTeam, int(row.FTHG), int(row.FTAG))

    return predictions


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    matches = load()
    matches = matches.reset_index(drop=True)
    outcomes = matches["outcome"].to_numpy()
    n = len(matches)
    print(f"\n{n:,} matches across {matches['season'].nunique()} seasons\n")

    base = np.array([(outcomes == o).mean() for o in (HOME, DRAW, AWAY)])
    print(f"  base rates   home {base[0]:.1%}  draw {base[1]:.1%}  away {base[2]:.1%}")

    # --- the three forecasters -------------------------------------------
    always_home = np.tile([1.0, 0.0, 0.0], (n, 1))
    base_rate = np.tile(base, (n, 1))
    elo = walk_forward(matches, k=20.0, draw_rate=base[1])

    raw_odds = matches[ODDS].to_numpy(dtype=float)
    book = implied_probabilities(raw_odds)
    book_raw = 1.0 / raw_odds  # deliberately un-normalised, to show what the trap costs
    margin = overround(raw_odds)
    print(
        f"  bookmaker overround: {margin.mean():.3f} mean, {margin.min():.3f}-{margin.max():.3f}\n"
    )

    forecasters = {
        "Always home": always_home,
        "Base rate": base_rate,
        "Elo (walk-forward)": elo,
        "Closing odds": book,
    }

    rows = []
    for name, p in forecasters.items():
        rows.append(
            {
                "forecaster": name,
                "accuracy": accuracy(p, outcomes),
                "brier": multiclass_brier(p, outcomes),
                "log_loss": log_loss(p, outcomes),
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    trap = multiclass_brier(book_raw, outcomes)
    print(
        f"\n  Same odds without removing the overround: Brier {trap:.4f} "
        f"(vs {table.loc[3, 'brier']:.4f}). Barely moves - Brier is forgiving here."
    )

    # --- where the margin actually bites ----------------------------------
    #
    # The overround costs almost nothing in Brier, which is exactly why it is worth
    # measuring somewhere it cannot hide. Elo beats the base rate on every metric, so it
    # knows something real about football. The question that settles whether "knows
    # something" is worth anything is not a score - it is a bankroll.
    stake = 1.0
    edge = elo - book  # where the model thinks the price is wrong
    bet_on = np.argmax(edge, axis=1)
    has_edge = edge[np.arange(n), bet_on] > 0.0

    won = bet_on == outcomes
    payout = raw_odds[np.arange(n), bet_on]
    pnl = np.where(won, (payout - 1.0) * stake, -stake)
    pnl_edge = np.where(has_edge, pnl, 0.0)

    favourite = np.argmin(raw_odds, axis=1)
    pnl_fav = np.where(
        favourite == outcomes, (raw_odds[np.arange(n), favourite] - 1.0) * stake, -stake
    )

    rng = np.random.default_rng(0)
    random_pick = rng.integers(0, 3, n)
    pnl_random = np.where(
        random_pick == outcomes, (raw_odds[np.arange(n), random_pick] - 1.0) * stake, -stake
    )

    strategies = {
        "Bet where Elo sees value": pnl_edge[has_edge],
        "Always back the favourite": pnl_fav,
        "Pick at random": pnl_random,
    }
    print("\n  A bankroll is the test a score cannot fake:")
    for name, series in strategies.items():
        roi = series.sum() / (len(series) * stake)
        print(f"    {name:<28} {len(series):>5,} bets   ROI {roi:+.2%}")
    print(f"    {'bookmaker margin per bet':<28} {'':>5}        {-(margin.mean() - 1):+.2%}")

    # --- 1. accuracy says one thing, Brier says another -------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.3))
    order = table.sort_values("accuracy")
    left.barh(order["forecaster"], order["accuracy"], color=PALETTE["sky"])
    for y, v in enumerate(order["accuracy"]):
        left.text(v + 0.005, y, f"{v:.1%}", va="center", fontweight="bold", fontsize=9)
    left.set_title("Accuracy")
    left.set_xlim(0, 0.65)
    percent_axis(left, "x")
    left.grid(axis="x")
    left.grid(axis="y", visible=False)

    order = table.sort_values("brier", ascending=False)
    right.barh(order["forecaster"], order["brier"], color=PALETTE["orange"])
    for y, v in enumerate(order["brier"]):
        right.text(v + 0.005, y, f"{v:.3f}", va="center", fontweight="bold", fontsize=9)
    right.set_title("Brier score (lower is better)")
    right.set_xlim(0, 1.05)
    right.grid(axis="x")
    right.grid(axis="y", visible=False)

    fig.suptitle(
        "The two metrics rank the same forecasters differently",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        f"Premier League {matches['season'].iloc[0]}-{matches['season'].iloc[-1]}, "
        f"n={n:,} matches. Source: football-data.co.uk closing odds (Pinnacle).",
    )
    print(
        "\n  "
        + str(save(fig, FIGURES / "01-accuracy-vs-brier.png").relative_to(HERE.parent.parent))
    )

    # --- 2. calibration ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.2, ls="--", label="perfect calibration")
    for name, colour in (("Elo (walk-forward)", "blue"), ("Closing odds", "orange")):
        curve = reliability(forecasters[name], outcomes, bins=10).dropna()
        ax.plot(
            curve["predicted"],
            curve["observed"],
            "o-",
            color=PALETTE[colour],
            lw=1.8,
            ms=5,
            label=name,
        )
    ax.set_xlabel("Probability the forecaster assigned")
    ax.set_ylabel("How often it actually happened")
    ax.set_title("Calibration: does 30% mean 30%?")
    percent_axis(ax)
    percent_axis(ax, "x")
    ax.legend(loc="upper left")
    caption(fig, f"All three outcomes pooled: {n * 3:,} (match, outcome) pairs in 10 bins.")
    print("  " + str(save(fig, FIGURES / "02-calibration.png").relative_to(HERE.parent.parent)))

    # --- 3. what the overround does if you forget to remove it ------------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.hist(margin, bins=40, color=PALETTE["purple"], alpha=0.8)
    ax.axvline(1.0, color=PALETTE["red"], lw=2)
    ax.text(
        1.001,
        ax.get_ylim()[1] * 0.9,
        "  a fair book sums to 1.00",
        color=PALETTE["red"],
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("Sum of implied probabilities (1/odds) across the three outcomes")
    ax.set_ylabel("Matches")
    ax.set_title(f"The bookmaker's margin: {margin.mean():.1%} of every book, on every match")
    caption(
        fig, "Comparing a model against un-normalised odds hands the model this margin for free."
    )
    print("  " + str(save(fig, FIGURES / "03-overround.png").relative_to(HERE.parent.parent)))

    # --- 4. Elo trajectories ----------------------------------------------
    model = Elo(k=20.0, mov_factor=0.35)
    history: dict[str, list[tuple[int, float]]] = {}
    season = None
    for i, row in enumerate(matches.itertuples(index=False)):
        if season is not None and row.season != season:
            model.regress_to_mean(0.25)
        season = row.season
        model.update(row.HomeTeam, row.AwayTeam, int(row.FTHG), int(row.FTAG))
        for team in (row.HomeTeam, row.AwayTeam):
            history.setdefault(team, []).append((i, model.rating(team)))

    top = sorted(model.ratings.items(), key=lambda kv: -kv[1])[:6]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for (team, _), colour in zip(top, PALETTE.values(), strict=False):
        pts = history[team]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.5, label=team, color=colour)
    ax.axhline(1500, color="#cccccc", lw=1, ls=":")
    ax.set_xlabel("Match number across nine seasons")
    ax.set_ylabel("Elo rating")
    ax.set_title("Six strongest clubs, rated match by match")
    ax.legend(ncols=3, loc="lower left", fontsize=8)
    caption(
        fig, "Ratings regress 25% toward 1500 between seasons, because squads change over a summer."
    )
    print(
        "  " + str(save(fig, FIGURES / "04-elo-trajectories.png").relative_to(HERE.parent.parent))
    )

    # --- 5. the bankroll --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for (name, series), colour in zip(
        strategies.items(), (PALETTE["blue"], PALETTE["orange"], PALETTE["green"]), strict=True
    ):
        ax.plot(np.arange(1, len(series) + 1), np.cumsum(series), lw=1.7, label=name, color=colour)
    ax.axhline(0, color="#bbbbbb", lw=1.2, ls="--")
    ax.set_xlabel("Bets placed")
    ax.set_ylabel("Cumulative profit (units of 1)")
    ax.set_title("Elo knows something about football. It is not enough to beat the price.")
    ax.legend(loc="lower left")
    caption(
        fig,
        f"Flat 1-unit stakes at Pinnacle closing odds, n={n:,} matches. "
        f"The house edge is {margin.mean() - 1:.1%} per bet and compounds downward.",
    )
    print("  " + str(save(fig, FIGURES / "05-bankroll.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "matches": int(n),
                "seasons": sorted(matches["season"].unique().tolist()),
                "base_rates": {
                    "home": round(float(base[0]), 4),
                    "draw": round(float(base[1]), 4),
                    "away": round(float(base[2]), 4),
                },
                "mean_overround": round(float(margin.mean()), 4),
                "brier_without_removing_overround": round(trap, 4),
                "betting_roi": {
                    name: round(float(series.sum() / len(series)), 4)
                    for name, series in strategies.items()
                },
                "scores": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
