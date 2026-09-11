"""Three detectors, no labels — then the labels, to see what they actually found.

    uv run python projects/16-anomaly/run.py
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

from anomaly import (  # noqa: E402
    LOF,
    Isolation,
    Mahalanobis,
    agreement_matrix,
    lift_at_k,
    precision_at_k,
    recall_at_k,
)

from shared.data import AI4I_MAINTENANCE, fetch  # noqa: E402
from shared.plotting import PALETTE, caption, percent_axis, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
BUDGET = 200  # machines an operations team can inspect
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    with zipfile.ZipFile(fetch(AI4I_MAINTENANCE, "ai4i-2020-maintenance.zip")) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, encoding="utf-8-sig")
    frame.columns = [c.strip() for c in frame.columns]

    sensors = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]
    X = StandardScaler().fit_transform(frame[sensors].to_numpy(dtype=float))
    labels = frame["Machine failure"].astype(bool).to_numpy()

    print(f"\n{len(frame):,} machine cycles, {labels.sum()} failures ({labels.mean():.2%})")
    print(f"  detectors see {len(sensors)} sensor readings and no labels at all\n")

    detectors = [Isolation(contamination=0.05), LOF(neighbours=35), Mahalanobis()]
    scores = {d.name: d.fit_score(X) for d in detectors}

    # --- before any label is touched ---------------------------------------
    print(f"  === do the detectors even agree? (top {BUDGET}) ===")
    agreements = agreement_matrix(scores, k=BUDGET)
    for a in agreements:
        print(f"    {a.a:<22} vs {a.b:<22} overlap {a.overlap:>6.1%}   "
              f"rank corr {a.rank_correlation:>6.3f}")
    print("    Three definitions of 'unusual'. Choosing one is a decision about which")
    print("    definition matters, not a hyperparameter.\n")

    # --- now the labels ----------------------------------------------------
    print(f"  === with {BUDGET} inspections, what did each one find? ===")
    print(f"  {'detector':<24}{'precision':>11}{'recall':>9}{'lift':>8}{'found':>8}")
    rows = []
    for name, s in scores.items():
        p = precision_at_k(s, labels, k=BUDGET)
        r = recall_at_k(s, labels, k=BUDGET)
        rows.append({"detector": name, "precision": p, "recall": r,
                     "lift": lift_at_k(s, labels, k=BUDGET), "found": int(p * BUDGET)})
        print(f"  {name:<24}{p:>11.2%}{r:>9.1%}{rows[-1]['lift']:>8.2f}{rows[-1]['found']:>8}")

    random_precision = labels.mean()
    print(f"  {'random inspection':<24}{random_precision:>11.2%}{BUDGET / len(frame):>9.1%}"
          f"{1.0:>8.2f}{int(random_precision * BUDGET):>8}")

    best = max(rows, key=lambda r: r["precision"])
    print(f"\n    best: {best['detector']} at {best['lift']:.1f}x random — "
          f"{best['found']} real failures in {BUDGET} inspections instead of "
          f"{int(random_precision * BUDGET)}.")

    # --- which failure modes does it find? ---------------------------------
    top = np.argsort(-scores[best["detector"]])[:BUDGET]
    print("\n  === 'anomalous' is not the same set as 'the failure I care about' ===")
    print(f"    {'mode':<8}{'in data':>10}{'in top ' + str(BUDGET):>14}{'caught':>9}")
    mode_rows = []
    for mode in MODES:
        present = int(frame[mode].sum())
        caught = int(frame.iloc[top][mode].sum())
        mode_rows.append({"mode": mode, "total": present, "caught": caught,
                          "share": caught / max(present, 1)})
        print(f"    {mode:<8}{present:>10}{caught:>14}{caught / max(present, 1):>9.1%}")

    missed = [m for m in mode_rows if m["share"] < 0.1 and m["total"] > 10]
    if missed:
        print(f"\n    {', '.join(m['mode'] for m in missed)} barely register: those failures "
              f"do not\n    look unusual in the sensors, so no unsupervised method can find them.")

    # --- what it costs to find more ----------------------------------------
    budgets = [50, 100, 200, 400, 800, 1600]
    curve_rows = [
        {"budget": b, "detector": name,
         "precision": precision_at_k(s, labels, k=b), "recall": recall_at_k(s, labels, k=b)}
        for b in budgets for name, s in scores.items()
    ]
    curve = pd.DataFrame(curve_rows)

    # --- 1. do they agree? --------------------------------------------------
    names = list(scores)
    matrix = np.eye(len(names))
    for a in agreements:
        i, j = names.index(a.a), names.index(a.b)
        matrix[i, j] = matrix[j, i] = a.overlap

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    image = left.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    left.set_xticks(range(len(names)), [n.split()[0] for n in names], fontsize=8)
    left.set_yticks(range(len(names)), [n.split()[0] for n in names], fontsize=8)
    left.grid(visible=False)
    for i in range(len(names)):
        for j in range(len(names)):
            left.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                      fontsize=10, fontweight="bold",
                      color="white" if matrix[i, j] > 0.5 else "#333333")
    left.set_title(f"Overlap in the top {BUDGET}")
    fig.colorbar(image, ax=left, shrink=0.8)

    for (name, s), colour in zip(scores.items(),
                                 (PALETTE["blue"], PALETTE["orange"], PALETTE["green"]),
                                 strict=True):
        right.hist(np.log1p(s - s.min()), bins=60, alpha=0.55, label=name, color=colour)
    right.set_xlabel("log anomaly score"); right.set_ylabel("Cycles")
    right.legend(fontsize=8); right.set_title("Three score distributions, three shapes")
    fig.suptitle("Before any label: do these methods even mean the same thing?",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    caption(fig, f"UCI AI4I 2020, {len(frame):,} cycles, 5 sensors, no labels used.")
    print("\n  " + str(save(fig, FIGURES / "01-do-they-agree.png").relative_to(HERE.parent.parent)))

    # --- 2. what the inspections find ---------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    order = sorted(rows, key=lambda r: r["precision"])
    bars = ax.barh([r["detector"] for r in order], [r["precision"] for r in order],
                   color=PALETTE["blue"])
    ax.axvline(random_precision, color=PALETTE["red"], lw=2, ls="--")
    ax.text(random_precision * 1.15, -0.42, "random inspection", color=PALETTE["red"],
            fontsize=9, fontweight="bold")
    for bar, r in zip(bars, order, strict=True):
        ax.text(r["precision"] + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{r['precision']:.1%}  ({r['lift']:.1f}x)", va="center",
                fontweight="bold", fontsize=9)
    ax.set_xlabel(f"Share of {BUDGET} inspections that found a real failure")
    percent_axis(ax, "x")
    ax.set_title("The only evaluation an operations team recognises")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    caption(fig, f"Labels withheld from every detector and used only here. Base rate "
                 f"{random_precision:.2%}.")
    print("  " + str(save(fig, FIGURES / "02-what-it-finds.png").relative_to(HERE.parent.parent)))

    # --- 3. which modes ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.3))
    colours = [PALETTE["green"] if m["share"] > 0.2 else PALETTE["red"] for m in mode_rows]
    bars = ax.bar([m["mode"] for m in mode_rows], [m["share"] for m in mode_rows], color=colours,
                  width=0.55)
    for bar, m in zip(bars, mode_rows, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, m["share"],
                f"  {m['caught']}/{m['total']}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel(f"Share caught in the top {BUDGET}")
    percent_axis(ax)
    ax.set_title("Some failures are unusual. Others just happen.")
    caption(fig, "TWF = tool wear, HDF = heat dissipation, PWF = power, OSF = overstrain, "
                 "RNF = random. An unsupervised detector cannot find a failure that leaves "
                 "no trace in the sensors.")
    print("  " + str(save(fig, FIGURES / "03-which-modes.png").relative_to(HERE.parent.parent)))

    # --- 4. the budget curve -------------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3))
    for (name, _), colour in zip(scores.items(),
                                 (PALETTE["blue"], PALETTE["orange"], PALETTE["green"]),
                                 strict=True):
        rows_for = curve[curve["detector"] == name]
        left.plot(rows_for["budget"], rows_for["precision"], "o-", lw=2, color=colour, label=name)
        right.plot(rows_for["budget"], rows_for["recall"], "o-", lw=2, color=colour, label=name)
    left.axhline(random_precision, color=PALETTE["red"], ls="--", lw=1.5)
    left.set_xscale("log"); left.set_xlabel("Inspections"); left.set_ylabel("Precision")
    percent_axis(left); left.set_title("Every extra inspection is worth less")
    right.set_xscale("log"); right.set_xlabel("Inspections"); right.set_ylabel("Recall")
    percent_axis(right); right.set_title("and catches proportionally more")
    right.legend(fontsize=8)
    caption(fig, "The operations question is not 'which model is best' but 'how many "
                 "inspections can we afford', and the two curves answer it together.")
    print("  " + str(save(fig, FIGURES / "04-budget.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "cycles": int(len(frame)), "failures": int(labels.sum()),
                "base_rate": round(float(labels.mean()), 5),
                "budget": BUDGET,
                "agreement": [{"a": a.a, "b": a.b, "overlap": round(a.overlap, 4),
                               "rank_correlation": round(a.rank_correlation, 4)}
                              for a in agreements],
                "detectors": [{k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in r.items()} for r in rows],
                "by_failure_mode": [{k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in m.items()} for m in mode_rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
