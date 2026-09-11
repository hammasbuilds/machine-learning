"""Five reductions of the same data, and what each one throws away.

uv run python projects/15-dimensionality/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from reduce import (  # noqa: E402
    distance_correlation,
    neighbourhood_overlap,
    nmf,
    pca,
    truncated_svd,
    trustworthiness,
)

from shared.data import online_retail  # noqa: E402
from shared.plotting import PALETTE, caption, percent_axis, save, use_house_style  # noqa: E402

FIGURES = HERE / "figures"
K = 8


def build_matrix(ledger: pd.DataFrame, *, customers: int = 700, products: int = 120):
    """Customer x product purchase counts. Sparse, skewed, non-negative — NMF's home ground."""
    purchases = ledger[(~ledger["is_return"]) & (ledger["Quantity"] > 0)].dropna(
        subset=["CustomerID"]
    )
    top_products = purchases["StockCode"].value_counts().head(products).index
    purchases = purchases[purchases["StockCode"].isin(top_products)]
    top_customers = purchases["CustomerID"].value_counts().head(customers).index
    purchases = purchases[purchases["CustomerID"].isin(top_customers)]

    matrix = purchases.pivot_table(
        index="CustomerID", columns="StockCode", values="Quantity", aggfunc="sum", fill_value=0
    ).astype(float)
    descriptions = (
        ledger.dropna(subset=["Description"]).groupby("StockCode")["Description"].first()
        if "Description" in ledger.columns
        else pd.Series(dtype=str)
    )
    return matrix, descriptions


def main() -> None:
    use_house_style()
    FIGURES.mkdir(exist_ok=True)

    ledger = online_retail()
    matrix, descriptions = build_matrix(ledger)
    X = matrix.to_numpy()
    print(f"\n{X.shape[0]} customers x {X.shape[1]} products")
    print(f"  {np.mean(X > 0):.1%} of cells non-zero, values 0 to {X.max():,.0f}\n")

    # --- the unit problem ---------------------------------------------------
    scaled = pca(X, K, scale=True)
    unscaled = pca(X, K, scale=False)

    def dominant(d) -> tuple[str, float]:
        weights = np.abs(d.components[0])
        top = int(np.argmax(weights))
        return matrix.columns[top], float(weights[top] / weights.sum())

    print("  === PCA on unscaled data measures your units ===")
    for d in (unscaled, scaled):
        code, share = dominant(d)
        label = str(descriptions.get(code, code))[:34]
        print(f"    {d.name:<16} first component is {share:>5.1%} one product: {label}")
    print(
        f"    variance explained by PC1: unscaled {unscaled.explained_variance_ratio[0]:.1%}, "
        f"scaled {scaled.explained_variance_ratio[0]:.1%}"
    )

    # --- linear methods, compared -------------------------------------------
    methods = {
        "PCA": scaled,
        "Truncated SVD": truncated_svd(X, K),
        "NMF": nmf(X, K, iterations=300),
    }
    print(f"\n  === {K} components ===")
    print(f"  {'method':<16}{'variance kept':>15}{'reconstruction MSE':>21}")
    rows = []
    for name, d in methods.items():
        reference = X if name != "PCA" else (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
        error = d.reconstruction_error(reference)
        rows.append({"method": name, "variance": float(d.cumulative_variance[-1]), "mse": error})
        print(f"  {name:<16}{d.cumulative_variance[-1]:>15.1%}{error:>21.4f}")

    # Computed on the full decomposition: asking an 8-component fit how many
    # components it needs can only answer "8" - a bug that returns a plausible number.
    full = pca(X, min(60, X.shape[1]), scale=True)
    print(
        f"\n    PCA needs {full.components_needed(0.90)} components for 90% of the variance, "
        f"{full.components_needed(0.95)} for 95% (of {X.shape[1]} products)."
    )

    # --- neighbourhood methods ----------------------------------------------
    print("\n  === preserving what? ===")
    embeddings = {
        "PCA (2d)": pca(X, 2, scale=True).embedding,
        "t-SNE": TSNE(
            n_components=2, perplexity=30, init="pca", random_state=0, max_iter=750
        ).fit_transform(X),
    }
    reference = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))

    print(
        f"  {'method':<12}{'trustworthiness':>18}{'neighbour overlap':>20}"
        f"{'distance correlation':>23}"
    )
    geometry = []
    for name, embedding in embeddings.items():
        t = trustworthiness(reference, embedding, k=12)
        o = neighbourhood_overlap(reference, embedding, k=12)
        c = distance_correlation(reference, embedding)
        geometry.append({"method": name, "trustworthiness": t, "overlap": o, "distance_corr": c})
        print(f"  {name:<12}{t:>18.4f}{o:>20.4f}{c:>23.4f}")

    gap = geometry[0]["distance_corr"] - geometry[1]["distance_corr"]
    print("\n    t-SNE keeps neighbourhoods and loses global geometry: distance correlation")
    print(
        f"    {geometry[1]['distance_corr']:.3f} against PCA's {geometry[0]['distance_corr']:.3f}. "
        f"Cluster distances on a t-SNE plot mean nothing."
    )

    # --- 1. the unit problem ------------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.3))
    for ax, d, title in ((left, unscaled, "Unscaled"), (right, scaled, "Scaled")):
        weights = np.abs(d.components[0])
        top = np.argsort(-weights)[:12]
        ax.barh(
            range(len(top)),
            weights[top][::-1],
            color=PALETTE["red"] if d is unscaled else PALETTE["green"],
        )
        ax.set_yticks(
            range(len(top)),
            [str(descriptions.get(matrix.columns[i], matrix.columns[i]))[:22] for i in top][::-1],
            fontsize=7,
        )
        ax.set_xlabel("|loading| on PC1")
        ax.set_title(f"{title} — PC1 explains {d.explained_variance_ratio[0]:.0%}")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "The same data. Scaling decides what the first component is about.",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        f"UCI Online Retail II, {X.shape[0]} customers x {X.shape[1]} products. "
        "Unscaled, PC1 is whichever product is sold in the largest quantities.",
    )
    print("\n  " + str(save(fig, FIGURES / "01-scaling.png").relative_to(HERE.parent.parent)))

    # --- 2. the scree plot ---------------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.2))
    left.bar(
        range(1, len(full.explained_variance_ratio) + 1),
        full.explained_variance_ratio,
        color=PALETTE["blue"],
    )
    left.set_xlabel("Component")
    left.set_ylabel("Variance explained")
    percent_axis(left)
    left.set_title("Scree")
    right.plot(
        range(1, len(full.cumulative_variance) + 1),
        full.cumulative_variance,
        "o-",
        lw=2,
        color=PALETTE["blue"],
        ms=4,
    )
    for share, colour in ((0.9, PALETTE["orange"]), (0.95, PALETTE["red"])):
        n = full.components_needed(share)
        right.axhline(share, color=colour, ls="--", lw=1.3)
        right.plot([n], [full.cumulative_variance[n - 1]], "o", color=colour, ms=8)
        right.annotate(
            f"{n} components = {share:.0%}",
            (n, share),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=colour,
        )
    right.set_xlabel("Components")
    right.set_ylabel("Cumulative variance")
    percent_axis(right)
    right.set_title("How many do you actually need?")
    caption(
        fig,
        "Explained variance is not information: a component can carry 40% of the "
        "variance and nothing you care about.",
    )
    print("  " + str(save(fig, FIGURES / "02-scree.png").relative_to(HERE.parent.parent)))

    # --- 3. PCA vs t-SNE, drawn ---------------------------------------------
    spend = np.log1p(matrix.sum(axis=1).to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (name, embedding) in zip(axes, embeddings.items(), strict=True):
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=spend,
            cmap="viridis",
            s=14,
            alpha=0.75,
            linewidths=0,
        )
        g = next(x for x in geometry if x["method"] == name)
        ax.set_title(
            f"{name} — trust {g['trustworthiness']:.2f}, distance corr {g['distance_corr']:.2f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
    fig.colorbar(scatter, ax=axes, shrink=0.85, label="log total units bought")
    fig.suptitle(
        "Both preserve neighbourhoods. Only one preserves distance.",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        "On the right, the gap between two clumps carries no information. It is read "
        "as meaning anyway, which is the most common misuse in the field.",
    )
    print("  " + str(save(fig, FIGURES / "03-pca-vs-tsne.png").relative_to(HERE.parent.parent)))

    # --- 4. NMF components are readable -------------------------------------
    nmf_model = methods["NMF"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for j, ax in enumerate(axes.ravel()):
        weights = nmf_model.components[j]
        top = np.argsort(-weights)[:8]
        ax.barh(range(len(top)), weights[top][::-1], color=PALETTE["purple"])
        ax.set_yticks(
            range(len(top)),
            [str(descriptions.get(matrix.columns[i], matrix.columns[i]))[:26] for i in top][::-1],
            fontsize=7,
        )
        ax.set_title(
            f"Component {j + 1} ({nmf_model.explained_variance_ratio[j]:.0%})", fontsize=10
        )
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "NMF parts add up and never subtract, so each one is a theme",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    caption(
        fig,
        'A PCA component saying "more of A, less of B" has no meaning for a basket '
        "of goods. An NMF component is a set of things bought together.",
    )
    print("  " + str(save(fig, FIGURES / "04-nmf-parts.png").relative_to(HERE.parent.parent)))

    (HERE / "result.json").write_text(
        json.dumps(
            {
                "shape": list(X.shape),
                "sparsity": round(float(np.mean(X > 0)), 4),
                "pc1_variance": {
                    "unscaled": round(float(unscaled.explained_variance_ratio[0]), 4),
                    "scaled": round(float(scaled.explained_variance_ratio[0]), 4),
                },
                "components_for_90pct": full.components_needed(0.90),
                "components_for_95pct": full.components_needed(0.95),
                "total_features": int(X.shape[1]),
                "linear_methods": [
                    {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows
                ],
                "geometry": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()}
                    for g in geometry
                ],
                "distance_correlation_gap": round(float(gap), 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
