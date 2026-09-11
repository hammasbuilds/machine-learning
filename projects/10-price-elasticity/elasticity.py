"""Price elasticity, and the confounder that flips its sign.

Demand curves slope down. Raise the price, sell less. It is the most robust finding in
economics and one of the easiest to contradict with real data.

Regress log quantity on log price across a catalogue of products and the coefficient
frequently comes out **positive** — apparently, customers buy more when things cost more.
Nobody believes that, but the regression is arithmetically correct, and understanding why it
lies is more useful than knowing the right answer.

**The confounder is the product.** Expensive products and cheap products are different
goods bought by different people for different reasons. A GBP 40 hamper and a GBP 0.85
greetings card differ in price by a factor of fifty and in units sold by a factor of
hundreds, and a regression across both is measuring *what kind of thing this is*, not *what
happens when its price changes*.

The fix is not a better model. It is a better comparison:

  naive      pool every product. Measures the catalogue, not the demand curve.
  within     compare each product only against itself, at different times. The only
             variation that answers the question.

A third source of confounding survives even the within-product estimate and is stated
rather than solved: prices are not set at random. They fall when demand is weak, rise when
it is strong, and both directions bias the estimate. Doing this properly needs an
instrument or an experiment, neither of which a historical ledger contains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Elasticity:
    """One estimate, with what it was computed from."""

    method: str
    elasticity: float
    standard_error: float
    n_observations: int
    n_products: int

    @property
    def t_statistic(self) -> float:
        return self.elasticity / self.standard_error if self.standard_error else 0.0

    @property
    def sign_is_sane(self) -> bool:
        """Demand curves slope down. A positive elasticity is a red flag, not a discovery."""
        return self.elasticity < 0

    def summary(self) -> dict:
        return {
            "method": self.method,
            "elasticity": round(self.elasticity, 3),
            "standard_error": round(self.standard_error, 3),
            "t": round(self.t_statistic, 2),
            "observations": self.n_observations,
            "products": self.n_products,
            "sign_is_sane": self.sign_is_sane,
        }

    def sentence(self) -> str:
        direction = "fall" if self.sign_is_sane else "RISE"
        return (
            f"{self.method}: a 10% price increase is associated with units that "
            f"{direction} {abs(self.elasticity) * 10:.1f}%."
        )


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and its standard error, with an intercept."""
    design = np.c_[np.ones(len(x)), x]
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    dof = max(len(x) - 2, 1)
    covariance = (residuals @ residuals / dof) * np.linalg.inv(design.T @ design)
    return float(coefficients[1]), float(np.sqrt(max(covariance[1, 1], 0.0)))


def product_month_panel(transactions: pd.DataFrame, *, min_months: int = 6) -> pd.DataFrame:
    """One row per product per month: units sold, and the price they sold at.

    Returns are excluded here. A refund is not a purchase at a negative price, and letting
    negative quantities through makes `log(units)` undefined for exactly the products with
    the most interesting price histories.

    `min_months` keeps only products observed often enough to have a within-product price
    history. A product seen in one month contributes nothing to a within estimate and would
    quietly load the naive one instead.
    """
    frame = transactions[(~transactions["is_return"]) & (transactions["Quantity"] > 0)].copy()
    frame["month"] = frame["InvoiceDate"].dt.to_period("M")

    panel = (
        frame.groupby(["StockCode", "month"])
        .agg(units=("Quantity", "sum"), revenue=("revenue", "sum"))
        .reset_index()
    )
    panel = panel[panel["units"] > 0]
    # Revenue-weighted price, not the mean of the listed prices: a product sold 500 times at
    # 1.00 and once at 9.99 was, for practical purposes, a 1.00 product that month.
    panel["price"] = panel["revenue"] / panel["units"]
    panel = panel[panel["price"] > 0]

    counts = panel.groupby("StockCode")["month"].transform("size")
    panel = panel[counts >= min_months]

    panel["log_units"] = np.log(panel["units"])
    panel["log_price"] = np.log(panel["price"])
    return panel.reset_index(drop=True)


def naive_elasticity(panel: pd.DataFrame) -> Elasticity:
    """Pool everything. Measures which products are expensive, not what price does."""
    slope, se = _ols(panel["log_price"].to_numpy(), panel["log_units"].to_numpy())
    return Elasticity("Pooled across products", slope, se, len(panel), panel["StockCode"].nunique())


def within_product_elasticity(panel: pd.DataFrame) -> Elasticity:
    """Compare each product only with itself.

    Implemented by demeaning within product — subtract each product's own average log price
    and log units, then regress the deviations. Algebraically identical to a fixed-effects
    regression with one dummy per product, and it does not build a 3,000-column matrix.
    """
    frame = panel.copy()
    for column in ("log_price", "log_units"):
        frame[f"d_{column}"] = frame[column] - frame.groupby("StockCode")[column].transform("mean")

    slope, se = _ols(frame["d_log_price"].to_numpy(), frame["d_log_units"].to_numpy())
    return Elasticity(
        "Within product (fixed effects)", slope, se, len(frame), frame["StockCode"].nunique()
    )


def within_product_and_month(panel: pd.DataFrame) -> Elasticity:
    """Within product, and within month.

    Removes anything that moved the whole catalogue at once — Christmas, a recession, a
    postage change. What is left is: when *this* product was dearer than its own average in
    a month when *everything* was not, did it sell less?
    """
    frame = panel.copy()
    for column in ("log_price", "log_units"):
        by_product = frame.groupby("StockCode")[column].transform("mean")
        by_month = frame.groupby("month")[column].transform("mean")
        frame[f"d_{column}"] = frame[column] - by_product - by_month + frame[column].mean()

    slope, se = _ols(frame["d_log_price"].to_numpy(), frame["d_log_units"].to_numpy())
    return Elasticity(
        "Within product AND month", slope, se, len(frame), frame["StockCode"].nunique()
    )


def per_product_elasticities(panel: pd.DataFrame, *, min_months: int = 10) -> pd.DataFrame:
    """A separate elasticity for every product with enough of its own history.

    The distribution is the interesting object. A single catalogue-wide number implies every
    product responds the same way, and the spread here shows how far from true that is.
    """
    rows = []
    for code, group in panel.groupby("StockCode"):
        if len(group) < min_months or group["log_price"].std() < 1e-9:
            continue
        slope, se = _ols(group["log_price"].to_numpy(), group["log_units"].to_numpy())
        rows.append({"StockCode": code, "elasticity": slope, "se": se, "months": len(group)})
    return pd.DataFrame(rows)
