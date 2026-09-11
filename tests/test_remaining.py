"""Tests for projects 07-10: stability, overlapping windows, backtesting, elasticity.

Each block leads with the test that carries the project's claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for project in (
    "07-segmentation-stability",
    "08-overlapping-windows",
    "09-forecast-backtest",
    "10-price-elasticity",
):
    sys.path.insert(0, str(ROOT / "projects" / project))

from backtest import bias, mape, mase, rolling_origin, seasonal_naive  # noqa: E402
from elasticity import (  # noqa: E402
    naive_elasticity,
    product_month_panel,
    within_product_elasticity,
)
from overlap import (  # noqa: E402
    disjoint_indices,
    effective_sample_size,
    forward_return,
    newey_west_se,
    ordinary_least_squares,
)
from stability import fit_labels, shuffled_columns  # noqa: E402

# --- 07 · segmentation stability -------------------------------------------


def test_shuffling_columns_keeps_every_marginal_distribution():
    """The null has to be the right null: same distributions, no relationships."""
    rng = np.random.default_rng(0)
    X = np.c_[rng.lognormal(0, 1, 500), rng.exponential(2, 500)]
    shuffled = shuffled_columns(X, seed=1)

    for column in range(X.shape[1]):
        assert np.allclose(np.sort(X[:, column]), np.sort(shuffled[:, column]))


def test_shuffling_columns_destroys_the_correlation():
    rng = np.random.default_rng(1)
    base = rng.normal(size=2000)
    X = np.c_[base, base * 2 + rng.normal(0, 0.1, 2000)]
    assert abs(np.corrcoef(X.T)[0, 1]) > 0.95
    assert abs(np.corrcoef(shuffled_columns(X, seed=2).T)[0, 1]) < 0.1


def test_kmeans_returns_k_clusters_from_structureless_data():
    """The reason the stability check exists: the algorithm cannot say 'there are none'."""
    rng = np.random.default_rng(2)
    formless = rng.normal(size=(600, 3))
    assert len(np.unique(fit_labels(formless, 5, seed=0))) == 5


def test_well_separated_blobs_are_found_the_same_way_every_time():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(3)
    blobs = np.r_[
        rng.normal(-8, 0.3, (200, 2)), rng.normal(0, 0.3, (200, 2)), rng.normal(8, 0.3, (200, 2))
    ]
    assert adjusted_rand_score(fit_labels(blobs, 3, seed=0), fit_labels(blobs, 3, seed=9)) > 0.99


# --- 08 · overlapping windows ----------------------------------------------


def test_effective_sample_size_divides_by_the_horizon():
    """The whole argument, as one assertion."""
    assert effective_sample_size(1593, 120) == pytest.approx(13.3, abs=0.1)
    assert effective_sample_size(1000, 1) == 1000


def test_forward_return_annualises_correctly():
    prices = np.array([100.0] * 13)
    prices[12] = 200.0
    result = forward_return(prices, 12)
    assert result[0] == pytest.approx(1.0)  # doubled in one year


def test_forward_return_is_nan_where_the_window_runs_out():
    result = forward_return(np.arange(1.0, 21.0), 12)
    assert np.all(np.isnan(result[-12:]))
    assert np.all(np.isfinite(result[:8]))


def test_newey_west_is_wider_than_ordinary_least_squares_on_overlapping_data():
    """The correction must actually correct. Autocorrelated residuals are built here by
    construction, which is exactly what an overlapping window produces."""
    rng = np.random.default_rng(4)
    n = 600
    x = np.cumsum(rng.normal(size=n))  # persistent regressor
    noise = pd.Series(rng.normal(size=n)).rolling(24, min_periods=1).mean().to_numpy()
    y = 0.3 * x + noise * 6

    naive = ordinary_least_squares(x, y, method="naive")
    _, se_nw = newey_west_se(x, y, lags=23)
    assert se_nw > naive.standard_error


def test_disjoint_indices_never_share_a_window():
    idx = disjoint_indices(1000, 120)
    assert np.all(np.diff(idx) == 120)
    assert len(idx) == pytest.approx(1000 / 120, abs=1)


def test_significance_uses_the_independent_count_not_the_row_count():
    """Same slope, same standard error, different p - because n changed."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=400)
    y = 0.1 * x + rng.normal(size=400)

    many = ordinary_least_squares(x, y, method="a", n_independent=400)
    few = ordinary_least_squares(x, y, method="b", n_independent=6)
    assert many.t_statistic == pytest.approx(few.t_statistic)
    assert few.p_value > many.p_value


# --- 09 · forecast backtesting ---------------------------------------------


def test_mase_of_the_seasonal_naive_forecast_on_its_own_history_is_one():
    """The scale that makes MASE readable."""
    rng = np.random.default_rng(6)
    history = np.tile([10.0, 12, 14, 11, 13, 9, 8], 30) + rng.normal(0, 0.5, 210)
    series = pd.Series(history)
    forecast = seasonal_naive(series, 7, season=7)
    actual = seasonal_naive(series, 7, season=7)  # a perfect repeat
    assert mase(actual, forecast, history, season=7) == pytest.approx(0.0, abs=1e-9)


def test_a_worse_forecast_scores_above_one():
    rng = np.random.default_rng(11)
    history = np.tile([10.0, 20.0], 60) + rng.normal(0, 1.0, 120)
    actual = np.array([10.0, 20.0] * 3)
    terrible = np.full(6, 100.0)
    assert mase(actual, terrible, history, season=2) > 1.0


def test_a_perfectly_periodic_history_returns_nan_rather_than_dividing_by_zero():
    """Found by writing the test above with noiseless data.

    If the seasonal naive forecast is exactly right on the training data, the scale MASE
    divides by is zero. Returning `nan` says "this series gives me no yardstick"; returning
    `inf`, or silently flooring the denominator, would report a number that means nothing.
    """
    flawless = np.tile([10.0, 20.0], 60)
    assert np.isnan(mase(np.array([10.0, 20.0]), np.array([99.0, 99.0]), flawless, season=2))


def test_too_little_history_to_scale_is_nan_not_a_guess():
    assert np.isnan(mase(np.array([1.0]), np.array([2.0]), np.array([5.0, 6.0]), season=7))


def test_mape_is_dominated_by_small_actuals():
    """Two forecasts, identical absolute error, wildly different MAPE."""
    small = mape(np.array([2.0]), np.array([4.0]))
    large = mape(np.array([2000.0]), np.array([2002.0]))
    assert small == pytest.approx(1.0)
    assert large < 0.01


def test_mape_skips_zeroes_rather_than_dividing_by_them():
    assert np.isfinite(mape(np.array([0.0, 10.0]), np.array([1.0, 11.0])))
    assert np.isnan(mape(np.array([0.0, 0.0]), np.array([1.0, 1.0])))


def test_bias_catches_a_forecast_that_is_accurate_and_always_low():
    actual = np.array([100.0, 100, 100, 100])
    always_low = np.array([95.0, 95, 95, 95])
    assert bias(actual, always_low) == pytest.approx(-5.0)


def test_rolling_origin_never_trains_on_the_test_window():
    series = pd.Series(np.arange(300.0))
    for train, test in rolling_origin(series, horizon=7, step=7, min_train=100):
        assert train.index.max() < test.index.min()


def test_the_embargo_removes_the_last_training_points():
    series = pd.Series(np.arange(300.0))
    plain = next(iter(rolling_origin(series, horizon=7, step=7, min_train=100)))
    gapped = next(iter(rolling_origin(series, horizon=7, step=7, min_train=100, embargo=10)))
    assert len(gapped[0]) == len(plain[0]) - 10


def test_each_fold_trains_on_more_than_the_last():
    series = pd.Series(np.arange(300.0))
    sizes = [len(train) for train, _ in rolling_origin(series, horizon=7, step=7, min_train=100)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


# --- 10 · price elasticity --------------------------------------------------


def _catalogue() -> pd.DataFrame:
    """Two products with genuinely downward-sloping demand, at very different price levels.

    Constructed so the pooled regression must get it wrong: the expensive product sells
    fewer units *because it is a different product*, not because of its price.
    """
    rows = []
    for month in range(12):
        # cheap product: price drifts 1.00 -> 1.22, units fall with it
        price = 1.0 + 0.02 * month
        rows.append(("CHEAP", month, price, 1000 * price**-2.0))
        # expensive product: price drifts 50 -> 61, units fall with it
        price = 50.0 + 1.0 * month
        rows.append(("DEAR", month, price, 40 * (price / 50.0) ** -2.0))

    frame = pd.DataFrame(rows, columns=["StockCode", "m", "Price", "Quantity"])
    frame["InvoiceDate"] = pd.to_datetime("2010-01-01") + pd.to_timedelta(frame["m"] * 31, "D")
    frame["revenue"] = frame["Quantity"] * frame["Price"]
    frame["is_return"] = False
    return frame


def test_pooling_across_products_destroys_the_elasticity():
    """The project's claim, as a test. Both products have elasticity exactly -2.0 by
    construction; pooling them recovers something else entirely."""
    panel = product_month_panel(_catalogue(), min_months=6)
    pooled = naive_elasticity(panel)
    within = within_product_elasticity(panel)

    assert within.elasticity == pytest.approx(-2.0, abs=0.05)
    assert abs(pooled.elasticity - (-2.0)) > 0.5  # the pooled estimate misses it


def test_within_product_recovers_the_true_elasticity():
    panel = product_month_panel(_catalogue(), min_months=6)
    assert within_product_elasticity(panel).elasticity == pytest.approx(-2.0, abs=0.05)


def test_a_positive_elasticity_is_flagged_as_insane():
    from elasticity import Elasticity

    assert not Elasticity("x", 0.4, 0.1, 100, 10).sign_is_sane
    assert Elasticity("x", -1.4, 0.1, 100, 10).sign_is_sane


def test_returns_are_excluded_from_the_panel():
    frame = _catalogue()
    refund = frame.iloc[[0]].copy()
    refund["Quantity"] = -500
    refund["is_return"] = True
    with_refund = pd.concat([frame, refund], ignore_index=True)

    assert len(product_month_panel(with_refund, min_months=6)) == len(
        product_month_panel(frame, min_months=6)
    )


def test_price_is_revenue_weighted_not_a_mean_of_listed_prices():
    """500 sales at 1.00 and one at 9.99 is a 1.00 product, not a 5.50 one."""
    rows = [("X", 1.0, 500), ("X", 9.99, 1)]
    frame = pd.DataFrame(rows, columns=["StockCode", "Price", "Quantity"])
    frame["InvoiceDate"] = pd.to_datetime("2010-01-15")
    frame["revenue"] = frame["Quantity"] * frame["Price"]
    frame["is_return"] = False

    panel = product_month_panel(frame, min_months=1)
    assert panel["price"].iloc[0] == pytest.approx(1.018, abs=0.01)


def test_products_with_too_little_history_are_dropped():
    frame = _catalogue()
    brief = frame.iloc[:2].copy()
    brief["StockCode"] = "BRIEF"
    panel = product_month_panel(pd.concat([frame, brief], ignore_index=True), min_months=6)
    assert "BRIEF" not in set(panel["StockCode"])
