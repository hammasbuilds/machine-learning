"""Actuarial pricing: frequency times severity, and the offset everybody forgets.

678,013 real motor policies. 95% of them never claim. The 5% that do produce claims
ranging from a few hundred euros to a few million.

Pricing that is **two models, not one**:

    pure premium  =  E[claims per year]  x  E[cost per claim]
                     ---- frequency ----     ---- severity ----

A single model on total loss can fit the same data, and insurers mostly do not use one.
The reason is not accuracy — it is that when next year's loss ratio moves, somebody has to
say whether people are **crashing more often** or **crashing more expensively**, and those
have completely different responses. A combined model cannot answer the question.

**The trap this file exists for, and it is not subtle.** Policies are not observed for the
same length of time. One is in force for 12 months, the next for 3 weeks. Model
`ClaimNb` directly and you learn that long policies have more claims — which is true,
useless, and will price a one-month policy as though it were safe.

The fix is one argument:

    offset = log(Exposure)

which turns a model of *counts* into a model of *rates*. It is one line, it is the
difference between a working price and a broken one, and it is omitted constantly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Feature engineering follows the actuarial convention: continuous risk factors are banded,
# because the relationship with risk is rarely linear and a banded factor is something an
# underwriter can read off a table and argue with.
DRIVER_AGE_BANDS = [17, 21, 26, 31, 41, 51, 61, 71, 100]
VEHICLE_AGE_BANDS = [-1, 1, 4, 9, 14, 100]
BONUS_MALUS_BANDS = [49, 51, 61, 76, 101, 300]


def band(series: pd.Series, edges: list[int], prefix: str) -> pd.Series:
    """Cut a continuous factor into actuarial bands, as an ordered category."""
    labels = [f"{prefix}{edges[i] + 1}-{edges[i + 1]}" for i in range(len(edges) - 1)]
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)


def prepare(frequency: pd.DataFrame) -> pd.DataFrame:
    """Band the continuous factors and drop the rows that cannot be priced."""
    frame = frequency.copy()

    # Zero exposure is not a cheap policy, it is a policy that was never on risk. Keeping
    # it means dividing by zero when converting counts to rates.
    frame = frame[frame["Exposure"] > 0]

    frame["driver_band"] = band(frame["DrivAge"], DRIVER_AGE_BANDS, "age")
    frame["vehicle_band"] = band(frame["VehAge"], VEHICLE_AGE_BANDS, "veh")
    frame["bonus_band"] = band(frame["BonusMalus"], BONUS_MALUS_BANDS, "bm")
    frame["power_band"] = frame["VehPower"].clip(4, 12).astype(int).astype(str)
    frame["density_band"] = pd.qcut(frame["Density"], 5, labels=[f"dens{i}" for i in range(1, 6)])

    return frame.dropna(subset=["driver_band", "vehicle_band", "bonus_band", "density_band"])


def design(frame: pd.DataFrame, factors: list[str]) -> tuple[np.ndarray, list[str]]:
    """Dummy-coded design matrix with an intercept, dropping one level per factor.

    The dropped level is the base class, and every other coefficient is read relative to it
    — which is exactly how a rating table is published: a base rate, then multipliers.
    """
    dummies = pd.get_dummies(frame[factors], drop_first=True, dtype=float)
    matrix = sm.add_constant(dummies.to_numpy(), has_constant="add")
    return matrix, ["intercept", *dummies.columns.tolist()]


@dataclass
class Model:
    """A fitted GLM, with the pieces needed to read it as a rating table."""

    name: str
    family: str
    columns: list[str]
    coefficients: np.ndarray
    deviance: float
    n_observations: int

    def relativities(self) -> pd.DataFrame:
        """Coefficients as multipliers on the base rate. The form underwriters use.

        A log link means `exp(beta)` is a multiplier: 1.35 is "35% more claims than the
        base class". Reporting the raw coefficient instead is how a pricing discussion
        stops being a discussion.
        """
        return pd.DataFrame(
            {
                "factor": self.columns,
                "coefficient": self.coefficients,
                "relativity": np.exp(self.coefficients),
            }
        ).sort_values("relativity", ascending=False)

    def predict(self, X: np.ndarray, *, exposure: np.ndarray | None = None) -> np.ndarray:
        rate = np.exp(X @ self.coefficients)
        return rate * exposure if exposure is not None else rate


def fit_frequency(frame: pd.DataFrame, factors: list[str], *, use_offset: bool = True) -> Model:
    """Poisson GLM on claim counts, with log(exposure) as an offset.

    Poisson because claims are counts. The offset because they are counts observed over
    *different lengths of time*, and `offset=log(exposure)` is what turns

        log(E[count]) = X.beta            (a model of counts)

    into

        log(E[count] / exposure) = X.beta  (a model of rates)

    `use_offset=False` exists to be wrong on purpose, and the results show what it costs.
    """
    X, columns = design(frame, factors)
    y = frame["ClaimNb"].to_numpy(dtype=float)
    offset = np.log(frame["Exposure"].to_numpy(dtype=float)) if use_offset else None

    fitted = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    return Model(
        name="Poisson frequency" + ("" if use_offset else " (NO exposure offset)"),
        family="Poisson",
        columns=columns,
        coefficients=np.asarray(fitted.params),
        deviance=float(fitted.deviance),
        n_observations=len(y),
    )


def fit_severity(claims: pd.DataFrame, factors: list[str]) -> Model:
    """Gamma GLM on the cost of a claim, fitted only on policies that had one.

    Gamma because claim amounts are positive, right-skewed and have variance that grows
    with the mean — a normal model here predicts negative claim costs for the cheapest
    risks, which is not a rounding error but a nonsense.

    Fitted on claims only. Including the 95% of policies with no claim would be modelling
    the cost of a claim that did not happen.
    """
    X, columns = design(claims, factors)
    y = claims["ClaimAmount"].to_numpy(dtype=float)

    fitted = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    return Model(
        name="Gamma severity",
        family="Gamma",
        columns=columns,
        coefficients=np.asarray(fitted.params),
        deviance=float(fitted.deviance),
        n_observations=len(y),
    )


def fit_tweedie(frame: pd.DataFrame, factors: list[str], *, power: float = 1.5) -> Model:
    """One Tweedie GLM straight onto the loss per exposure-year.

    Tweedie with 1 < p < 2 is a compound Poisson-Gamma: exactly a Poisson number of Gamma
    claims. It fits the same structure in a single model and is a perfectly reasonable way
    to price.

    What it cannot do is decompose. When the loss ratio moves next year this model says
    "risk went up"; the two-model version says whether people crashed **more often** or
    **more expensively**, which are different problems with different answers.
    """
    X, columns = design(frame, factors)
    exposure = frame["Exposure"].to_numpy(dtype=float)
    y = frame["total_loss"].to_numpy(dtype=float) / exposure

    fitted = sm.GLM(
        y,
        X,
        family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=power),
        freq_weights=exposure,  # a policy on risk for a year counts more than one for a week
    ).fit()
    return Model(
        name=f"Tweedie pure premium (p={power})",
        family="Tweedie",
        columns=columns,
        coefficients=np.asarray(fitted.params),
        deviance=float(fitted.deviance),
        n_observations=len(y),
    )


# --- how good is a price? -------------------------------------------------


def lift_table(
    predicted: np.ndarray, actual: np.ndarray, exposure: np.ndarray, *, bins: int = 10
) -> pd.DataFrame:
    """Sort policies by predicted risk, bucket them, and compare to what happened.

    The actuarial equivalent of a calibration plot, and the only chart a pricing committee
    actually looks at. A working model produces a monotone staircase: the tenth the model
    called cheapest really is cheapest.

    Everything is per exposure-year. Comparing raw totals across deciles would rank on how
    long the policies were in force.
    """
    order = np.argsort(predicted)
    edges = np.linspace(0, len(order), bins + 1).astype(int)

    rows = []
    for b in range(bins):
        take = order[edges[b] : edges[b + 1]]
        years = exposure[take].sum()
        rows.append(
            {
                "decile": b + 1,
                "policies": len(take),
                "exposure_years": years,
                "predicted": predicted[take].sum() / years,
                "actual": actual[take].sum() / years,
            }
        )

    table = pd.DataFrame(rows)
    table["ratio"] = table["actual"] / table["predicted"].replace(0, np.nan)
    return table


def gini(predicted: np.ndarray, actual: np.ndarray, exposure: np.ndarray) -> float:
    """Ordered Lorenz / Gini on losses. How well the price separates good risks from bad.

    0 means the ranking is random. Higher is better discrimination. A pricing model with a
    good Gini and bad calibration still has a use - it sorts risks correctly and the base
    rate can be rescaled. The reverse is not true.

    **Ranks by predicted RATE, not predicted count**, and that distinction produced a wrong
    answer here before it was fixed. Exposure in this portfolio spans 0.003 to 1.0 years -
    a factor of 300 - while the risk relativities span about 3. Sort by predicted count and
    you have sorted by exposure with a rounding error attached, which measures how long
    policies were in force and calls it discrimination. It scored the correct model at
    -0.03 and the broken one at 0.22.
    """
    rate = np.asarray(predicted, dtype=float) / np.clip(exposure, 1e-9, None)
    order = np.argsort(rate)
    losses = actual[order]
    years = exposure[order]

    cumulative_loss = np.cumsum(losses) / max(losses.sum(), 1e-12)
    cumulative_exposure = np.cumsum(years) / max(years.sum(), 1e-12)
    return float(1 - 2 * np.trapezoid(cumulative_loss, cumulative_exposure))


def poisson_deviance(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean Poisson deviance. The right error measure for counts; MSE is not."""
    predicted = np.clip(predicted, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(actual > 0, actual * np.log(actual / predicted), 0.0)
    return float(2 * np.mean(term - (actual - predicted)))


def credible_bands(
    predicted: np.ndarray,
    actual: np.ndarray,
    exposure: np.ndarray,
    *,
    edges: list[float] | None = None,
    min_exposure_years: float = 500.0,
) -> pd.DataFrame:
    """Observed against predicted frequency, by exposure band, with thin bands flagged.

    The second thing that went wrong here, and the more interesting one.

    A policy on risk for 0.02 years with one claim has an *empirical* frequency of fifty
    claims a year. Bucket those policies together and the band appears to run at 1.06
    claims per year against a portfolio average of 0.10 - a tenfold "failure" of the model
    that is nothing but a small denominator.

    The tell is in the other column: those same policies have the **lowest** probability of
    claiming at all (2.8%, against 6.7% for full-year policies). They are not dangerous.
    They are briefly observed.

    So every band carries the exposure behind it, and bands under `min_exposure_years` are
    marked `credible=False` rather than quietly plotted at the same weight as the rest.
    This is the actuarial idea of credibility in its simplest form: an observed rate is
    only evidence in proportion to the exposure that produced it.
    """
    edges = edges or [0.0, 0.05, 0.15, 0.35, 0.6, 0.85, 1.0]
    bands = pd.cut(exposure, bins=edges, include_lowest=True)

    frame = pd.DataFrame(
        {"band": bands, "predicted": predicted, "actual": actual, "exposure": exposure}
    )
    grouped = frame.groupby("band", observed=True).agg(
        policies=("actual", "size"),
        exposure_years=("exposure", "sum"),
        claims=("actual", "sum"),
        predicted_claims=("predicted", "sum"),
    )

    grouped["observed_rate"] = grouped["claims"] / grouped["exposure_years"]
    grouped["predicted_rate"] = grouped["predicted_claims"] / grouped["exposure_years"]
    grouped["share_claiming"] = (
        frame.assign(had=frame["actual"] > 0).groupby("band", observed=True)["had"].mean()
    )
    grouped["credible"] = grouped["exposure_years"] >= min_exposure_years
    return grouped.reset_index()
