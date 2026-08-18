"""Evaluation metrics: out-of-sample R-squared, and how many factors are earning it.

Both R-squared statistics compare realized returns against the model's
reconstruction with an uncentered denominator (the total sum of squared returns),
following the autoencoder asset-pricing convention:

    total R^2      = 1 - sum (r - beta . f_t)^2          / sum r^2
    predictive R^2 = 1 - sum (r - beta . lambda_{t-1})^2 / sum r^2

where ``f_t`` is the fitted factor for period ``t`` and ``lambda_{t-1}`` forecasts
it from the factors of *earlier* periods only. Which forecast is a configuration
choice — the prevailing (expanding) mean of the factor history, or an exponentially
weighted one that lets a drifting risk premium be tracked rather than averaged —
and both are one :class:`FactorForecast` at different decays.

That forecast is state, and it crosses split boundaries: a fold warms it on the
training window, carries it through validation and opens the test window with what
validation left, which is a single walk in date order with two joins in it rather
than three unrelated statistics. A period reached before any factor has been
observed is *excluded* from the predictive statistic rather than scored against
nothing.

Two consequences of carrying state rather than recomputing it are worth stating
because nothing here rejects them. Under a decaying forecast, a state carried
across a purged fold boundary is stale by the purge width, since the periods
dropped there are dropped from the walk too. And a non-finite factor is permanent:
the decay is positive, so an ``inf`` entering the running total stays ``inf`` in
every later forecast and a ``nan`` stays ``nan`` — a decaying forecast does not
eventually forget it. Downstream that is an infinite or undefined squared error, so
the predictive statistic lands at ``-inf`` or ``nan`` rather than quietly at a
plausible number; a diverged model announces itself there and in ``factors/*``, as
it always did.

:func:`factor_diagnostics` answers the question R-squared cannot: whether the K
factors the model was given are all doing work, or whether it has quietly
collapsed onto fewer. Both statistics there are chosen to survive the model's
scale indeterminacy — see that function.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

PERIOD_COLUMNS = ("date", "n_stocks", "sse_total", "sst_total", "sse_pred", "sst_pred")
"""The authoritative fields of a per-period error table; everything else derives."""


@dataclass
class RSquared:
    """Total and predictive R-squared over an evaluation split."""

    total: float
    predictive: float


# --------------------------------------------------------------------------- #
# The forecast the predictive statistic is scored against
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False, repr=False)
class FactorForecast:
    """The running one-period-ahead factor forecast ``lambda_{t-1}``.

    One recursion covers both estimators — a decayed running total over the decayed
    sum of the weights that built it::

        total_n   = decay * total_{n-1}   + weight * f_n
        divisor_n = decay * divisor_{n-1} + weight
        lambda_n  = total_n / divisor_n

    seeded on the first factor with ``total_1 = f_1`` and ``divisor_1 = 1``, so
    either estimator opens forecasting ``f_1`` itself rather than a fraction of it.

    ``expanding_mean`` is ``decay = weight = 1``: the total is the plain sum, the
    divisor is the period count, and the level is the prevailing mean of every
    factor observed. It weights a factor from five years ago exactly as heavily as
    last period's, so after a 1260-period training window a new factor moves it by
    under a tenth of a percent. This is the estimator the model has always been
    scored under, and walking a split here is not merely equivalent to the
    sum-over-count it replaces but bit-identical to it, because multiplying a float
    by ``1.0`` is exact. Chaining two splits is *not* the same float as adding their
    two subtotals, which is how the seed used to be assembled — floating-point
    addition does not associate — so a test window's predictive R-squared moves in
    its last few units in the last place even under this mode.

    ``ewma`` is ``decay = 1 - alpha``, ``weight = alpha``. The divisor has fixed
    point ``1`` and is seeded there, so the recursion collapses to
    ``lambda_n = (1 - alpha) lambda_{n-1} + alpha f_n`` — pandas'
    ``ewm(alpha=..., adjust=False)``. A factor ``k`` periods old carries weight
    ``(1 - alpha)^k``; ``alpha`` converts from the other common parameterizations as
    ``alpha = 2 / (span + 1)`` and ``alpha = 1 - 0.5 ** (1 / halflife)``. At
    ``alpha = 1`` it degenerates to "last period's factor", a legitimate
    random-walk forecast.

    Carrying a divisor rather than a bare level is what makes the two one family,
    and it removes a class of seeding bug by construction: a total cannot exist
    without the divisor that divides it, which is the "a sum supplied without a
    count" failure this module used to guard against by hand.

    Immutable, so :meth:`observe` returns a new forecast rather than advancing this
    one. A split scored twice from one state gives the same answer twice, and a
    state handed to two splits cannot be moved by one of them behind the other's
    back — which matters precisely because a fold's validation and test windows are
    seeded from states one split's walk apart. ``eq=False`` because the array field
    would make a generated ``__eq__`` ambiguous and a generated ``__hash__``
    unusable; compare :meth:`level` instead.
    """

    decay: float
    weight: float
    total: FloatArray | None = None
    divisor: float = 0.0
    periods: int = 0

    def __post_init__(self) -> None:
        # Zero is admitted: it is ``alpha = 1``, the forecast that is simply last
        # period's factor, and the divisor stays at one there rather than collapsing.
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"FactorForecast decay must lie in [0, 1]; got {self.decay}.")
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"FactorForecast weight must lie in (0, 1]; got {self.weight}.")
        # A total and the divisor that divides it, or neither. Half a state is the
        # one way this object could still forecast a premium several times too
        # large, so it is refused at construction rather than checked at use.
        if (self.total is None) != (self.periods == 0):
            raise ValueError(
                "FactorForecast carries a total and a period count or neither; got "
                f"total={'None' if self.total is None else 'set'}, periods={self.periods}."
            )
        if (self.total is None) != (self.divisor == 0.0):
            raise ValueError(
                "FactorForecast carries a total and a divisor or neither; got "
                f"total={'None' if self.total is None else 'set'}, divisor={self.divisor}."
            )

    @classmethod
    def expanding_mean(cls) -> FactorForecast:
        """The prevailing mean of every factor observed, with no history yet."""
        return cls(decay=1.0, weight=1.0)

    @classmethod
    def ewma(cls, alpha: float) -> FactorForecast:
        """An exponentially weighted forecast, with no history yet.

        ``alpha`` is the weight the newest factor enters with, so a larger value
        forgets faster; see the class docstring for the conversions from ``span``
        and ``halflife``.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"ewma alpha must lie in (0, 1]; got {alpha}.")
        # 1 - alpha rather than a separately configured decay: the pair has to sum
        # to one for the divisor to settle at one, which is what makes the level the
        # textbook EWMA recursion instead of a renormalized weighted sum.
        return cls(decay=1.0 - alpha, weight=alpha)

    def observe(self, factor: FloatArray) -> FactorForecast:
        """This forecast advanced by one period's factor; call *after* scoring it.

        The float64 cast is here rather than at the call sites: this is the only
        door a float32 model output can use to reach the accumulated state.
        """
        value = np.asarray(factor, dtype=np.float64)
        if self.total is None:
            return FactorForecast(self.decay, self.weight, value.copy(), 1.0, 1)
        if value.shape != self.total.shape:
            raise ValueError(
                f"factor of shape {value.shape} does not match the forecast's {self.total.shape}."
            )
        return FactorForecast(
            self.decay,
            self.weight,
            self.decay * self.total + self.weight * value,
            self.decay * self.divisor + self.weight,
            self.periods + 1,
        )

    def extended(self, factors: Iterable[FloatArray]) -> FactorForecast:
        """This forecast advanced over ``factors``, in date order.

        Walking one split and then the next gives exactly the state one walk over
        their concatenation gives, which is what makes a fold's train -> validation
        -> test chain well defined.
        """
        state = self
        for factor in factors:
            state = state.observe(factor)
        return state

    def level(self) -> FloatArray | None:
        """``lambda`` from everything observed so far, or ``None`` while nothing is.

        A freshly allocated array the caller owns.
        """
        if self.total is None:
            return None
        return self.total / self.divisor

    def __repr__(self) -> str:
        estimator = (
            "expanding_mean"
            if self.decay == 1.0 and self.weight == 1.0
            else f"ewma(alpha={self.weight:g})"
        )
        return f"FactorForecast({estimator}, periods={self.periods})"


@dataclass
class SplitScore:
    """A split's per-period error rows, and the forecast state it ends in.

    The two travel together because the next split in chronological order is seeded
    from that state: returning it beside the rows it produced makes the seed
    *provably* the state those rows were scored under, where recomputing it would
    leave a second copy of the same recursion for a divergence to hide in — and the
    divergence would land directly in ``test_predictive_r2``.
    """

    rows: pd.DataFrame
    forecast: FactorForecast


# --------------------------------------------------------------------------- #
# R-squared
# --------------------------------------------------------------------------- #


def r_squared(
    betas: Sequence[FloatArray],
    factors: Sequence[FloatArray],
    returns: Sequence[FloatArray],
    *,
    forecast: FactorForecast,
) -> RSquared:
    """Total and predictive R-squared from per-period model outputs in date order.

    Parameters
    ----------
    betas:
        Per-period factor loadings, each of shape ``[N_t, K]``.
    factors:
        Per-period fitted latent factors, each of shape ``[K]``.
    returns:
        Per-period realized returns, each of shape ``[N_t]``.
    forecast:
        The state earlier splits left the factor forecast in, which is what makes
        the predictive statistic reflect the full factor history rather than only
        this split's. Required, and required to name its estimator: a default would
        be a way to score an ``ewma`` run with an expanding mean and never notice.
        :meth:`FactorForecast.expanding_mean` is the cold start.
    """
    total_sse, total_sst, pred_sse, pred_sst, _ = _period_errors(
        betas, factors, returns, forecast=forecast
    )
    return RSquared(
        total=_ratio(total_sse, total_sst),
        predictive=_ratio(pred_sse, pred_sst),
    )


def _period_errors(
    betas: Sequence[FloatArray],
    factors: Sequence[FloatArray],
    returns: Sequence[FloatArray],
    *,
    forecast: FactorForecast,
) -> tuple[list[float], list[float], list[float], list[float], FactorForecast]:
    """Each period's contribution to the four sums the two statistics are ratios of.

    Split out so :func:`r_squared` and :func:`per_period_errors` cannot drift. The
    forecast bookkeeping is the part that is easy to reimplement almost right: a
    period is scored predictively only once a forecast built from *earlier* factors
    exists, so a split that inherited no history never scores its first period, and
    a split seeded with an earlier one's history scores all of them. An unscored
    period contributes zero to the predictive numerator *and* to its denominator —
    it is excluded from the statistic, not counted as a perfect or a failed
    forecast. Reading the level before observing the period's own factor is what
    keeps the statistic out of sample.

    The advanced forecast is returned rather than left for the caller to rebuild,
    so the state the next split is seeded from is the state these rows were scored
    under, by construction.

    A zero added to a running total leaves it unchanged, so summing these
    sequentially — which is what :func:`_ratio` does — reproduces an accumulation
    inside this loop bit for bit. :func:`pooled_r_squared` sums a column instead,
    which pandas does pairwise, so it can differ from :func:`r_squared` in the last
    unit in the last place.
    """
    total_sse: list[float] = []
    total_sst: list[float] = []
    pred_sse: list[float] = []
    pred_sst: list[float] = []
    state = forecast

    for beta, factor, ret in zip(betas, factors, returns, strict=True):
        total_sse.append(float(np.sum((ret - beta @ factor) ** 2)))
        total_sst.append(float(np.sum(ret**2)))
        prevailing = state.level()
        if prevailing is None:
            pred_sse.append(0.0)
            pred_sst.append(0.0)
        else:
            pred_sse.append(float(np.sum((ret - beta @ prevailing) ** 2)))
            pred_sst.append(float(np.sum(ret**2)))
        state = state.observe(factor)

    return total_sse, total_sst, pred_sse, pred_sst, state


def _ratio(sse: Sequence[float], sst: Sequence[float]) -> float:
    """``1 - sum(sse) / sum(sst)``, or ``nan`` when nothing was scored."""
    numerator = 0.0
    denominator = 0.0
    for error, total in zip(sse, sst, strict=True):
        numerator += error
        denominator += total
    return 1.0 - numerator / denominator if denominator > 0.0 else float("nan")


def per_period_errors(
    betas: Sequence[FloatArray],
    factors: Sequence[FloatArray],
    returns: Sequence[FloatArray],
    dates: Sequence[pd.Timestamp],
    *,
    forecast: FactorForecast,
) -> SplitScore:
    """One row per period — the squared errors and the totals they are scored against
    — and the forecast state the split ends in.

    The sums, not the ratios, are what is recorded. An R-squared over a set of
    periods is a ratio of sums, so it cannot be recovered from a set of per-period
    R-squareds — each has a different denominator — and pooling folds, years or
    regimes after the fact is only possible from the sums. ``total_r2`` and
    ``predictive_r2`` are included for convenience but are derived; see
    :func:`pooled_r_squared` for the aggregation that is correct.

    ``sst_pred`` is zero for a period the predictive statistic does not score, so
    pooling is a plain sum over both columns with no condition to remember.
    """
    total_sse, total_sst, pred_sse, pred_sst, state = _period_errors(
        betas, factors, returns, forecast=forecast
    )
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(list(dates)),
            # Typed explicitly: inferring from an empty list would give float64,
            # and concatenating an empty split with real ones then promotes the
            # column for every fold.
            "n_stocks": np.array([np.asarray(ret).shape[0] for ret in returns], dtype=np.int64),
            "sse_total": total_sse,
            "sst_total": total_sst,
            "sse_pred": pred_sse,
            "sst_pred": pred_sst,
        }
    )
    frame["total_r2"] = _safe_ratio(frame["sse_total"], frame["sst_total"])
    frame["predictive_r2"] = _safe_ratio(frame["sse_pred"], frame["sst_pred"])
    return SplitScore(rows=frame, forecast=state)


def pooled_r_squared(frame: pd.DataFrame) -> RSquared:
    """The R-squared of every period in ``frame``, pooled.

    The ratio of the sums, never the mean of the per-period ratios: periods differ
    in how many stocks they price and in how much return variation they carry, and
    averaging their ratios would weight a thin, quiet cross-section like a large,
    volatile one. The same argument applies across folds, which is why a
    walk-forward headline number is this function over every fold's rows at once
    rather than the mean of the folds' scores.

    Sums a column rather than accumulating in order, so it can disagree with
    :func:`r_squared` over the same periods in the last unit in the last place.
    """
    return RSquared(
        total=_pooled(frame, "sse_total", "sst_total"),
        predictive=_pooled(frame, "sse_pred", "sst_pred"),
    )


def _pooled(frame: pd.DataFrame, sse: str, sst: str) -> float:
    denominator = float(frame[sst].sum())
    return 1.0 - float(frame[sse].sum()) / denominator if denominator > 0.0 else float("nan")


def _safe_ratio(sse: pd.Series, sst: pd.Series) -> pd.Series:
    """``1 - sse / sst`` per row, ``nan`` where the row was not scored."""
    return (1.0 - sse / sst.where(sst > 0.0)).astype("float64")


# --------------------------------------------------------------------------- #
# Factor diagnostics: how much of K the model actually uses
# --------------------------------------------------------------------------- #


@dataclass
class FactorDiagnostics:
    """How much of the model's factor capacity a fitted model actually uses."""

    effective: float
    max_abs_correlation: float


def factor_diagnostics(
    betas: Sequence[FloatArray], factors: Sequence[FloatArray]
) -> FactorDiagnostics:
    """How many of the ``K`` latent factors are carrying the reconstruction.

    A conditional autoencoder given more factors than the data supports does not
    fail loudly: it collapses, leaving some factors contributing nothing and others
    duplicating one another, while the R-squared barely moves. Both statistics here
    detect that, and both are chosen to be meaningful despite the model's scale
    indeterminacy — ``beta_k -> beta_k / c``, ``f_k -> c * f_k`` leaves the fitted
    returns untouched, so anything read off the size of ``beta`` or of ``f`` alone
    (a factor's variance, a loading's norm) says nothing about the model.

    ``effective`` is the participation ratio ``1 / sum_k s_k^2`` of the per-factor
    contribution shares

        s_k = sum_t sum_n (beta_{t,n,k} f_{t,k})^2 / sum_j sum_t sum_n (...)^2,

    which is built from the products ``beta_{t,n,k} f_{t,k}`` and is therefore
    invariant to that rescaling. It runs from 1, when a single factor does all the
    work, to ``K``, when all contribute equally: well below ``K`` means the model is
    paying for factors it does not use, and ``num_factors`` (or the L1 penalty) is
    the knob. It is a summary of contribution concentration, not a variance
    decomposition — the per-factor contributions are not orthogonal, so the shares
    do not partition the explained variance.

    ``max_abs_correlation`` is the largest absolute pairwise correlation between the
    estimated factor series ``{f_{t,k}}_t``, which is invariant to per-factor
    rescaling because correlation is. Approaching 1 means two factors are the same
    factor twice, again pointing at a smaller ``K``. A factor that is constant over
    the window has no defined correlation and is skipped; the collapse it
    represents is what ``effective`` reports.

    Returns ``nan`` for a statistic the split is too small to define — fewer than
    two periods or fewer than two factors for the correlation, an all-zero
    reconstruction for the participation ratio — which the tracker drops.
    """
    if not betas:
        return FactorDiagnostics(effective=float("nan"), max_abs_correlation=float("nan"))

    num_factors = int(np.asarray(factors[0]).shape[0])
    contributions = np.zeros(num_factors, dtype=np.float64)
    for beta, factor in zip(betas, factors, strict=True):
        contributions += np.sum((beta * factor) ** 2, axis=0)

    total = float(contributions.sum())
    if total > 0.0:
        shares = contributions / total
        effective = 1.0 / float(np.sum(shares**2))
    else:
        effective = float("nan")

    return FactorDiagnostics(
        effective=effective,
        max_abs_correlation=_max_abs_correlation(np.stack(factors)),
    )


def _max_abs_correlation(factor_series: FloatArray) -> float:
    """Largest absolute off-diagonal correlation of a ``[W, K]`` factor panel.

    Factors that are constant over the window are dropped first: their correlation
    is undefined, and letting :func:`numpy.corrcoef` return ``nan`` for them would
    propagate through the maximum and hide every real correlation in the panel.
    """
    periods, num_factors = factor_series.shape
    if periods < 2 or num_factors < 2:
        return float("nan")
    varying = factor_series[:, factor_series.std(axis=0) > 0.0]
    if varying.shape[1] < 2:
        return float("nan")
    correlations = np.asarray(np.corrcoef(varying, rowvar=False), dtype=np.float64)
    upper = np.triu_indices(varying.shape[1], k=1)
    return float(np.max(np.abs(correlations[upper])))
