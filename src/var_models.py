"""
var_models.py
=============
Computes Historical Value-at-Risk (VaR) and Expected Shortfall (ES) using a
rolling window of past daily returns.  No distribution is assumed — we simply
sort the historical losses and read off the relevant percentile.

Plain-English idea
------------------
"Look at the worst 1% of days in the last 250 trading days.  The loss on the
best of those bad days is VaR.  The average loss across all those bad days is
ES (also called CVaR or Expected Shortfall)."

No look-ahead rule
------------------
VaR for day t is computed using only the 250 days ending on day t-1,
so the model never 'knows' what happens today.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core single-window functions (used in rolling computations)
# ---------------------------------------------------------------------------

def historical_var(losses: np.ndarray, confidence: float = 0.99) -> float:
    """
    Compute historical VaR for a single window of losses.

    Parameters
    ----------
    losses     : 1-D array of daily losses (positive = money lost).
    confidence : VaR confidence level, e.g. 0.99 for 99%.

    Returns
    -------
    var : The loss level exceeded on only (1 - confidence) of days.
          A positive number in the same units as the input.

    Example
    -------
    >>> historical_var(np.array([0.01, 0.02, 0.05, 0.10]), 0.75)
    0.075
    """
    if len(losses) == 0:
        return np.nan
    return float(np.percentile(losses, confidence * 100))


def historical_es(losses: np.ndarray, confidence: float = 0.99) -> float:
    """
    Compute historical Expected Shortfall for a single window of losses.

    ES is the average of the losses that exceed the VaR threshold.  It tells
    us 'given that we breached VaR, how bad was it on average?'

    Parameters
    ----------
    losses     : 1-D array of daily losses.
    confidence : Confidence level, e.g. 0.99.

    Returns
    -------
    es : Mean of losses in the tail beyond VaR.  NaN if the tail is empty.
    """
    if len(losses) == 0:
        return np.nan
    var = historical_var(losses, confidence)
    tail = losses[losses > var]
    if len(tail) == 0:
        return var   # degenerate case: return VaR itself
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Rolling VaR and ES
# ---------------------------------------------------------------------------

def rolling_var(
    losses: pd.Series,
    window: int = 250,
    confidence: float = 0.99,
) -> pd.Series:
    """
    Compute a rolling historical VaR series.

    For each day t, the VaR is estimated from the *previous* window days
    (strictly no look-ahead).

    Parameters
    ----------
    losses     : Daily loss series (positive = loss).
    window     : Number of past days to include.
    confidence : VaR confidence level.

    Returns
    -------
    var_series : Series aligned with *losses*, NaN for the first *window* days.
    """
    # Shift by 1 so day-t VaR uses data up to day t-1
    var_values = (
        losses
        .shift(1)
        .rolling(window)
        .quantile(confidence)
    )
    return var_values


def rolling_es(
    losses: pd.Series,
    window: int = 250,
    confidence: float = 0.99,
) -> pd.Series:
    """
    Compute a rolling historical Expected Shortfall series.

    For each day t, ES is the mean of past losses that exceeded the VaR
    threshold, using only data up to day t-1.

    Parameters
    ----------
    losses     : Daily loss series.
    window     : Rolling window size.
    confidence : Confidence level.

    Returns
    -------
    es_series : Series aligned with *losses*, NaN for the first *window* days.
    """
    shifted = losses.shift(1)

    es_values = []
    for i in range(len(shifted)):
        if i < window:
            es_values.append(np.nan)
            continue
        window_data = shifted.iloc[i - window : i].values
        es_values.append(historical_es(window_data, confidence))

    return pd.Series(es_values, index=losses.index, name=losses.name)


# ---------------------------------------------------------------------------
# Multi-day (overlapping) VaR for margin calculations
# ---------------------------------------------------------------------------

def rolling_var_nday(
    losses: pd.Series,
    n: int = 10,
    window: int = 250,
    confidence: float = 0.99,
) -> pd.Series:
    """
    Compute rolling VaR over an n-day horizon using overlapping windows.

    Each overlapping n-day loss is defined as the negative of the n-day
    cumulative return (i.e. sum of daily log-approx returns is close enough
    for percentage returns in normal conditions).

    Parameters
    ----------
    losses : Daily loss series.
    n      : Horizon in trading days (e.g. 10 for margin purposes).
    window : Historical look-back window in days.
    confidence : Confidence level.

    Returns
    -------
    var_nday : Series of n-day VaR estimates (NaN for early dates).
    """
    # Build n-day overlapping losses using rolling sum of returns
    # loss over n days = sum of daily losses (linear approximation)
    nday_losses = losses.rolling(n).sum().shift(1)

    var_values = (
        nday_losses
        .rolling(window)
        .quantile(confidence)
    )
    return var_values


# ---------------------------------------------------------------------------
# Snapshot summary (for the report)
# ---------------------------------------------------------------------------

def compute_var_es_summary(
    all_losses: Dict[str, pd.Series],
    window: int = 250,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """
    Compute the most-recent rolling VaR and ES for every series.

    Parameters
    ----------
    all_losses : {label: loss Series}.
    window     : Rolling window.
    confidence : Confidence level.

    Returns
    -------
    summary : DataFrame with columns [VaR_pct, ES_pct] and one row per series.
    """
    rows = []
    for label, ls in all_losses.items():
        var_series = rolling_var(ls, window, confidence)
        es_series = rolling_es(ls, window, confidence)
        last_var = var_series.dropna().iloc[-1] if var_series.dropna().shape[0] > 0 else np.nan
        last_es = es_series.dropna().iloc[-1] if es_series.dropna().shape[0] > 0 else np.nan
        rows.append({"Series": label, "VaR_1d_99pct": last_var, "ES_1d_99pct": last_es})

    return pd.DataFrame(rows).set_index("Series")


def diversification_table(
    asset_losses: Dict[str, pd.Series],
    portfolio_losses: Dict[str, pd.Series],
    window: int = 250,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """
    Build a table comparing portfolio VaR with the sum of individual VaRs.

    Shows the diversification benefit: a portfolio of assets that do not move
    in lockstep will have a VaR smaller than the weighted sum of individual
    VaRs because some bad days for one asset are good days for another.

    Parameters
    ----------
    asset_losses     : Dict of individual asset loss series.
    portfolio_losses : Dict of portfolio loss series.  Each value must be a
                       Series whose name starts with the portfolio label.
    window, confidence : Risk parameters.

    Returns
    -------
    df : DataFrame with columns [Portfolio, Sum_of_Ind_VaR, Portfolio_VaR,
         Diversification_Benefit_pct].
    """
    rows = []
    for port_label, port_ls in portfolio_losses.items():
        port_var = rolling_var(port_ls, window, confidence).dropna().iloc[-1]

        # We need to know the weights.  Instead of passing config here, we
        # compare against the simple undiversified sum (equal-risk concept).
        # The caller passes only the assets that make up this portfolio so we
        # just sum their individual last-day VaRs.
        sum_ind = sum(
            rolling_var(ls, window, confidence).dropna().iloc[-1]
            for ls in asset_losses.values()
        )
        benefit = (sum_ind - port_var) / sum_ind * 100 if sum_ind > 0 else 0
        rows.append({
            "Portfolio": port_label,
            "Sum_Individual_VaR": round(sum_ind, 6),
            "Portfolio_VaR": round(port_var, 6),
            "Diversification_Benefit_pct": round(benefit, 2),
        })

    note = (
        "Portfolio VaR is smaller than the sum of individual VaRs because "
        "losses across assets are not perfectly correlated: on a bad day for "
        "equities, bonds often rise, cushioning the total portfolio loss."
    )
    df = pd.DataFrame(rows)
    df.attrs["note"] = note
    return df
