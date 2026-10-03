"""
margin.py
=========
Computes a 10-day 99% historical VaR-based initial margin for each asset and
portfolio, back-tests how often the actual 10-day loss exceeds that margin,
and illustrates a simple procyclicality floor: margin never drops below 50% of
its highest level in the previous 3 years.

Plain-English idea
------------------
Before you can trade on a derivatives exchange, the clearinghouse requires you
to post 'initial margin' — a buffer large enough to cover a bad 10-day move.
We estimate that buffer using the 99th percentile of overlapping 10-day losses
from the last 250 trading days.  The floor rule is a simple anti-procyclicality
device: it stops margin from falling too fast after a quiet period, so the
system is slightly more resilient when volatility returns.

DISCLAIMER: This is a simplified illustrative model. It is NOT the regulatory
SIMM (Standard Initial Margin Model) or any official industry margin model.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.var_models import rolling_var_nday


# ---------------------------------------------------------------------------
# Margin computation
# ---------------------------------------------------------------------------

def compute_margin(
    losses: pd.Series,
    margin_days: int = 10,
    window: int = 250,
    confidence: float = 0.99,
) -> pd.Series:
    """
    Compute the rolling n-day VaR-based initial margin.

    Parameters
    ----------
    losses      : Daily loss series (positive = loss).
    margin_days : Horizon in days (default 10).
    window      : Historical look-back window.
    confidence  : Confidence level.

    Returns
    -------
    margin : Series of margin levels (positive percentage of notional).
    """
    return rolling_var_nday(losses, n=margin_days, window=window, confidence=confidence)


def apply_floor(
    margin: pd.Series,
    floor_lookback: int = 756,
    floor_fraction: float = 0.50,
) -> pd.Series:
    """
    Apply a simple procyclicality floor: margin_floor = max(margin,
    floor_fraction * rolling_max(margin, floor_lookback)).

    This prevents margin from dropping below a fraction of its recent peak,
    keeping a buffer even after a long quiet spell.

    Parameters
    ----------
    margin         : Margin series from compute_margin().
    floor_lookback : Rolling window for the historical peak (default 3 years).
    floor_fraction : Fraction of the peak to enforce as the floor.

    Returns
    -------
    floored_margin : Series with the floor applied.
    """
    rolling_peak = margin.rolling(floor_lookback, min_periods=1).max()
    floor = floor_fraction * rolling_peak
    floored = np.maximum(margin, floor)
    return pd.Series(floored, index=margin.index, name="floored_margin")


# ---------------------------------------------------------------------------
# Margin back-test
# ---------------------------------------------------------------------------

def margin_breach(
    losses: pd.Series,
    margin: pd.Series,
    margin_days: int = 10,
) -> pd.Series:
    """
    Identify days when the actual n-day loss exceeded the margin set *today*.

    The margin set on day t is compared with the realised loss over days
    t+1 … t+margin_days (the forward realised loss).

    Parameters
    ----------
    losses      : Daily loss series.
    margin      : Margin series aligned to the same dates.
    margin_days : Horizon in days.

    Returns
    -------
    breach : Boolean Series, True when margin was breached.
    """
    # Forward realised loss: sum of next margin_days daily losses
    fwd_loss = losses.rolling(margin_days).sum().shift(-margin_days)
    both = pd.DataFrame({"fwd_loss": fwd_loss, "margin": margin}).dropna()
    return both["fwd_loss"] > both["margin"]


def margin_backtest_summary(
    all_losses: Dict[str, pd.Series],
    margin_days: int = 10,
    window: int = 250,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """
    Compute margin, back-test, and report breach rate for every series.

    Parameters
    ----------
    all_losses : {label: loss Series}.
    margin_days, window, confidence : Risk parameters.

    Returns
    -------
    df : DataFrame with columns [Series, Margin_breaches, Test_Days,
         Breach_Rate_pct, Avg_Margin_pct].
    """
    rows = []
    for label, ls in all_losses.items():
        m = compute_margin(ls, margin_days, window, confidence)
        breach = margin_breach(ls, m, margin_days)
        n_breach = int(breach.sum())
        n_days = int(breach.shape[0])
        rate = n_breach / n_days * 100 if n_days > 0 else np.nan
        avg_margin = float(m.dropna().mean())
        rows.append({
            "Series": label,
            "Breaches": n_breach,
            "Test_Days": n_days,
            "Breach_Rate_pct": round(rate, 3),
            "Avg_Margin_pct": round(avg_margin * 100, 4),
        })
    return pd.DataFrame(rows).set_index("Series")


def margin_breaches_per_year(
    losses: pd.Series,
    margin: pd.Series,
    margin_days: int = 10,
) -> pd.Series:
    """
    Count margin breaches for a single series grouped by calendar year.

    Parameters
    ----------
    losses, margin : Aligned loss and margin series.
    margin_days    : Horizon in days.

    Returns
    -------
    yearly : Series with years as index and breach counts as values.
    """
    breach = margin_breach(losses, margin, margin_days)
    breach = breach[breach]  # keep only True rows
    return breach.groupby(breach.index.year).count().rename("Margin_Breaches")


def floor_comparison_table(
    all_losses: Dict[str, pd.Series],
    margin_days: int = 10,
    window: int = 250,
    confidence: float = 0.99,
    floor_lookback: int = 756,
    floor_fraction: float = 0.50,
) -> pd.DataFrame:
    """
    Compare the base margin and the floored margin on breach rate and average
    level for each series.

    Parameters
    ----------
    all_losses : {label: loss Series}.
    All other parameters from config.

    Returns
    -------
    df : DataFrame comparing base and floor variants.
    """
    rows = []
    for label, ls in all_losses.items():
        m_base = compute_margin(ls, margin_days, window, confidence)
        m_floor = apply_floor(m_base, floor_lookback, floor_fraction)

        breach_base = margin_breach(ls, m_base, margin_days)
        breach_floor = margin_breach(ls, m_floor, margin_days)

        n_base = int(breach_base.sum())
        n_floor = int(breach_floor.sum())
        n_days = int(breach_base.shape[0])

        rows.append({
            "Series": label,
            "Base_Breach_Rate_pct": round(n_base / n_days * 100, 3) if n_days else np.nan,
            "Floor_Breach_Rate_pct": round(n_floor / n_days * 100, 3) if n_days else np.nan,
            "Base_Avg_Margin_pct": round(float(m_base.dropna().mean()) * 100, 4),
            "Floor_Avg_Margin_pct": round(float(m_floor.dropna().mean()) * 100, 4),
        })
    return pd.DataFrame(rows).set_index("Series")


def procyclicality_events(
    label: str,
    losses: pd.Series,
    margin_days: int = 10,
    window: int = 250,
    confidence: float = 0.99,
    floor_lookback: int = 756,
    floor_fraction: float = 0.50,
) -> pd.DataFrame:
    """
    Show how margin moved around the GFC (2008) and COVID (2020) crises.

    For each crisis, reports the margin 30 days before, at the peak during,
    and the ratio of peak to pre-crisis, for both the base and floored margin.

    Parameters
    ----------
    label  : Series label for display.
    losses : Loss series for one asset or portfolio.

    Returns
    -------
    df : Small table showing pre-crisis, peak, and multiple for each event.
    """
    m_base = compute_margin(losses, margin_days, window, confidence)
    m_floor = apply_floor(m_base, floor_lookback, floor_fraction)

    events = {
        "GFC 2008": ("2008-07-01", "2008-09-02", "2009-03-31"),
        "COVID 2020": ("2020-01-01", "2020-02-19", "2020-05-31"),
    }

    rows = []
    for event_label, (pre_start, crisis_start, crisis_end) in events.items():
        pre_window = m_base.loc[pre_start:crisis_start].dropna()
        crisis_window = m_base.loc[crisis_start:crisis_end].dropna()
        if pre_window.empty or crisis_window.empty:
            continue
        pre_val = float(pre_window.iloc[-1])
        peak_val = float(crisis_window.max())
        multiple = peak_val / pre_val if pre_val > 0 else np.nan

        pre_floor = float(m_floor.loc[pre_start:crisis_start].dropna().iloc[-1]) if not m_floor.loc[pre_start:crisis_start].dropna().empty else np.nan
        peak_floor = float(m_floor.loc[crisis_start:crisis_end].dropna().max()) if not m_floor.loc[crisis_start:crisis_end].dropna().empty else np.nan

        rows.append({
            "Series": label,
            "Event": event_label,
            "Pre_Crisis_Margin_pct": round(pre_val * 100, 3),
            "Peak_Margin_pct": round(peak_val * 100, 3),
            "Multiple": round(multiple, 2),
            "Pre_Crisis_Floored_pct": round(pre_floor * 100, 3) if not np.isnan(pre_floor) else np.nan,
            "Peak_Floored_pct": round(peak_floor * 100, 3) if not np.isnan(peak_floor) else np.nan,
        })

    return pd.DataFrame(rows)
