"""
backtest.py
===========
Back-tests historical VaR by counting 'exceptions' — days when the realised
loss exceeded the VaR forecast — and applies the Basel traffic-light framework
to classify model performance.

Plain-English idea
------------------
If our 99% VaR is working correctly, losses should exceed it on about 1% of
days.  We count how often they actually do (the exception rate), compare it to
1%, and use the Basel traffic-light system to flag periods where the model is
performing poorly.

No look-ahead: the VaR forecast on day t was computed using data up to day
t-1 only, so we are genuinely testing the model out-of-sample.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.var_models import rolling_var


# ---------------------------------------------------------------------------
# Exception detection
# ---------------------------------------------------------------------------

def find_exceptions(
    losses: pd.Series,
    var_series: pd.Series,
) -> pd.Series:
    """
    Return a boolean Series that is True on days when the realised loss
    exceeded the VaR forecast.

    Parameters
    ----------
    losses     : Daily realised losses (positive = loss).
    var_series : Rolling VaR forecast aligned to the same dates.

    Returns
    -------
    exceptions : Boolean Series, True on exception days.
    """
    both = pd.DataFrame({"loss": losses, "var": var_series}).dropna()
    return both["loss"] > both["var"]


def exception_summary(
    all_losses: Dict[str, pd.Series],
    window: int = 250,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """
    For every series compute the exception count, total test days, and rate.

    Parameters
    ----------
    all_losses : {label: loss Series}.
    window, confidence : Risk parameters.

    Returns
    -------
    df : DataFrame with columns [Series, Exceptions, Test_Days,
         Exception_Rate_pct, Expected_Rate_pct].
    """
    rows = []
    for label, ls in all_losses.items():
        var_s = rolling_var(ls, window, confidence)
        exc = find_exceptions(ls, var_s)
        n_exc = int(exc.sum())
        n_days = int(exc.shape[0])
        rate = n_exc / n_days * 100 if n_days > 0 else np.nan
        rows.append({
            "Series": label,
            "Exceptions": n_exc,
            "Test_Days": n_days,
            "Exception_Rate_pct": round(rate, 3),
            "Expected_Rate_pct": round((1 - confidence) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Series")


def exceptions_per_year(
    losses: pd.Series,
    var_series: pd.Series,
) -> pd.Series:
    """
    Count exceptions for a single series grouped by calendar year.

    Parameters
    ----------
    losses, var_series : Aligned loss and VaR series.

    Returns
    -------
    yearly : Series with years as index and exception counts as values.
    """
    exc = find_exceptions(losses, var_series)
    exc = exc[exc]  # keep only True rows
    return exc.groupby(exc.index.year).count().rename("Exceptions")


# ---------------------------------------------------------------------------
# Basel traffic-light zones
# ---------------------------------------------------------------------------

def traffic_light_zone(n_exceptions: int, green_max: int = 4, yellow_max: int = 9) -> str:
    """
    Map an exception count to a Basel traffic-light zone.

    Parameters
    ----------
    n_exceptions : Number of exceptions in a 250-day window.
    green_max    : Maximum exceptions for the green zone (default 4).
    yellow_max   : Maximum exceptions for the yellow zone (default 9).

    Returns
    -------
    zone : 'green', 'yellow', or 'red'.
    """
    if n_exceptions <= green_max:
        return "green"
    if n_exceptions <= yellow_max:
        return "yellow"
    return "red"


def rolling_traffic_light(
    losses: pd.Series,
    var_series: pd.Series,
    window: int = 250,
    green_max: int = 4,
    yellow_max: int = 9,
) -> pd.Series:
    """
    Compute the Basel traffic-light zone for every day using a rolling window
    of the past *window* days of exceptions.

    Parameters
    ----------
    losses, var_series : Loss and VaR series.
    window     : Rolling window for counting exceptions.
    green_max, yellow_max : Zone boundaries.

    Returns
    -------
    zones : Series of zone strings ('green', 'yellow', 'red').
    """
    exc = find_exceptions(losses, var_series).astype(int)
    rolling_count = exc.rolling(window).sum()
    zones = rolling_count.map(
        lambda n: traffic_light_zone(int(n), green_max, yellow_max)
        if not np.isnan(n)
        else np.nan
    )
    return zones


def traffic_light_summary(
    zones: pd.Series,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Summarise the percentage of time in each zone and list red-zone dates.

    Parameters
    ----------
    zones : Series of zone strings from rolling_traffic_light().

    Returns
    -------
    summary    : DataFrame with zone and percentage of days.
    red_dates  : DatetimeIndex of all red-zone days.
    """
    clean = zones.dropna()
    counts = clean.value_counts()
    total = len(clean)
    rows = []
    for zone in ["green", "yellow", "red"]:
        c = int(counts.get(zone, 0))
        rows.append({"Zone": zone.capitalize(), "Days": c, "Pct_of_time": round(c / total * 100, 2)})
    summary = pd.DataFrame(rows)
    red_dates = clean[clean == "red"].index
    return summary, red_dates


def backtest_cluster_note(exc_dates: pd.DatetimeIndex) -> str:
    """
    Return a plain-English note describing where exceptions cluster and why.

    Parameters
    ----------
    exc_dates : DatetimeIndex of exception days.

    Returns
    -------
    note : Multi-sentence plain-English explanation.
    """
    gfc_count = int(((exc_dates >= "2008-01-01") & (exc_dates <= "2009-06-30")).sum())
    covid_count = int(((exc_dates >= "2020-02-01") & (exc_dates <= "2020-05-31")).sum())
    note = (
        f"Exception clustering: {gfc_count} exceptions fell in 2008-2009 (the Global "
        f"Financial Crisis) and {covid_count} in early 2020 (the COVID-19 crash).  "
        "This is expected because the 250-day historical window did not contain any "
        "similarly extreme moves before those events, so the VaR threshold was far too "
        "low.  The model is slow to react to a sudden crash because it needs several "
        "days of extreme losses to raise its window before VaR climbs to the new level "
        "of market stress - a well-known weakness of short-memory historical simulation."
    )
    return note
