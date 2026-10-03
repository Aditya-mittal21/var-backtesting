"""
data_loader.py
==============
Downloads daily adjusted-close prices from Yahoo Finance, caches them to
data/prices.csv, aligns all series on common trading dates, applies a short
forward-fill for weekends/holidays, and builds two portfolio return series:
an equal-weight portfolio and a 60/40 (SPY/TLT) portfolio.

Plain-English idea
------------------
We pull historical prices, compute "what percentage did each asset move each
day?", then combine those daily moves using portfolio weights to get a single
daily return number for each portfolio.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load the YAML configuration file and return it as a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """
    Download adjusted-close prices for a list of tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : list of ticker symbols recognised by Yahoo Finance.
    start   : ISO date string, e.g. '2007-01-01'.

    Returns
    -------
    DataFrame with one column per ticker and dates as the index.
    """
    print(f"Downloading prices from Yahoo Finance ({start} -> today)...")
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
    )
    # yfinance returns a MultiIndex when >1 ticker; pick 'Close'
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices


def _align_and_fill(prices: pd.DataFrame, max_ffill: int) -> pd.DataFrame:
    """
    Keep only dates where at least one ticker has data, forward-fill gaps
    of at most *max_ffill* days, then drop any remaining NaNs.

    Parameters
    ----------
    prices    : Raw price DataFrame.
    max_ffill : Maximum number of consecutive days to forward-fill.

    Returns
    -------
    Cleaned DataFrame.
    """
    prices = prices.ffill(limit=max_ffill)
    prices = prices.dropna()
    return prices


def _print_quality_summary(prices: pd.DataFrame) -> None:
    """Print a brief data-quality summary to the console."""
    print("\n-- Data Quality Summary ------------------------------------------")
    print(f"  Date range : {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"  Trading days: {len(prices)}")
    print(f"  Tickers     : {list(prices.columns)}")
    print(f"  Missing cells after cleaning: {prices.isna().sum().sum()}")
    print("-----------------------------------------------------------------\n")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def load_prices(config: dict) -> pd.DataFrame:
    """
    Return a clean DataFrame of adjusted-close prices.

    Uses a local CSV cache so the project works offline after the first run.

    Parameters
    ----------
    config : The parsed config.yaml dictionary.

    Returns
    -------
    prices : DataFrame, dates × tickers, already cleaned and aligned.
    """
    cache_path = Path(config["data"]["cache_file"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading prices from cache: {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        prices = _download_prices(
            config["data"]["tickers"],
            config["data"]["start_date"],
        )
        prices = _align_and_fill(prices, config["data"]["max_ffill_days"])
        prices.to_csv(cache_path)
        print(f"Prices cached to {cache_path}")

    _print_quality_summary(prices)
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily percentage returns from price levels.

    Return on day t = (price_t / price_{t-1}) - 1.

    Parameters
    ----------
    prices : DataFrame of adjusted-close prices.

    Returns
    -------
    returns : DataFrame of the same shape minus the first row (which is NaN).
    """
    returns = prices.pct_change().dropna()
    return returns


def compute_losses(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Convert returns to losses: loss = -return.

    A loss is a positive number. VaR and ES are expressed as positive
    percentages of portfolio value.

    Parameters
    ----------
    returns : DataFrame of daily returns.

    Returns
    -------
    losses : DataFrame of daily losses.
    """
    return -returns


def build_portfolio_returns(
    returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Compute daily portfolio returns as the weighted sum of asset returns.

    Parameters
    ----------
    returns : DataFrame of individual-asset daily returns.
    weights : Dict mapping ticker → weight. Weights need not sum exactly to 1;
              they are used as given (config already handles normalisation).

    Returns
    -------
    portfolio_returns : Series of daily portfolio returns.
    """
    tickers = [t for t in weights if weights[t] != 0]
    w = pd.Series(weights)[tickers]
    port_ret = returns[tickers].dot(w)
    return port_ret


def get_all_series(
    returns: pd.DataFrame,
    config: dict,
) -> Dict[str, pd.Series]:
    """
    Return a dictionary of all return series: individual assets + portfolios.

    Keys are the ticker names and portfolio names from config.

    Parameters
    ----------
    returns : DataFrame of individual-asset daily returns.
    config  : Parsed config.yaml dictionary.

    Returns
    -------
    series_dict : {label: pd.Series of daily returns}
    """
    result: Dict[str, pd.Series] = {}

    # Individual assets
    for col in returns.columns:
        result[col] = returns[col]

    # Portfolios
    for port_key, port_cfg in config["portfolios"].items():
        weights = port_cfg["weights"]
        name = port_cfg["name"]
        result[name] = build_portfolio_returns(returns, weights)

    return result
