"""
stress.py
=========
Runs two types of stress tests on the portfolios:

1. Historical replays: apply the actual market returns from three crisis
   periods to the *current* portfolio weights and compute the total loss.
2. Hypothetical shocks: apply fixed percentage moves to each asset as defined
   in config.yaml, then compute the portfolio loss.

Plain-English idea
------------------
'What would happen to my portfolio today if history repeated itself?' or
'What if equities fell 25% overnight?'  We apply those moves to the current
weights and report the loss as a percentage of portfolio value, as a multiple
of the recent 1-day VaR, and compared to the 10-day margin held just before
the scenario.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.var_models import rolling_var
from src.margin import compute_margin


# ---------------------------------------------------------------------------
# Historical replays
# ---------------------------------------------------------------------------

def historical_replay(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    start: str,
    end: str,
    label: str,
) -> Tuple[float, pd.Series]:
    """
    Apply the actual returns in [start, end] to the given weights.

    Parameters
    ----------
    returns : DataFrame of individual-asset daily returns.
    weights : Dict of {ticker: weight}.
    start, end : ISO date strings for the scenario window.
    label  : Name of the scenario (for display only).

    Returns
    -------
    total_loss   : Cumulative portfolio loss over the period (positive = loss).
    daily_losses : Series of daily portfolio losses during the scenario.
    """
    tickers = [t for t in weights if weights[t] != 0]
    w = pd.Series(weights)[tickers]
    scenario_ret = returns.loc[start:end, tickers]
    daily_port_ret = scenario_ret.dot(w)
    # Cumulative return: (1+r1)*(1+r2)*... - 1
    cum_ret = (1 + daily_port_ret).prod() - 1
    total_loss = -cum_ret  # positive means loss
    daily_losses = -daily_port_ret
    return float(total_loss), daily_losses


def run_historical_replays(
    returns: pd.DataFrame,
    portfolios_cfg: dict,
    stress_periods: dict,
    all_losses: Dict[str, pd.Series],
    window: int = 250,
    confidence: float = 0.99,
    margin_days: int = 10,
) -> pd.DataFrame:
    """
    Run all historical replay scenarios for all portfolios.

    For each scenario and portfolio, computes:
    - Total portfolio loss over the scenario window.
    - Loss as a multiple of the most recent 1-day VaR (using data up to the
      day before the scenario starts).
    - Whether the 10-day margin set just before the scenario covers the loss.

    Parameters
    ----------
    returns         : Full historical returns DataFrame.
    portfolios_cfg  : 'portfolios' section of config.yaml.
    stress_periods  : 'stress_periods' section of config.yaml.
    all_losses      : {label: loss Series} for all series.
    window, confidence, margin_days : Risk parameters.

    Returns
    -------
    results : DataFrame, one row per (portfolio, scenario).
    """
    rows = []
    for port_key, port_cfg in portfolios_cfg.items():
        port_name = port_cfg["name"]
        weights = port_cfg["weights"]
        port_losses = all_losses[port_name]
        var_series = rolling_var(port_losses, window, confidence)
        margin_series = compute_margin(port_losses, margin_days, window, confidence)

        for scenario_key, sc in stress_periods.items():
            sc_start = sc["start"]
            sc_end = sc["end"]
            sc_label = sc["label"]

            total_loss, _ = historical_replay(returns, weights, sc_start, sc_end, sc_label)

            # VaR and margin just before the scenario starts
            pre_var = var_series.loc[:sc_start].dropna()
            last_var = float(pre_var.iloc[-1]) if not pre_var.empty else np.nan

            pre_margin = margin_series.loc[:sc_start].dropna()
            last_margin = float(pre_margin.iloc[-1]) if not pre_margin.empty else np.nan

            var_multiple = total_loss / last_var if last_var and last_var > 0 else np.nan
            margin_cover = "Yes" if (not np.isnan(last_margin) and last_margin >= total_loss) else "No"

            rows.append({
                "Portfolio": port_name,
                "Scenario": sc_label,
                "Total_Loss_pct": round(total_loss * 100, 3),
                "Pre_VaR_1d_pct": round(last_var * 100, 3) if not np.isnan(last_var) else np.nan,
                "Loss_as_VaR_Multiple": round(var_multiple, 2) if not np.isnan(var_multiple) else np.nan,
                "Pre_10d_Margin_pct": round(last_margin * 100, 3) if not np.isnan(last_margin) else np.nan,
                "Margin_Covers_Loss": margin_cover,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hypothetical shocks
# ---------------------------------------------------------------------------

def apply_hypothetical_shock(
    weights: Dict[str, float],
    shocks: Dict[str, float],
) -> float:
    """
    Compute the portfolio loss from a fixed-percentage shock to each asset.

    Parameters
    ----------
    weights : Dict of {ticker: weight}.
    shocks  : Dict of {ticker: percentage_move}, positive = price up.

    Returns
    -------
    portfolio_loss : Positive means a loss for the portfolio.
    """
    loss = 0.0
    for ticker, weight in weights.items():
        shock = shocks.get(ticker, 0.0)
        # A positive shock (price up) reduces loss, negative increases it
        loss += -weight * shock
    return float(loss)


def run_hypothetical_shocks(
    portfolios_cfg: dict,
    hypothetical_shocks: dict,
    all_losses: Dict[str, pd.Series],
    window: int = 250,
    confidence: float = 0.99,
    margin_days: int = 10,
) -> pd.DataFrame:
    """
    Run all hypothetical shock scenarios for all portfolios.

    Parameters
    ----------
    portfolios_cfg       : 'portfolios' section of config.yaml.
    hypothetical_shocks  : 'hypothetical_shocks' section of config.yaml.
    all_losses           : {label: loss Series}.
    window, confidence, margin_days : Risk parameters.

    Returns
    -------
    results : DataFrame, one row per (portfolio, shock scenario).
    """
    rows = []
    for port_key, port_cfg in portfolios_cfg.items():
        port_name = port_cfg["name"]
        weights = port_cfg["weights"]
        port_losses = all_losses[port_name]
        var_series = rolling_var(port_losses, window, confidence)
        margin_series = compute_margin(port_losses, margin_days, window, confidence)

        last_var = float(var_series.dropna().iloc[-1]) if not var_series.dropna().empty else np.nan
        last_margin = float(margin_series.dropna().iloc[-1]) if not margin_series.dropna().empty else np.nan

        for shock_key, sc in hypothetical_shocks.items():
            sc_label = sc["label"]
            shocks = sc["shocks"]
            total_loss = apply_hypothetical_shock(weights, shocks)
            var_multiple = total_loss / last_var if last_var and last_var > 0 else np.nan
            margin_cover = "Yes" if (not np.isnan(last_margin) and last_margin >= total_loss) else "No"

            rows.append({
                "Portfolio": port_name,
                "Scenario": sc_label,
                "Note": "Illustrative assumption — not based on observed history",
                "Total_Loss_pct": round(total_loss * 100, 3),
                "Last_VaR_1d_pct": round(last_var * 100, 3) if not np.isnan(last_var) else np.nan,
                "Loss_as_VaR_Multiple": round(var_multiple, 2) if not np.isnan(var_multiple) else np.nan,
                "Last_10d_Margin_pct": round(last_margin * 100, 3) if not np.isnan(last_margin) else np.nan,
                "Margin_Covers_Loss": margin_cover,
            })

    return pd.DataFrame(rows)
