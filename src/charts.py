"""
charts.py
=========
Generates all charts for the risk report and saves them as PNG files.

Each chart function is standalone: pass in the required data and it saves the
file to results/charts/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


CHARTS_DIR = Path("results/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Global style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linewidth": 0.6,
    "font.size": 9,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

_ZONE_COLORS = {"green": "#2ea043", "yellow": "#d29922", "red": "#f85149"}


def _save(fig: plt.Figure, name: str) -> Path:
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Returns vs VaR with exceptions marked
# ---------------------------------------------------------------------------

def plot_returns_vs_var(
    losses: pd.Series,
    var_series: pd.Series,
    exc_mask: pd.Series,
    title: str,
    filename: str,
) -> Path:
    """
    Plot daily losses, the rolling VaR line, and mark exception days.

    Parameters
    ----------
    losses, var_series : Loss and VaR series.
    exc_mask  : Boolean series, True on exception days.
    title     : Chart title.
    filename  : Output file name (no directory prefix needed).

    Returns
    -------
    Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    both = pd.DataFrame({"loss": losses, "var": var_series}).dropna()
    exc = exc_mask.reindex(both.index).fillna(False)

    ax.fill_between(both.index, 0, both["loss"] * 100,
                    color="#388bfd", alpha=0.35, label="Daily loss (%)")
    ax.plot(both.index, both["var"] * 100, color="#f0883e",
            linewidth=1.2, label="99% VaR")
    exc_idx = both.index[exc]
    ax.scatter(exc_idx, both.loc[exc_idx, "loss"] * 100,
               color="#f85149", s=18, zorder=5, label=f"Exceptions ({exc.sum()})")

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel("Loss (% of portfolio)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 2. Traffic-light zone over time
# ---------------------------------------------------------------------------

def plot_traffic_light(
    zones: pd.Series,
    title: str,
    filename: str,
) -> Path:
    """
    Plot a colour-coded strip showing the Basel traffic-light zone each day.

    Parameters
    ----------
    zones    : Series of 'green'/'yellow'/'red' strings.
    title    : Chart title.
    filename : Output file name.
    """
    clean = zones.dropna()
    colors = clean.map(_ZONE_COLORS).fillna("#8b949e")

    fig, ax = plt.subplots(figsize=(13, 2.5))
    for i, (dt, col) in enumerate(zip(clean.index, colors)):
        ax.axvspan(dt, dt + pd.Timedelta(days=1), color=col, alpha=0.8, linewidth=0)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ea043", label="Green (0-4)"),
        Patch(facecolor="#d29922", label="Yellow (5-9)"),
        Patch(facecolor="#f85149", label="Red (10+)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(clean.index[0], clean.index[-1])
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 3. Margin over time (base vs floored)
# ---------------------------------------------------------------------------

def plot_margin_over_time(
    margin_base: pd.Series,
    margin_floored: pd.Series,
    title: str,
    filename: str,
    crisis_periods: Optional[Dict[str, tuple]] = None,
) -> Path:
    """
    Plot the base margin and floored margin over time, with shaded crisis zones.

    Parameters
    ----------
    margin_base, margin_floored : Margin series (decimal fractions).
    title, filename             : Chart metadata.
    crisis_periods : Dict {label: (start, end)} for shaded crisis bands.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(margin_base.index, margin_base * 100,
            color="#388bfd", linewidth=1.2, label="Base margin")
    ax.plot(margin_floored.index, margin_floored * 100,
            color="#3fb950", linewidth=1.2, linestyle="--", label="Floored margin")

    if crisis_periods:
        for lbl, (s, e) in crisis_periods.items():
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       color="#f85149", alpha=0.12, label=lbl)

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel("10-day 99% Margin (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 4. Stress loss vs margin bar chart
# ---------------------------------------------------------------------------

def plot_stress_vs_margin(
    stress_df: pd.DataFrame,
    title: str,
    filename: str,
) -> Path:
    """
    Grouped bar chart comparing stress-scenario losses with the 10-day margin.

    Parameters
    ----------
    stress_df : Combined historical replay + hypothetical shock DataFrame.
    title, filename : Chart metadata.
    """
    # Filter to rows that have margin data
    df = stress_df.copy()

    # Use first portfolio for a clean visual (or all portfolios if small)
    portfolios = df["Portfolio"].unique()

    fig, axes = plt.subplots(
        1, len(portfolios), figsize=(7 * len(portfolios), 5), sharey=False
    )
    if len(portfolios) == 1:
        axes = [axes]

    for ax, port in zip(axes, portfolios):
        sub = df[df["Portfolio"] == port].copy()
        # Prefer the 10d margin column (historical has Pre_10d_Margin_pct, hypo has Last_10d_Margin_pct)
        if "Pre_10d_Margin_pct" in sub.columns:
            margin_col = sub["Pre_10d_Margin_pct"].combine_first(sub.get("Last_10d_Margin_pct", pd.Series(dtype=float)))
        else:
            margin_col = sub.get("Last_10d_Margin_pct", pd.Series(dtype=float))

        scenarios = sub["Scenario"].tolist()
        losses = sub["Total_Loss_pct"].tolist()
        margins = margin_col.tolist()

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax.bar(x - width / 2, losses, width, color="#f85149", alpha=0.85, label="Scenario Loss %")
        bars2 = ax.bar(x + width / 2, margins, width, color="#388bfd", alpha=0.85, label="10d Margin %")

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{port}", fontsize=10)
        ax.set_ylabel("Percentage (%)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Convenience: generate all charts
# ---------------------------------------------------------------------------

def generate_all_charts(
    all_losses: Dict[str, pd.Series],
    var_series_all: Dict[str, pd.Series],
    exc_masks_all: Dict[str, pd.Series],
    zones_all: Dict[str, pd.Series],
    margin_base_all: Dict[str, pd.Series],
    margin_floored_all: Dict[str, pd.Series],
    stress_df: pd.DataFrame,
    crisis_periods: Dict[str, tuple],
) -> Dict[str, Path]:
    """
    Generate all charts and return a dict of {name: Path}.
    """
    chart_paths: Dict[str, Path] = {}

    # Returns vs VaR for each series
    for label in all_losses:
        safe = label.replace("/", "-").replace("=", "").replace(" ", "_")
        p = plot_returns_vs_var(
            losses=all_losses[label],
            var_series=var_series_all[label],
            exc_mask=exc_masks_all[label],
            title=f"Daily Loss vs 99% VaR — {label}",
            filename=f"returns_vs_var_{safe}.png",
        )
        chart_paths[f"returns_vs_var_{label}"] = p

    # Traffic-light zones
    for label, zones in zones_all.items():
        safe = label.replace("/", "-").replace("=", "").replace(" ", "_")
        p = plot_traffic_light(
            zones=zones,
            title=f"Basel Traffic-Light Zone — {label}",
            filename=f"traffic_light_{safe}.png",
        )
        chart_paths[f"traffic_light_{label}"] = p

    # Margin over time (one chart per portfolio)
    for label in margin_base_all:
        safe = label.replace("/", "-").replace("=", "").replace(" ", "_")
        p = plot_margin_over_time(
            margin_base=margin_base_all[label],
            margin_floored=margin_floored_all[label],
            title=f"10-day 99% Margin — {label}",
            filename=f"margin_{safe}.png",
            crisis_periods=crisis_periods,
        )
        chart_paths[f"margin_{label}"] = p

    # Stress vs margin
    p = plot_stress_vs_margin(
        stress_df=stress_df,
        title="Stress Scenario Loss vs 10-day Margin",
        filename="stress_vs_margin.png",
    )
    chart_paths["stress_vs_margin"] = p

    print(f"Charts saved to: {CHARTS_DIR.resolve()}")
    return chart_paths
