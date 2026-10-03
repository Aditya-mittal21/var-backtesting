"""
run_pipeline.py
===============
Master entry point.  Run this from the project root:

    python run_pipeline.py

It orchestrates the full pipeline in order:
  1. Load / download data
  2. Compute VaR and ES
  3. Back-test VaR
  4. Compute and back-test margin
  5. Run stress tests
  6. Write Excel report and charts
  7. Print end-of-run sanity checks
"""

from __future__ import annotations

import sys
import io

# Force UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

# Make sure 'src' is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.data_loader import (
    load_config, load_prices, compute_returns, compute_losses,
    build_portfolio_returns, get_all_series,
)
from src.var_models import (
    rolling_var, rolling_es, rolling_var_nday,
    compute_var_es_summary, diversification_table,
)
from src.backtest import (
    find_exceptions, exception_summary, exceptions_per_year,
    rolling_traffic_light, traffic_light_summary, backtest_cluster_note,
)
from src.margin import (
    compute_margin, apply_floor,
    margin_backtest_summary, margin_breaches_per_year, floor_comparison_table, procyclicality_events,
)
from src.stress import run_historical_replays, run_hypothetical_shocks
from src.charts import generate_all_charts
from src.report import write_report


def main() -> None:
    print("=" * 65)
    print("  Market Risk & Margin Back-Testing Framework")
    print("=" * 65)

    # ── 0. Config ──────────────────────────────────────────────────────────
    cfg = load_config("config.yaml")
    WINDOW     = cfg["risk"]["window"]
    CONFIDENCE = cfg["risk"]["confidence"]
    MARGIN_D   = cfg["risk"]["margin_days"]
    FLOOR_LB   = cfg["risk"]["floor_lookback"]
    FLOOR_FR   = cfg["risk"]["floor_fraction"]
    GREEN_MAX  = cfg["traffic_light"]["green_max"]
    YELLOW_MAX = cfg["traffic_light"]["yellow_max"]

    # ── 1. Data ────────────────────────────────────────────────────────────
    prices  = load_prices(cfg)
    returns = compute_returns(prices)
    losses  = compute_losses(returns)

    # Extend prices to include portfolio columns
    for port_key, port_cfg in cfg["portfolios"].items():
        weights = port_cfg["weights"]
        name = port_cfg["name"]
        returns[name] = build_portfolio_returns(returns, weights)
        losses[name]  = -returns[name]

    # All series dict: individual tickers + portfolio names
    all_returns: dict[str, pd.Series] = {}
    all_losses_d: dict[str, pd.Series] = {}
    tickers = cfg["data"]["tickers"]
    port_names = [p["name"] for p in cfg["portfolios"].values()]

    for col in tickers + port_names:
        if col in returns.columns:
            all_returns[col] = returns[col]
            all_losses_d[col] = losses[col]

    # ── 2. VaR & ES ────────────────────────────────────────────────────────
    print("\n[Step 2] Computing rolling VaR and ES...")
    var_series_all: dict[str, pd.Series] = {}
    es_series_all:  dict[str, pd.Series] = {}
    for label, ls in all_losses_d.items():
        var_series_all[label] = rolling_var(ls, WINDOW, CONFIDENCE)
        es_series_all[label]  = rolling_es(ls, WINDOW, CONFIDENCE)

    var_summary    = compute_var_es_summary(all_losses_d, WINDOW, CONFIDENCE)
    asset_losses_d = {k: v for k, v in all_losses_d.items() if k in tickers}
    port_losses_d  = {k: v for k, v in all_losses_d.items() if k in port_names}
    diversif_tbl   = diversification_table(asset_losses_d, port_losses_d, WINDOW, CONFIDENCE)

    print(var_summary.to_string())

    # ── 3. Back-test ───────────────────────────────────────────────────────
    print("\n[Step 3] Back-testing VaR exceptions...")
    exc_masks_all: dict[str, pd.Series] = {}
    exc_per_year_all: dict[str, pd.Series] = {}

    for label, ls in all_losses_d.items():
        exc_masks_all[label] = find_exceptions(ls, var_series_all[label])
        exc_per_year_all[label] = exceptions_per_year(ls, var_series_all[label])

    bt_summary = exception_summary(all_losses_d, WINDOW, CONFIDENCE)
    print(bt_summary.to_string())

    # Cluster note (using the equal-weight portfolio)
    eq_name = cfg["portfolios"]["equal_weight"]["name"]
    all_exc_dates = pd.DatetimeIndex([])
    for label in port_names:
        all_exc_dates = all_exc_dates.append(
            exc_masks_all[label][exc_masks_all[label]].index
        )
    cluster_note = backtest_cluster_note(all_exc_dates.sort_values())
    print(f"\nCluster note:\n  {cluster_note}\n")

    # ── 4. Traffic light ───────────────────────────────────────────────────
    print("[Step 4] Computing Basel traffic-light zones...")
    zones_all: dict[str, pd.Series] = {}
    traffic_results: dict[str, tuple] = {}

    for label, ls in all_losses_d.items():
        z = rolling_traffic_light(ls, var_series_all[label], WINDOW, GREEN_MAX, YELLOW_MAX)
        zones_all[label] = z
        tl_sum, red_dates = traffic_light_summary(z)
        traffic_results[label] = (tl_sum, red_dates)

    # ── 5. Margin ──────────────────────────────────────────────────────────
    print("[Step 5] Computing 10-day margin...")
    margin_base_all:    dict[str, pd.Series] = {}
    margin_floored_all: dict[str, pd.Series] = {}
    margin_per_year_all: dict[str, pd.Series] = {}

    for label, ls in all_losses_d.items():
        mb = compute_margin(ls, MARGIN_D, WINDOW, CONFIDENCE)
        mf = apply_floor(mb, FLOOR_LB, FLOOR_FR)
        margin_base_all[label]    = mb
        margin_floored_all[label] = mf
        margin_per_year_all[label] = margin_breaches_per_year(ls, mb, MARGIN_D)

    margin_summary_df  = margin_backtest_summary(all_losses_d, MARGIN_D, WINDOW, CONFIDENCE)
    floor_cmp_df       = floor_comparison_table(all_losses_d, MARGIN_D, WINDOW, CONFIDENCE, FLOOR_LB, FLOOR_FR)

    print(margin_summary_df.to_string())

    # Procyclicality events (portfolios only for brevity in report)
    procycl_frames = []
    for label in port_names:
        df_pc = procyclicality_events(label, all_losses_d[label], MARGIN_D, WINDOW, CONFIDENCE, FLOOR_LB, FLOOR_FR)
        procycl_frames.append(df_pc)
    procycl_df = pd.concat(procycl_frames, ignore_index=True) if procycl_frames else pd.DataFrame()

    # ── 6. Stress testing ──────────────────────────────────────────────────
    print("[Step 6] Running stress tests...")
    hist_stress = run_historical_replays(
        returns=returns[tickers],
        portfolios_cfg=cfg["portfolios"],
        stress_periods=cfg["stress_periods"],
        all_losses=all_losses_d,
        window=WINDOW,
        confidence=CONFIDENCE,
        margin_days=MARGIN_D,
    )
    hypo_stress = run_hypothetical_shocks(
        portfolios_cfg=cfg["portfolios"],
        hypothetical_shocks=cfg["hypothetical_shocks"],
        all_losses=all_losses_d,
        window=WINDOW,
        confidence=CONFIDENCE,
        margin_days=MARGIN_D,
    )
    print("\nHistorical stress results:")
    print(hist_stress.to_string(index=False))
    print("\nHypothetical stress results:")
    print(hypo_stress.to_string(index=False))

    # ── 7. Charts ──────────────────────────────────────────────────────────
    print("\n[Step 7] Generating charts...")
    crisis_bands = {
        "GFC 2008": ("2008-09-02", "2008-11-20"),
        "COVID 2020": ("2020-02-19", "2020-03-23"),
        "Rate Shock 2022": ("2022-01-03", "2022-10-14"),
    }
    # Chart for portfolios only (too many assets otherwise)
    port_losses_chart  = {k: v for k, v in all_losses_d.items() if k in port_names}
    port_var_chart     = {k: v for k, v in var_series_all.items() if k in port_names}
    port_exc_chart     = {k: v for k, v in exc_masks_all.items() if k in port_names}
    port_zones_chart   = {k: v for k, v in zones_all.items() if k in port_names}
    port_mbase_chart   = {k: v for k, v in margin_base_all.items() if k in port_names}
    port_mfloor_chart  = {k: v for k, v in margin_floored_all.items() if k in port_names}

    # Also include individual assets for returns-vs-var
    all_losses_chart  = all_losses_d
    all_var_chart     = var_series_all
    all_exc_chart     = exc_masks_all
    all_zones_chart   = zones_all

    combined_stress = pd.concat([hist_stress, hypo_stress], ignore_index=True)
    # Unify margin column name for chart
    combined_stress["Margin_pct"] = combined_stress.get(
        "Pre_10d_Margin_pct", combined_stress.get("Last_10d_Margin_pct")
    )

    generate_all_charts(
        all_losses=all_losses_chart,
        var_series_all=all_var_chart,
        exc_masks_all=all_exc_chart,
        zones_all=all_zones_chart,
        margin_base_all=port_mbase_chart,
        margin_floored_all=port_mfloor_chart,
        stress_df=combined_stress,
        crisis_periods=crisis_bands,
    )

    # ── 8. Excel report ────────────────────────────────────────────────────
    print("[Step 8] Writing Excel report...")
    exc_per_year_report = {
        label: exc_per_year_all[label]
        for label in port_names
        if label in exc_per_year_all
    }
    margin_per_year_report = {
        label: margin_per_year_all[label]
        for label in port_names
        if label in margin_per_year_all
    }
    write_report(
        var_summary=var_summary,
        diversification=diversif_tbl,
        backtest_summary=bt_summary,
        exc_per_year=exc_per_year_report,
        traffic_summary=traffic_results,
        red_dates_dict={k: v[1] for k, v in traffic_results.items()},
        margin_summary=margin_summary_df,
        margin_per_year=margin_per_year_report,
        floor_comparison=floor_cmp_df,
        procyclicality=procycl_df,
        historical_stress=hist_stress,
        hypo_stress=hypo_stress,
    )

    # ── 9. End-of-run sanity checks ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  END-OF-RUN SANITY CHECKS")
    print("=" * 65)

    all_pass = True
    for label, row in bt_summary.iterrows():
        rate = row["Exception_Rate_pct"]
        ok = 0.5 <= rate <= 2.0
        status = "PASS" if ok else "WARN"
        if not ok:
            all_pass = False
        print(f"  [{status}] {label:30s}  exception rate = {rate:.3f}%  (target 0.5-2.0%)")

    print()
    gfc_any = False
    covid_any = False
    for label in port_names:
        exc_idx = exc_masks_all[label][exc_masks_all[label]].index
        gfc_any  |= bool(((exc_idx >= "2008-01-01") & (exc_idx <= "2009-06-30")).sum() > 0)
        covid_any |= bool(((exc_idx >= "2020-02-01") & (exc_idx <= "2020-05-31")).sum() > 0)

    print(f"  [{'PASS' if gfc_any else 'WARN'}] Exceptions cluster in 2008      : {gfc_any}")
    print(f"  [{'PASS' if covid_any else 'WARN'}] Exceptions cluster in Mar 2020 : {covid_any}")

    print()
    for label, row in margin_summary_df.iterrows():
        rate = row["Breach_Rate_pct"]
        ok = 0.3 <= rate <= 3.0
        status = "PASS" if ok else "WARN"
        print(f"  [{status}] Margin breach rate  {label:25s}: {rate:.3f}%  (target ~1%)")

    # Floor vs base
    for label in floor_cmp_df.index:
        base_r = floor_cmp_df.loc[label, "Base_Breach_Rate_pct"]
        floor_r = floor_cmp_df.loc[label, "Floor_Breach_Rate_pct"]
        base_m  = floor_cmp_df.loc[label, "Base_Avg_Margin_pct"]
        floor_m = floor_cmp_df.loc[label, "Floor_Avg_Margin_pct"]
        ok = (floor_r <= base_r) and (floor_m >= base_m)
        status = "PASS" if ok else "WARN"
        print(f"  [{status}] Floor lowers breach & raises avg margin — {label}: breach {base_r:.2f}%→{floor_r:.2f}%  avg margin {base_m:.2f}%→{floor_m:.2f}%")

    print()
    print("  Done. Results in results/risk_report.xlsx and results/charts/")
    print("=" * 65)


if __name__ == "__main__":
    main()
