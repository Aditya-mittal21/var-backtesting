"""
report.py
=========
Writes the Excel risk report (results/risk_report.xlsx) with one sheet per
section: VaR_Summary, Backtest, TrafficLight, Margin, Stress.
Uses openpyxl with simple colour formatting — no macros.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import (
    Alignment, Font, PatternFill, Border, Side, numbers
)
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_HDR_FILL  = PatternFill("solid", fgColor="1F3864")   # dark navy
_ALT_FILL  = PatternFill("solid", fgColor="EEF2F7")   # light blue-grey
_GREEN_F   = PatternFill("solid", fgColor="C6EFCE")
_YELLOW_F  = PatternFill("solid", fgColor="FFEB9C")
_RED_F     = PatternFill("solid", fgColor="FFC7CE")
_HDR_FONT  = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
_BODY_FONT = Font(name="Calibri", size=10)
_THIN      = Side(style="thin", color="B0B8C1")
_BORDER    = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _write_df(ws, df: pd.DataFrame, title: str = "") -> None:
    """Write a DataFrame to a worksheet with header formatting."""
    if title:
        ws.append([title])
        ws[f"A{ws.max_row}"].font = Font(bold=True, size=12, name="Calibri", color="1F3864")
        ws.append([])

    rows = list(dataframe_to_rows(df.reset_index(), index=False, header=True))
    for r_idx, row in enumerate(rows, start=ws.max_row + 1):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.font = _BODY_FONT
            cell.border = _BORDER
            cell.alignment = Alignment(horizontal="center")
            if r_idx == ws.max_row - len(rows) + 1:  # header row
                cell.font = _HDR_FONT
                cell.fill = _HDR_FILL
            elif r_idx % 2 == 0:
                cell.fill = _ALT_FILL

    # Auto-fit column widths
    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 40)


def _write_df_simple(ws, df: pd.DataFrame, title: str = "") -> None:
    """Simpler writer that handles header row properly."""
    if title:
        title_cell = ws.cell(row=ws.max_row + 1, column=1, value=title)
        title_cell.font = Font(bold=True, size=12, name="Calibri", color="1F3864")
        ws.append([])

    start_row = ws.max_row + 1

    # Write header
    df_reset = df.reset_index()
    headers = list(df_reset.columns)
    for c_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=start_row, column=c_idx, value=h)
        cell.font = _HDR_FONT
        cell.fill = _HDR_FILL
        cell.border = _BORDER
        cell.alignment = Alignment(horizontal="center")

    # Write data rows
    for r_offset, (_, row) in enumerate(df_reset.iterrows(), 1):
        for c_idx, val in enumerate(row, 1):
            cell = ws.cell(row=start_row + r_offset, column=c_idx, value=val)
            cell.font = _BODY_FONT
            cell.border = _BORDER
            cell.alignment = Alignment(horizontal="center")
            if r_offset % 2 == 0:
                cell.fill = _ALT_FILL

    # Auto-fit columns
    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 42)


def _color_traffic_light_cells(ws, df: pd.DataFrame, start_row: int) -> None:
    """Colour-code the Zone column in the traffic-light sheet."""
    zone_map = {"Green": _GREEN_F, "Yellow": _YELLOW_F, "Red": _RED_F}
    df_reset = df.reset_index()
    if "Zone" not in df_reset.columns:
        return
    zone_col_idx = list(df_reset.columns).index("Zone") + 1
    for r_offset, (_, row) in enumerate(df_reset.iterrows(), 1):
        zone_val = row.get("Zone", "")
        fill = zone_map.get(str(zone_val), None)
        if fill:
            ws.cell(row=start_row + r_offset, column=zone_col_idx).fill = fill


# ---------------------------------------------------------------------------
# Main report writer
# ---------------------------------------------------------------------------

def write_report(
    var_summary: pd.DataFrame,
    diversification: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    exc_per_year: Dict[str, pd.DataFrame],
    traffic_summary: pd.DataFrame,
    red_dates_dict: Dict[str, List],
    margin_summary: pd.DataFrame,
    margin_per_year: Dict[str, pd.DataFrame],
    floor_comparison: pd.DataFrame,
    procyclicality: pd.DataFrame,
    historical_stress: pd.DataFrame,
    hypo_stress: pd.DataFrame,
    output_path: str = "results/risk_report.xlsx",
) -> None:
    """
    Write the full risk report to an Excel workbook.

    Parameters correspond to the outputs of each src/ module.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    # ── Sheet 1: VaR Summary ──────────────────────────────────────────────
    ws_var = wb.active
    ws_var.title = "VaR_Summary"
    _write_df_simple(ws_var, var_summary, "Historical VaR & Expected Shortfall (1-day, 99%)")
    ws_var.append([])
    _write_df_simple(ws_var, diversification, "Diversification Table")
    if "note" in diversification.attrs:
        note_row = ws_var.max_row + 2
        ws_var.cell(row=note_row, column=1,
                    value="Note: " + diversification.attrs["note"]).font = Font(italic=True, size=9)

    # ── Sheet 2: Backtest ─────────────────────────────────────────────────
    ws_bt = wb.create_sheet("Backtest")
    _write_df_simple(ws_bt, backtest_summary, "Exception Summary (1-day 99% VaR)")
    for label, df_yr in exc_per_year.items():
        ws_bt.append([])
        _write_df_simple(ws_bt, df_yr.to_frame(), f"Exceptions per Year — {label}")

    # ── Sheet 3: TrafficLight ─────────────────────────────────────────────
    ws_tl = wb.create_sheet("TrafficLight")
    for label, (tl_df, rd) in traffic_summary.items():
        start = ws_tl.max_row + 2
        _write_df_simple(ws_tl, tl_df, f"Basel Traffic-Light — {label}")
        _color_traffic_light_cells(ws_tl, tl_df, start)
        # Red dates
        if len(rd) > 0:
            ws_tl.append([])
            ws_tl.append([f"Red-zone dates for {label}:"])
            ws_tl.cell(row=ws_tl.max_row, column=1).font = Font(bold=True, italic=True, size=9)
            for chunk_start in range(0, len(rd), 10):
                chunk = rd[chunk_start:chunk_start+10]
                ws_tl.append([str(d.date()) for d in chunk])

    # ── Sheet 4: Margin ───────────────────────────────────────────────────
    ws_mg = wb.create_sheet("Margin")
    _write_df_simple(ws_mg, margin_summary, "10-day 99% Margin Back-Test (Base)")
    for label, df_yr in margin_per_year.items():
        ws_mg.append([])
        _write_df_simple(ws_mg, df_yr.to_frame(), f"Margin Breaches per Year — {label}")
    ws_mg.append([])
    _write_df_simple(ws_mg, floor_comparison, "Base vs Floor Margin Comparison")
    ws_mg.append([])
    _write_df_simple(ws_mg, procyclicality, "Procyclicality — Margin Around Crises")
    # Disclaimer
    disc_row = ws_mg.max_row + 2
    ws_mg.cell(row=disc_row, column=1,
               value=(
                   "DISCLAIMER: This is a simplified illustrative margin model. "
                   "It is NOT the regulatory SIMM or any official margin model."
               )).font = Font(italic=True, size=9, color="C00000")

    # ── Sheet 5: Stress ───────────────────────────────────────────────────
    ws_st = wb.create_sheet("Stress")
    _write_df_simple(ws_st, historical_stress, "Historical Stress Replays")
    ws_st.append([])
    _write_df_simple(ws_st, hypo_stress, "Hypothetical Shocks (Illustrative Assumptions)")
    note_r = ws_st.max_row + 2
    ws_st.cell(row=note_r, column=1,
               value="Note: Hypothetical shock magnitudes are illustrative assumptions, not calibrated estimates.").font = Font(italic=True, size=9)

    wb.save(output_path)
    print(f"Excel report saved to: {Path(output_path).resolve()}")
