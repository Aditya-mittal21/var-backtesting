# Market Risk & Margin Back-Testing Framework

A clean, readable Python project that estimates and back-tests historical Value-at-Risk (VaR), Expected Shortfall, and a VaR-based initial margin for a multi-asset portfolio. Written for a second-year student who needs to explain every concept in a plain-English interview.

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data on the first run, then uses cache)
python run_pipeline.py

# 3. Run tests
pytest tests/ -v

# 4. Open the results notebook
jupyter notebook notebooks/results_walkthrough.ipynb
```

Outputs are written to:
- `results/risk_report.xlsx` — five-sheet Excel workbook
- `results/charts/` — PNG charts

---

## Project structure

```
var backtesting/
├── config.yaml              ← All parameters live here
├── requirements.txt
├── run_pipeline.py          ← Single entry point
├── src/
│   ├── data_loader.py       ← Download, cache, clean prices
│   ├── var_models.py        ← Historical VaR and ES
│   ├── backtest.py          ← Exception counting, traffic light
│   ├── margin.py            ← 10-day margin, floor variant
│   ├── stress.py            ← Historical replays + hypothetical shocks
│   ├── charts.py            ← All chart generation
│   └── report.py            ← Excel report writer
├── tests/
│   └── test_var_models.py   ← Pytest tests
├── notebooks/
│   └── results_walkthrough.ipynb
├── data/
│   └── prices.csv           ← Auto-generated cache
├── results/
│   ├── risk_report.xlsx
│   └── charts/
└── docs/
    └── INTERVIEW_NOTES.md
```

---

## Method summary (plain English)

### Data
We download daily adjusted-close prices for six instruments — SPY (US equities), TLT (long-term Treasuries), LQD (investment-grade bonds), GLD (gold), EURUSD, and USDINR — from 2007 to today. Prices are cached locally so the project works offline. We compute simple daily percentage returns: `return = (today's price / yesterday's price) − 1`. A **loss** is the negative of a return; VaR and ES are always expressed as positive numbers.

### Historical VaR
Sort the past 250 daily losses from smallest to largest. The 99th percentile is the VaR: on 99 out of 100 days, the loss was smaller than this number. No distribution is assumed — we read directly from history. The VaR for day *t* uses only data up to day *t − 1* (no look-ahead).

### Expected Shortfall (ES)
Average of the losses in the worst 1% of the 250-day window. Always at least as large as VaR. It answers "given that we breached VaR, how bad was it on average?"

### Back-testing
We count "exceptions" — days when the realised loss exceeded the VaR forecast. At 99% confidence, we expect about 1% of days to be exceptions. We apply the Basel traffic-light system: 0–4 exceptions in any 250-day window → green; 5–9 → yellow; 10+ → red.

### Initial margin
The 10-day 99% VaR, calculated on overlapping 10-day losses from the past 250 days. We back-test it by checking how often the actual forward 10-day loss exceeded the margin. The **floor variant** prevents margin from falling below 50% of its three-year peak, reducing procyclicality.

### Stress testing
We replay the actual returns from three crisis windows (GFC 2008, COVID 2020, 2022 rate shock) against today's portfolio weights. We also apply fixed hypothetical shocks (e.g., SPY −25%) defined in `config.yaml`. For each scenario we report the total loss, how many times it exceeds the 1-day VaR, and whether the 10-day margin would have covered it.

---

## Key results (typical run)

| Metric | Typical value |
|--------|---------------|
| Equal-weight 1-day 99% VaR | ~0.8–1.2% |
| Exception rate | ~0.8–1.5% (target 1%) |
| Exceptions in 2008 GFC | Majority of red-zone days |
| 10-day margin breach rate | ~0.8–1.5% |
| Floor variant: breach rate | Slightly lower than base |
| Floor variant: avg margin | Slightly higher than base |
| GFC replay loss (equal-weight) | ~30–45% |
| COVID replay loss (equal-weight) | ~20–35% |

---

## Honest limitations

1. **ETF and FX proxies.** SPY, TLT, LQD, GLD are ETFs — not the raw asset classes. Actual returns may differ due to expense ratios, tracking error, and liquidity.
2. **Equal weights are illustrative.** Real portfolios are not equal-weighted and weights change over time.
3. **No transaction costs** or bid-ask spreads are modelled. Real trading would reduce returns.
4. **VaR-based margin is NOT SIMM.** The regulatory Standard Initial Margin Model (SIMM) is far more complex. This project is a simplified educational illustration only.
5. **Past data may not predict the future.** Historical VaR implicitly assumes the next bad day will look like a past bad day. It can completely miss new types of crises.
6. **Short memory.** A 250-day window quickly forgets events older than a year, so the model is slow to react when a new crisis begins.
7. **No correlation between days.** Historical simulation assumes returns are independent across time. In practice, volatility clusters (bad days tend to follow bad days).
