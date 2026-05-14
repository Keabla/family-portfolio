# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Fetch fresh data (run before launching app if data is stale)
.venv/Scripts/python scripts/fetch_data.py          # Windows
.venv/bin/python scripts/fetch_data.py              # Linux/Mac (Streamlit Cloud)

# Run the app locally
.venv/Scripts/streamlit run app.py

# Rebuild venv from scratch
rm -rf .venv && python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

## Architecture

**Two-script system:**
- `scripts/fetch_data.py` — data pipeline, runs independently, writes to `data/`
- `app.py` — Streamlit dashboard, reads from `data/`, never writes data files

**Data flow:**
```
policy.yaml → fetch_data.py → data/*.csv / data/*.parquet → app.py
data/holdings_top5_manual.csv ↗                           ↗
```

**policy.yaml is the single source of truth** for all candidate assets, weight ranges, constraints, and metadata. `fetch_data.py` reads it to know which tickers to pull. `app.py` reads it for the designer sliders and constraint checker.

**Secrets:** `app_password` must be set in `.streamlit/secrets.toml` (local) or the Streamlit Cloud Secrets dashboard (deployed). The app calls `require_password()` before any rendering.

**Auto-fetch on cold start:** `ensure_data_fresh()` in `app.py` runs `fetch_data.py` via `subprocess` when data is missing or older than 1 hour. It is guarded by `st.session_state["_data_checked"]` so it only fires once per browser session, not on every Streamlit rerun.

## Key files

| File | Purpose |
|------|---------|
| `policy.yaml` | Investment policy — candidates, weight ranges, constraints |
| `scripts/fetch_data.py` | Data pipeline — do not modify lightly; yfinance + ECB + iShares |
| `data/holdings_top5_manual.csv` | Hand-populated top-5 holdings for tickers without iShares API |
| `data/metrics_snapshot.csv` | Generated — gitignored, refreshed hourly |

## Deployment

Hosted on Streamlit Cloud Community (free tier, Python 3.14). Dependencies that require compiled wheels (`pandas`, `numpy`, `pyarrow`) use `>=` bounds in `requirements.txt` so pip can select Python 3.14-compatible wheels. Exact pins are only used for pure-Python packages (`streamlit`, `yfinance`, `pyyaml`, `requests`).

`ensure_data_fresh()` handles the cold-start fetch on Streamlit Cloud — no data files are committed to the repo.
