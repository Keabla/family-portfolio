"""
fetch_data.py — Pull all data needed by the allocation designer.

Outputs (under data/):
  - prices_history.parquet : daily OHLCV for each candidate (2y lookback)
  - metrics_snapshot.csv   : current metrics per candidate (price, MAs, drawdown, P/E, yield, TER, vol, AUM)
  - holdings_top5.csv      : merged top-5 holdings (auto from iShares + manual)
  - macro_snapshot.csv     : key macro indicators (EUR HICP, ECB rate, EUR/USD)

Run: `python scripts/fetch_data.py`
"""
from __future__ import annotations
import re, sys, json, time, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import yfinance as yf
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
POLICY = ROOT / "policy.yaml"

LOOKBACK_DAYS = 730  # 2y
USER_AGENT = "Mozilla/5.0 (compatible; family-portfolio/1.0)"

# ---------- Helpers ----------

def load_policy() -> dict:
    with open(POLICY, "r") as f:
        return yaml.safe_load(f)

def safe_yf_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch with retry and period fallback; return empty DataFrame on failure."""
    # Some thinly-traded MI tickers have gaps that make long-period fetches fail;
    # progressively shorten the window before giving up.
    fallback_periods = ["1y", "6mo"] if period == "2y" else []
    for p in [period] + fallback_periods:
        for attempt in (1, 2):
            try:
                df = yf.Ticker(ticker).history(period=p, auto_adjust=False)
                if not df.empty:
                    if p != period:
                        print(f"  {ticker}: used period={p} (fallback)", file=sys.stderr)
                    return df
            except Exception as e:
                print(f"  yfinance error for {ticker} (attempt {attempt}, period={p}): {e}",
                      file=sys.stderr)
                time.sleep(2)
    return pd.DataFrame()

_fx_cache: dict[str, pd.Series] = {}

def to_eur(prices: pd.Series, from_ccy: str) -> pd.Series:
    """
    Convert a price series from `from_ccy` to EUR using daily yfinance FX data.
    Dates that have no FX observation are forward-filled (weekends/holidays).
    If conversion fails the original series is returned unchanged.
    """
    if from_ccy == "EUR":
        return prices
    fx_ticker = f"{from_ccy}EUR=X"
    if fx_ticker not in _fx_cache:
        fx_hist = safe_yf_history(fx_ticker, period="2y")
        if fx_hist.empty:
            print(f"  FX fetch failed for {fx_ticker}; prices kept as-is", file=sys.stderr)
            _fx_cache[fx_ticker] = pd.Series(dtype=float)
        else:
            _fx_cache[fx_ticker] = fx_hist["Close"].dropna()
    fx = _fx_cache[fx_ticker]
    if fx.empty:
        return prices
    # Align FX to price index, forward-fill gaps, then multiply
    fx_aligned = fx.reindex(prices.index, method="ffill")
    return (prices * fx_aligned).dropna()


def compute_technicals(prices: pd.Series) -> dict:
    """Latest price + MA / drawdown / volatility metrics."""
    if prices.empty:
        return {k: np.nan for k in
                ("last_price","ma50","ma200","pct_vs_ma50","pct_vs_ma200",
                 "high_52w","drawdown_from_52w_high","vol_20d_ann","vol_252d_ann")}
    last = float(prices.iloc[-1])
    ma50 = float(prices.tail(50).mean()) if len(prices) >= 50 else np.nan
    ma200 = float(prices.tail(200).mean()) if len(prices) >= 200 else np.nan
    high_52w = float(prices.tail(252).max()) if len(prices) >= 1 else np.nan
    rets = prices.pct_change().dropna()
    return {
        "last_price": last,
        "ma50": ma50,
        "ma200": ma200,
        "pct_vs_ma50":   (last/ma50 - 1)*100  if ma50  and not np.isnan(ma50)  else np.nan,
        "pct_vs_ma200":  (last/ma200 - 1)*100 if ma200 and not np.isnan(ma200) else np.nan,
        "high_52w": high_52w,
        "drawdown_from_52w_high": (last/high_52w - 1)*100 if high_52w else np.nan,
        "vol_20d_ann":  float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets) >= 20 else np.nan,
        "vol_252d_ann": float(rets.tail(252).std() * np.sqrt(252) * 100) if len(rets) >= 60 else np.nan,
    }

# ---------- iShares JSON ----------

ISHARES_URL = (
    "https://www.ishares.com/uk/individual/en/products/{pid}/fund/1506575576011.ajax"
    "?fileType=json&dataType=fund"
)

def fetch_ishares_holdings(product_id: str, candidate_id: str) -> pd.DataFrame:
    """Fetch top-5 holdings from iShares public JSON. Returns DataFrame or empty."""
    cache_file = CACHE / f"ishares_{product_id}.json"
    payload = None
    # Try cache first if < 24h old
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
        try:
            payload = json.loads(cache_file.read_text())
        except Exception:
            payload = None
    if payload is None:
        try:
            r = requests.get(ISHARES_URL.format(pid=product_id),
                             headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            payload = r.json()
            cache_file.write_text(json.dumps(payload))
        except Exception as e:
            print(f"  iShares fetch failed for {candidate_id} ({product_id}): {e}", file=sys.stderr)
            return pd.DataFrame()

    # iShares JSON shape varies; try common locations.
    holdings = None
    for key in ("holdings", "topHoldings", "fundHoldings"):
        if key in payload:
            holdings = payload[key]
            break
    if holdings is None and isinstance(payload.get("aaData"), list):
        holdings = payload["aaData"]
    if not holdings:
        print(f"  iShares: no holdings field found for {candidate_id}", file=sys.stderr)
        return pd.DataFrame()

    rows = []
    today = dt.date.today().isoformat()
    for i, h in enumerate(holdings[:5]):
        if isinstance(h, dict):
            name = h.get("name") or h.get("issuerName") or h.get("Issuer Name") or ""
            wt = h.get("weight") or h.get("weighting") or h.get("Weight (%)") or ""
            sector = h.get("sectorName") or h.get("sector") or ""
            country = h.get("countryName") or h.get("country") or ""
        elif isinstance(h, list):
            # Older iShares format: list of cells
            name = h[0] if len(h) > 0 else ""
            wt = h[2] if len(h) > 2 else ""
            sector = h[5] if len(h) > 5 else ""
            country = h[7] if len(h) > 7 else ""
        else:
            continue
        try:
            wt_num = float(str(wt).replace(",", ".").replace("%", "").strip()) if wt else np.nan
        except ValueError:
            wt_num = np.nan
        rows.append({
            "candidate_id": candidate_id,
            "rank": i + 1,
            "security_name": str(name).strip(),
            "weight_pct": wt_num,
            "sector": str(sector).strip(),
            "country": str(country).strip(),
            "as_of_date": today,
        })
    return pd.DataFrame(rows)

# ---------- Italian deposit rate (ECB SDW) ----------

# Italian HICP (all items, annual rate of change) — ECB ICP dataset.
# Used to convert BTP Italia real yield to a comparable nominal yield.
_ECB_IT_HICP_URL = (
    "https://data-api.ecb.europa.eu/service/data/ICP"
    "/M.IT.N.000000.4.ANR"
    "?lastNObservations=3&format=jsondata"
)

def fetch_italian_hicp() -> float:
    """Return the latest Italian HICP annual rate (%) from ECB SDW, or 2.0 as fallback."""
    try:
        val = _ecb_sdw_latest(_ECB_IT_HICP_URL)
        if val is not None:
            return val
    except Exception as e:
        print(f"  ECB HICP fetch failed: {e}", file=sys.stderr)
    return 2.0  # fallback: ECB medium-term inflation target


# MIR M.IT.B.L21.A.2240.R.EUR.N.A = Italian MFI new-business overnight deposits to
# households (EUR), annualized agreed rate — the retail demand-deposit rate,
# typically 0.3–1%, which is what Italian bank clients actually receive.
# Fallback: ECB deposit facility rate (FM dataset) minus a typical bank spread.
_ECB_MIR_URL = (
    "https://data-api.ecb.europa.eu/service/data/MIR"
    "/M.IT.B.L21.A.2240.R.EUR.N.A"
    "?lastNObservations=6&format=jsondata"
)
_ECB_DFR_URL = (
    "https://data-api.ecb.europa.eu/service/data/FM"
    "/D.U2.EUR.4F.KR.DFR.LEV"   # daily ECB deposit facility rate
    "?lastNObservations=5&format=jsondata"
)

def _ecb_sdw_latest(url: str) -> float | None:
    """Fetch the latest non-null observation from an ECB SDW jsondata series."""
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    data = r.json()
    series = data["dataSets"][0]["series"]
    obs = list(series.values())[0]["observations"]
    for idx in sorted(obs.keys(), key=lambda x: int(x), reverse=True):
        val = obs[idx][0]
        if val is not None:
            return round(float(val), 2)
    return None

def fetch_ecb_it_deposit_rate() -> float:
    """
    Return the latest Italian household overnight deposit rate (%) from ECB SDW.
    Falls back to ECB deposit facility rate minus 1.5 pp (typical bank spread),
    then to 0.5 as a last resort.
    """
    # Primary: Italian MFI overnight deposit rate (what retail clients earn)
    try:
        val = _ecb_sdw_latest(_ECB_MIR_URL)
        if val is not None:
            return val
    except Exception as e:
        print(f"  ECB MIR fetch failed: {e}", file=sys.stderr)
    # Fallback: ECB deposit facility rate minus typical retail spread
    try:
        dfr = _ecb_sdw_latest(_ECB_DFR_URL)
        if dfr is not None:
            return round(max(dfr - 1.5, 0.1), 2)
    except Exception as e:
        print(f"  ECB DFR fallback failed: {e}", file=sys.stderr)
    return 0.5


# ---------- Borsa Italiana bond price ----------

_BI_CANDIDATE_URLS = [
    # Euronext JSON API (MOT market MIC = XMOT)
    ("json", "https://live.euronext.com/en/ajax/getDetailedQuote/{isin}-XMOT"),
    # Borsa Italiana HTML — BTP Italia section (most common for IT0005497000)
    ("html", "https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp-italia/dati/{isin}.html"),
    # Borsa Italiana HTML — BTP inflation-linked section (alternate slug)
    ("html", "https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp-inflation-linked/dati/{isin}.html"),
    # Borsa Italiana HTML — generic bond page (no category slug)
    ("html", "https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/dati/{isin}.html"),
]

_PRICE_PATTERNS = (
    r'"lastPrice"\s*:\s*"?([\d]+[,.][\d]+)"?',
    r'class=["\']l-utilimp["\'][^>]*>\s*([\d,]+\.[\d]+|[\d]+,[\d]+)',
    r'Ultimo\s+prezzo[^<]*<[^>]+>\s*([\d,]+\.[\d]+|[\d]+,[\d]+)',
    r'"price"\s*:\s*"?([\d]+[,.][\d]+)"?',
)

def fetch_borsaitaliana_bond_price(isin: str) -> float | None:
    """
    Fetch the current price for an Italian MOT bond.
    Tries the Euronext JSON API then several Borsa Italiana HTML paths.
    """
    for mode, url_tpl in _BI_CANDIDATE_URLS:
        url = url_tpl.format(isin=isin)
        try:
            headers = {"User-Agent": USER_AGENT}
            if mode == "json":
                headers["X-Requested-With"] = "XMLHttpRequest"
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            if mode == "json":
                data = r.json()
                for field in ("lastPrice", "last", "price", "currentPrice", "referencePrice"):
                    val = data.get(field)
                    if val is not None:
                        return float(str(val).replace(",", "."))
            else:
                html = r.text
                for pattern in _PRICE_PATTERNS:
                    m = re.search(pattern, html, re.IGNORECASE)
                    if m:
                        return float(m.group(1).replace(",", "."))
        except Exception as e:
            print(f"  Bond price fetch failed ({url}): {e}", file=sys.stderr)
    return None


# ---------- Macro ----------

def fetch_macro() -> pd.DataFrame:
    """Light-touch macro snapshot via yfinance proxies. Macro module proper comes later."""
    rows = []
    today = dt.date.today().isoformat()
    proxies = [
        ("EUR/USD",       "EURUSD=X"),
        ("EUR Gov 10Y",   "^TNX"),     # placeholder; refine with FRED later
        ("Brent",         "BZ=F"),
        ("Gold",          "GC=F"),
        ("VIX",           "^VIX"),
    ]
    for name, tk in proxies:
        df = safe_yf_history(tk, period="5d")
        if df.empty:
            rows.append({"name": name, "ticker": tk, "value": np.nan, "as_of": today})
        else:
            rows.append({"name": name, "ticker": tk,
                         "value": float(df["Close"].iloc[-1]), "as_of": today})
    return pd.DataFrame(rows)

# ---------- Main ----------

def main():
    DATA.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    policy = load_policy()

    metrics = []
    price_frames = []
    holdings_frames = []

    print("Fetching per-candidate data...")
    for cand in policy["candidates"]:
        cid = cand["id"]
        ticker = cand.get("yf_ticker")
        # data_ticker (if key present) overrides yf_ticker for fetching.
        # Explicit null means "no yfinance source" (do not fall back to yf_ticker).
        if "data_ticker" in cand:
            data_ticker = cand["data_ticker"]   # may be a string or None
        else:
            data_ticker = ticker
        label = ticker or "—"
        if data_ticker and data_ticker != ticker:
            label += f" (data: {data_ticker})"
        print(f"  {cid:25s} ticker={label}")

        if data_ticker:
            hist = safe_yf_history(data_ticker, period="2y")
            if not hist.empty:
                close = hist["Close"].dropna()
                # Detect source currency and convert to EUR if needed
                target_ccy = cand.get("currency", "EUR")
                info_pe, info_yield, info_aum = np.nan, np.nan, np.nan
                src_ccy = target_ccy
                try:
                    info = yf.Ticker(data_ticker).info or {}
                    src_ccy = info.get("currency", target_ccy)
                    info_pe    = info.get("trailingPE", np.nan)
                    info_yield = info.get("yield", np.nan)
                    if info_yield is not None and not np.isnan(info_yield) and info_yield < 1:
                        info_yield *= 100
                    info_aum   = info.get("totalAssets", np.nan)
                    # P/E fallback: some Milan-listed ETFs don't report it; try pe_ticker
                    pe_ticker = cand.get("pe_ticker")
                    if pe_ticker and cand.get("has_equity_pe") and (
                        info_pe is None or (isinstance(info_pe, float) and np.isnan(info_pe))
                    ):
                        try:
                            pe_info = yf.Ticker(pe_ticker).info or {}
                            fallback_pe = pe_info.get("trailingPE", np.nan)
                            if fallback_pe and not np.isnan(float(fallback_pe)):
                                info_pe = float(fallback_pe)
                                print(f"    P/E from pe_ticker {pe_ticker}: {round(info_pe,1)}")
                        except Exception:
                            pass
                except Exception:
                    pass
                if src_ccy != target_ccy:
                    print(f"    converting {data_ticker} prices {src_ccy} -> {target_ccy}")
                    close = to_eur(close, src_ccy)
                tech = compute_technicals(close)
                # 20d avg volume (keep in source currency units — it's just for liquidity ranking)
                vol_20d = float(hist["Volume"].tail(20).mean()) if "Volume" in hist else np.nan
                metrics.append({
                    "id": cid, "name": cand["name"], "ticker": ticker,
                    "asset_class": cand["asset_class"],
                    "weight_min": cand["weight_min"], "weight_max": cand["weight_max"],
                    **tech,
                    "pe_ratio": info_pe if cand.get("has_equity_pe") else np.nan,
                    "yield_pct": info_yield, "aum": info_aum, "avg_volume_20d": vol_20d,
                })
                # also keep price history
                ph = close.reset_index().rename(columns={"Date": "date", "Close": "price"})
                ph["candidate_id"] = cid
                price_frames.append(ph[["candidate_id", "date", "price"]])
            else:
                metrics.append({"id": cid, "name": cand["name"], "ticker": ticker,
                                "asset_class": cand["asset_class"],
                                "weight_min": cand["weight_min"], "weight_max": cand["weight_max"]})
        elif cid == "cash":
            deposit_rate = fetch_ecb_it_deposit_rate()
            print(f"    Italian household deposit rate: {deposit_rate}%")
            metrics.append({
                "id": cid, "name": cand["name"], "ticker": None,
                "asset_class": cand["asset_class"],
                "weight_min": cand["weight_min"], "weight_max": cand["weight_max"],
                "yield_pct": deposit_rate,
            })
        elif cid == "btp_italia":
            isin = cand.get("isin")
            price = fetch_borsaitaliana_bond_price(isin) if isin else None
            real_yield = cand.get("static_yield_pct")   # net real YTM
            hicp = fetch_italian_hicp()
            # BTP Italia is inflation-linked: nominal yield = real yield + Italian HICP
            nominal_yield = round(real_yield + hicp, 2) if real_yield is not None else None
            if price:
                print(f"    BTP Italia price: {price}")
            print(f"    BTP Italia: real yield={real_yield}% + HICP={hicp}% = nominal {nominal_yield}%")
            metrics.append({
                "id": cid, "name": cand["name"], "ticker": None,
                "asset_class": cand["asset_class"],
                "weight_min": cand["weight_min"], "weight_max": cand["weight_max"],
                "last_price": price,
                "yield_pct": nominal_yield,
            })
        else:
            # Other assets without yfinance ticker
            metrics.append({"id": cid, "name": cand["name"], "ticker": None,
                            "asset_class": cand["asset_class"],
                            "weight_min": cand["weight_min"], "weight_max": cand["weight_max"]})

        # Holdings
        src = cand.get("holdings_source")
        if src == "ishares" and cand.get("ishares_product_id"):
            df_h = fetch_ishares_holdings(cand["ishares_product_id"], cid)
            if not df_h.empty:
                holdings_frames.append(df_h)
        # manual & none → handled below

    # Merge auto + manual holdings
    manual_csv = DATA / "holdings_top5_manual.csv"
    if manual_csv.exists():
        df_manual = pd.read_csv(manual_csv).dropna(subset=["security_name"], how="all")
        df_manual = df_manual[df_manual["security_name"].astype(str).str.strip() != ""]
        if not df_manual.empty:
            holdings_frames.append(df_manual)

    # Write outputs
    pd.DataFrame(metrics).to_csv(DATA / "metrics_snapshot.csv", index=False)
    print(f"  -> data/metrics_snapshot.csv ({len(metrics)} rows)")

    if price_frames:
        ph_all = pd.concat(price_frames, ignore_index=True)
        ph_all.to_parquet(DATA / "prices_history.parquet", index=False)
        print(f"  -> data/prices_history.parquet ({len(ph_all)} rows)")

    if holdings_frames:
        holdings_all = pd.concat(holdings_frames, ignore_index=True)
        holdings_all.to_csv(DATA / "holdings_top5.csv", index=False)
        print(f"  -> data/holdings_top5.csv ({len(holdings_all)} rows)")
    else:
        print("  -> no holdings data yet; populate holdings_top5_manual.csv")

    macro = fetch_macro()
    macro.to_csv(DATA / "macro_snapshot.csv", index=False)
    print(f"  -> data/macro_snapshot.csv ({len(macro)} rows)")

    print("Done.")

if __name__ == "__main__":
    main()
