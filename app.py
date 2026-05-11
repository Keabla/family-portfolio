"""
Streamlit allocation designer (Tool 1).

Run: `streamlit run app.py`

Tabs:
  1. Designer  — sliders per candidate within allowed ranges; live blended metrics
  2. Compare   — side-by-side metrics table for all candidates
  3. Overlap   — pairwise top-5 holdings overlap
  4. Scenarios — save/compare named candidate allocations

Data inputs (from data/):
  - metrics_snapshot.csv
  - holdings_top5.csv (auto + manual)
  - prices_history.parquet
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
import yaml
import streamlit as st


def require_password():
    """Block all rendering until correct password is entered."""
    if "app_password" not in st.secrets:
        st.error("App password is not configured. Set `app_password` in secrets.")
        st.stop()
    if st.session_state.get("auth_ok"):
        return
    st.title("🔒 Family Portfolio")
    st.caption("Inserisci la password per accedere.")
    pw = st.text_input("Password", type="password", key="_pw_input")
    if st.button("Entra", type="primary") or pw:
        if pw == st.secrets["app_password"]:
            st.session_state["auth_ok"] = True
            st.rerun()
        elif pw:
            st.error("Password errata.")
    st.stop()


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POLICY_PATH = ROOT / "policy.yaml"
SCENARIOS_PATH = DATA / "scenarios.json"

st.set_page_config(page_title="Allocation Designer", layout="wide")
require_password()

# ---------- Data loading (cached) ----------
@st.cache_data
def load_policy() -> dict:
    with open(POLICY_PATH, "r") as f:
        return yaml.safe_load(f)

@st.cache_data(ttl=300)   # re-read CSVs every 5 min so a fetch run is picked up on next interaction
def load_metrics() -> pd.DataFrame:
    p = DATA / "metrics_snapshot.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=300)
def load_holdings() -> pd.DataFrame:
    p = DATA / "holdings_top5.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_prices() -> pd.DataFrame:
    p = DATA / "prices_history.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

policy = load_policy()
metrics = load_metrics()
holdings = load_holdings()
prices = load_prices()
candidates = policy["candidates"]
constraints = policy["constraints"]

cand_by_id = {c["id"]: c for c in candidates}
metrics_by_id = metrics.set_index("id").to_dict("index") if not metrics.empty else {}

# ---------- Sidebar ----------
st.sidebar.title("Allocation Designer")
st.sidebar.caption(f"Capital: €{policy['capital']['total_eur']:,}")
st.sidebar.caption(f"BTP fixed: 35% (€350,000)")
if metrics.empty:
    st.sidebar.error("Run `python scripts/fetch_data.py` first.")
last_refresh = (DATA / "metrics_snapshot.csv").stat().st_mtime if (DATA / "metrics_snapshot.csv").exists() else None
if last_refresh:
    import datetime as dt
    st.sidebar.caption(f"Data refreshed: {dt.datetime.fromtimestamp(last_refresh):%Y-%m-%d %H:%M}")

tab_design, tab_compare, tab_overlap, tab_scenarios = st.tabs(
    ["🎚️ Designer", "📊 Compare", "🔁 Overlap", "💾 Scenarios"]
)

# ============================================================
# TAB 1 — DESIGNER
# ============================================================
with tab_design:
    st.header("Allocation Designer")
    st.caption("Set weights within allowed ranges. Locked items show as fixed.")

    cols = st.columns([2, 1, 1, 1])
    cols[0].markdown("**Candidate**")
    cols[1].markdown("**Range**")
    cols[2].markdown("**Weight (%)**")
    cols[3].markdown("**EUR**")

    weights = {}
    total_eur = policy["capital"]["total_eur"]

    for c in candidates:
        cid = c["id"]
        wmin, wmax = c["weight_min"], c["weight_max"]
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        col1.write(f"{c['name']}  \n*{c['asset_class']}*")
        col2.write(f"{wmin*100:.0f}–{wmax*100:.0f}%")
        if wmin == wmax:
            w = wmin
            col3.write(f"**{w*100:.1f}%** (locked)")
        else:
            default = (wmin + wmax) / 2
            w = col3.slider(
                f"w_{cid}", min_value=float(wmin*100), max_value=float(wmax*100),
                value=float(default*100), step=0.5, key=f"slider_{cid}",
                label_visibility="collapsed",
            ) / 100.0
        col4.write(f"€{w*total_eur:,.0f}")
        weights[cid] = w

    total_w = sum(weights.values())
    st.divider()

    # Constraint check
    cs = st.columns(5)
    cs[0].metric("Sum weights", f"{total_w*100:.2f}%",
                 delta=f"{(total_w-1)*100:+.2f} pp", delta_color="inverse")
    equity_w = sum(weights[c["id"]] for c in candidates if c["asset_class"].startswith("equity"))
    cs[1].metric("Equity %", f"{equity_w*100:.1f}%")
    thematic_w = sum(weights[c["id"]] for c in candidates if c.get("is_thematic"))
    cs[2].metric("Thematic %", f"{thematic_w*100:.1f}%")
    cash_w = weights.get("cash", 0)
    cs[3].metric("Cash %", f"{cash_w*100:.1f}%")
    bond_w = sum(weights[c["id"]] for c in candidates if c["asset_class"].startswith("bond"))
    cs[4].metric("Bond %", f"{bond_w*100:.1f}%")

    # Constraint warnings
    issues = []
    if abs(total_w - constraints["weights_sum_to"]) > 0.001:
        issues.append(f"❌ Weights sum to {total_w*100:.2f}%, must equal 100%")
    if cash_w < constraints["cash_min"]:
        issues.append(f"⚠️  Cash {cash_w*100:.1f}% < min {constraints['cash_min']*100:.0f}%")
    if equity_w < constraints["total_equity_min"]:
        issues.append(f"⚠️  Equity {equity_w*100:.1f}% < min {constraints['total_equity_min']*100:.0f}%")
    if equity_w > constraints["total_equity_max"]:
        issues.append(f"⚠️  Equity {equity_w*100:.1f}% > max {constraints['total_equity_max']*100:.0f}%")
    if thematic_w > constraints["max_total_thematic"]:
        issues.append(f"⚠️  Thematic total {thematic_w*100:.1f}% > max {constraints['max_total_thematic']*100:.0f}%")
    for c in candidates:
        if c.get("is_thematic") and weights[c["id"]] > constraints["max_single_thematic"]:
            issues.append(f"⚠️  {c['name']} {weights[c['id']]*100:.1f}% > thematic cap {constraints['max_single_thematic']*100:.0f}%")
    if issues:
        for m in issues: st.warning(m)
    else:
        st.success("✅ All constraints satisfied")

    # Blended metrics
    st.subheader("Blended portfolio metrics")
    rows = []
    for c in candidates:
        cid = c["id"]; w = weights[cid]
        m = metrics_by_id.get(cid, {})
        rows.append({
            "id": cid, "weight": w,
            "ter": np.nan,  # placeholder; populate via policy if you add it
            "pe": m.get("pe_ratio", np.nan) if c.get("has_equity_pe") else np.nan,
            "yield": m.get("yield_pct", np.nan),
            "vol_252d": m.get("vol_252d_ann", np.nan),
            "ddown": m.get("drawdown_from_52w_high", np.nan),
        })
    df = pd.DataFrame(rows)

    def wavg(col):
        v = df[col].astype(float); w = df["weight"].astype(float)
        mask = ~v.isna() & (w > 0)
        return float((v[mask] * w[mask]).sum() / w[mask].sum()) if mask.any() else np.nan

    bm_cols = st.columns(4)
    bm_cols[0].metric("Weighted P/E", f"{wavg('pe'):.1f}" if not np.isnan(wavg('pe')) else "n/a")
    bm_cols[1].metric("Weighted yield", f"{wavg('yield'):.2f}%" if not np.isnan(wavg('yield')) else "n/a")
    bm_cols[2].metric("Weighted 1y vol", f"{wavg('vol_252d'):.1f}%" if not np.isnan(wavg('vol_252d')) else "n/a")
    bm_cols[3].metric("Weighted ddown", f"{wavg('ddown'):.1f}%" if not np.isnan(wavg('ddown')) else "n/a")

    st.caption("P/E and yield are weighted only over candidates with valid values (cash & bond ETFs excluded).")

    # Save scenario
    st.divider()
    sc_col1, sc_col2 = st.columns([3, 1])
    scenario_name = sc_col1.text_input("Scenario name", value="Draft 1")
    if sc_col2.button("💾 Save scenario", type="primary"):
        SCENARIOS_PATH.parent.mkdir(exist_ok=True)
        existing = json.loads(SCENARIOS_PATH.read_text()) if SCENARIOS_PATH.exists() else {}
        existing[scenario_name] = {
            "weights": weights,
            "saved_at": pd.Timestamp.now().isoformat(),
        }
        SCENARIOS_PATH.write_text(json.dumps(existing, indent=2))
        st.success(f"Saved scenario '{scenario_name}'")

# ============================================================
# TAB 2 — COMPARE
# ============================================================
with tab_compare:
    st.header("Candidate comparison")
    if metrics.empty:
        st.warning("No metrics data. Run scripts/fetch_data.py first.")
    else:
        m = metrics.copy()
        m = m.merge(
            pd.DataFrame([{"id": c["id"], "is_thematic": c.get("is_thematic", False)} for c in candidates]),
            on="id", how="left"
        )
        display_cols = [
            "name", "ticker", "asset_class", "last_price",
            "pct_vs_ma50", "pct_vs_ma200", "drawdown_from_52w_high",
            "pe_ratio", "yield_pct", "vol_252d_ann", "avg_volume_20d", "aum",
        ]
        display_cols = [c for c in display_cols if c in m.columns]
        nice = {
            "name": "Name", "ticker": "Ticker", "asset_class": "Class",
            "last_price": "Price",
            "pct_vs_ma50": "vs MA50 %", "pct_vs_ma200": "vs MA200 %",
            "drawdown_from_52w_high": "DD 52w %",
            "pe_ratio": "P/E", "yield_pct": "Yield %",
            "vol_252d_ann": "Vol 1y %", "avg_volume_20d": "Vol(20d)", "aum": "AUM",
        }
        view = m[display_cols].rename(columns=nice)
        st.dataframe(
            view.style.format({
                "Price": "{:.2f}", "vs MA50 %": "{:+.1f}", "vs MA200 %": "{:+.1f}",
                "DD 52w %": "{:.1f}", "P/E": "{:.1f}", "Yield %": "{:.2f}",
                "Vol 1y %": "{:.1f}", "Vol(20d)": "{:,.0f}", "AUM": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

# ============================================================
# TAB 3 — OVERLAP
# ============================================================
with tab_overlap:
    st.header("Top-5 holdings overlap")
    if holdings.empty:
        st.warning("No holdings data. Populate data/holdings_top5_manual.csv and re-run fetch_data.py.")
    else:
        ids_present = sorted(holdings["candidate_id"].unique())
        # Pairwise overlap by security_name (case-insensitive)
        overlap_rows = []
        sets = {
            cid: set(holdings[holdings["candidate_id"] == cid]["security_name"]
                     .dropna().astype(str).str.strip().str.lower())
            for cid in ids_present
        }
        for i, a in enumerate(ids_present):
            for b in ids_present[i+1:]:
                inter = sets[a] & sets[b]
                if inter:
                    overlap_rows.append({
                        "A": a, "B": b,
                        "shared": len(inter),
                        "names": ", ".join(sorted(inter)),
                    })
        if overlap_rows:
            st.dataframe(pd.DataFrame(overlap_rows).sort_values("shared", ascending=False),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No overlaps in current top-5 lists.")

        st.divider()
        st.subheader("Top-5 by candidate")
        for cid in ids_present:
            sub = holdings[holdings["candidate_id"] == cid].sort_values("rank")
            with st.expander(f"{cand_by_id.get(cid,{}).get('name', cid)}"):
                st.dataframe(sub[["rank","security_name","weight_pct","sector","country"]],
                             use_container_width=True, hide_index=True)

# ============================================================
# TAB 4 — SCENARIOS
# ============================================================
with tab_scenarios:
    st.header("Saved scenarios")
    if not SCENARIOS_PATH.exists():
        st.info("No scenarios saved yet. Save one from the Designer tab.")
    else:
        scenarios = json.loads(SCENARIOS_PATH.read_text())
        if not scenarios:
            st.info("No scenarios saved yet.")
        else:
            chosen = st.multiselect("Compare scenarios", list(scenarios.keys()),
                                    default=list(scenarios.keys())[:3])
            if chosen:
                rows = []
                for name in chosen:
                    w = scenarios[name]["weights"]
                    row = {"Scenario": name}
                    for c in candidates:
                        row[c["name"]] = f"{w.get(c['id'], 0)*100:.1f}%"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.divider()
            del_choice = st.selectbox("Delete scenario", ["—"] + list(scenarios.keys()))
            if del_choice != "—" and st.button("🗑️  Delete"):
                del scenarios[del_choice]
                SCENARIOS_PATH.write_text(json.dumps(scenarios, indent=2))
                st.rerun()
