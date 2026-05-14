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
import base64
import json
import datetime as dt
import subprocess
import sys
import time
import pandas as pd
import numpy as np
import yaml
import streamlit as st
import plotly.graph_objects as go


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


def ensure_data_fresh() -> None:
    if st.session_state.get("_data_checked"):
        return
    st.session_state["_data_checked"] = True
    DATA_TTL_SECONDS = 3600
    snapshot = DATA / "metrics_snapshot.csv"
    first_run = not snapshot.exists() or snapshot.stat().st_size == 0
    if not first_run and (time.time() - snapshot.stat().st_mtime) < DATA_TTL_SECONDS:
        return
    msg = "Caricamento dati in corso per la prima volta…" if first_run else "Aggiornamento dati in corso…"
    with st.spinner(msg):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_data.py")],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        st.error("⚠️ Aggiornamento dati fallito. Riprova premendo **Aggiorna Dati**.")
        st.stop()


def _github_save_scenarios(data: dict) -> bool:
    """Push scenarios.json to the GitHub repo via API. Returns True on success."""
    pat = st.secrets.get("github_pat")
    repo = st.secrets.get("github_repo")
    if not pat or not repo:
        return False

    content_str = json.dumps(data, indent=2, ensure_ascii=False)
    encoded = base64.b64encode(content_str.encode()).decode()

    api_url = f"https://api.github.com/repos/{repo}/contents/data/scenarios.json"
    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    r = requests.get(api_url, headers=headers, timeout=10)
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload: dict = {"message": "Aggiorna scenari", "content": encoded}
    if sha:
        payload["sha"] = sha

    r = requests.put(api_url, headers=headers, json=payload, timeout=10)
    return r.status_code in (200, 201)


st.set_page_config(page_title="Portafoglio di Famiglia", layout="wide", initial_sidebar_state="collapsed")
require_password()
ensure_data_fresh()

st.markdown("""
<style>
/* ── Global background ── */
[data-testid="stAppViewContainer"] { background-color: #f8fafc; }
[data-testid="stMain"] { background-color: #f8fafc; }

/* ── Hide sidebar entirely ── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Card helper ── */
.card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}

/* ── Primary buttons ── */
.stButton > button[kind="primary"] {
    background-color: #10b981 !important;
    border-color: #10b981 !important;
    color: white !important;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background-color: #059669 !important;
    border-color: #059669 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    border-bottom: 3px solid #10b981 !important;
    color: #10b981 !important;
}

/* ── Rounded dataframe container ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Alert boxes ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Header divider ── */
hr { margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

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

# ---------- Top header (replaces sidebar) ----------
hdr_left, hdr_center, hdr_right = st.columns([2, 3, 2])

with hdr_left:
    st.markdown("""
    <div class="card">
        <h2 style="margin:0;color:#1e293b;">Portafoglio di Famiglia</h2>
        <p style="margin:0;color:#64748b;font-size:0.9rem;">Strumento di allocazione investimenti</p>
    </div>
    """, unsafe_allow_html=True)

with hdr_center:
    cap_a, cap_b = st.columns(2)
    cap_a.metric("Capitale Totale", f"€{policy['capital']['total_eur']:,}")
    cap_b.metric("BTP Italia (fisso)", "€350.000 · 35%")

with hdr_right:
    _snap = DATA / "metrics_snapshot.csv"
    if _snap.exists() and _snap.stat().st_size > 0:
        _age_min = int((time.time() - _snap.stat().st_mtime) / 60)
        _color = "#10b981" if _age_min < 30 else "#f59e0b" if _age_min < 120 else "#ef4444"
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:0.75rem 1rem;">
            <span style="color:{_color};font-weight:600;">● Dati aggiornati {_age_min} min fa</span>
        </div>
        """, unsafe_allow_html=True)
    if st.button("🔄 Aggiorna Dati", type="primary"):
        with st.spinner("Aggiornamento dati in corso…"):
            _result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "fetch_data.py")],
                capture_output=True,
                text=True,
            )
        if _result.returncode != 0:
            st.error("⚠️ Aggiornamento fallito. Riprova.")
        else:
            st.cache_data.clear()
            st.rerun()

if metrics.empty:
    st.warning("⚠️ Dati non disponibili. Premi **Aggiorna Dati** per caricare le informazioni più recenti.")

st.divider()

tab_design, tab_compare, tab_scenarios = st.tabs(
    ["Allocazione 🎚️", "Confronto 📊", "Scenari 💾"]
)

# Asset class display labels and colors (shared between donut and bar charts)
_CLASS_LABELS = {
    "cash":                        "Liquidità",
    "bond_govt_inflation_linked":  "Obbligazioni",
    "commodities":                 "Materie Prime",
    "equity_developed_global":     "Azionario",
    "equity_developed_europe":     "Azionario",
    "equity_emerging":             "Azionario",
    "equity_thematic":             "Tematici",
}
_CLASS_COLORS = {
    "Liquidità":     "#3b82f6",
    "Obbligazioni":  "#f59e0b",
    "Materie Prime": "#8b5cf6",
    "Azionario":     "#10b981",
    "Tematici":      "#f97316",
}

# ============================================================
# TAB 1 — ALLOCAZIONE (Designer)
# ============================================================
with tab_design:
    st.header("Allocazione del Portafoglio")
    st.caption("Imposta i pesi entro gli intervalli consentiti. Le voci fisse mostrano un valore bloccato.")

    cols = st.columns([2, 1, 1, 1])
    cols[0].markdown("**Investimento**")
    cols[1].markdown("**Intervallo**")
    cols[2].markdown("**Peso (%)**")
    cols[3].markdown("**Valore EUR**")

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
            col3.write(f"**{w*100:.1f}%** (fisso)")
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
    equity_w = sum(weights[c["id"]] for c in candidates if c["asset_class"].startswith("equity"))
    thematic_w = sum(weights[c["id"]] for c in candidates if c.get("is_thematic"))
    cash_w = weights.get("cash", 0)
    bond_w = sum(weights[c["id"]] for c in candidates if c["asset_class"].startswith("bond"))
    st.divider()

    left_col, right_col = st.columns([3, 2])

    with left_col:
        # ── Constraint metrics ──
        cs = st.columns(5)
        cs[0].metric("Totale Pesi", f"{total_w*100:.2f}%",
                     delta=f"{(total_w-1)*100:+.2f} pp", delta_color="inverse")
        cs[1].metric("Azionario %", f"{equity_w*100:.1f}%")
        cs[2].metric("Liquidità %", f"{cash_w*100:.1f}%")
        cs[3].metric("Obbligazioni %", f"{bond_w*100:.1f}%")
        cs[4].metric("Tematici %", f"{thematic_w*100:.1f}%")

        # ── Constraint warnings ──
        issues = []
        if abs(total_w - constraints["weights_sum_to"]) > 0.001:
            issues.append(f"❌ I pesi sommano a {total_w*100:.2f}%, devono essere 100%")
        if cash_w < constraints["cash_min"]:
            issues.append(f"⚠️ Liquidità {cash_w*100:.1f}% < minimo {constraints['cash_min']*100:.0f}%")
        if equity_w < constraints["total_equity_min"]:
            issues.append(f"⚠️ Azionario {equity_w*100:.1f}% < minimo {constraints['total_equity_min']*100:.0f}%")
        if equity_w > constraints["total_equity_max"]:
            issues.append(f"⚠️ Azionario {equity_w*100:.1f}% > massimo {constraints['total_equity_max']*100:.0f}%")
        if thematic_w > constraints["max_total_thematic"]:
            issues.append(f"⚠️ Tematici totale {thematic_w*100:.1f}% > massimo {constraints['max_total_thematic']*100:.0f}%")
        for c in candidates:
            if c.get("is_thematic") and weights[c["id"]] > constraints["max_single_thematic"]:
                issues.append(f"⚠️ {c['name']} {weights[c['id']]*100:.1f}% > limite tematico {constraints['max_single_thematic']*100:.0f}%")
        if issues:
            for msg in issues:
                st.warning(msg)
        else:
            st.success("✅ Tutti i vincoli rispettati")

    with right_col:
        # ── Live donut chart grouped by asset class ──
        donut_data: dict = {}
        for c in candidates:
            label = _CLASS_LABELS.get(c["asset_class"], c["asset_class"])
            donut_data[label] = donut_data.get(label, 0) + weights[c["id"]]

        d_labels = list(donut_data.keys())
        d_values = list(donut_data.values())
        d_colors = [_CLASS_COLORS.get(lbl, "#94a3b8") for lbl in d_labels]

        fig_donut = go.Figure(go.Pie(
            labels=d_labels,
            values=d_values,
            hole=0.5,
            marker=dict(colors=d_colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="%{label}: %{percent}<extra></extra>",
        ))
        fig_donut.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
            title=dict(text="Allocazione per Classe", font=dict(size=15), x=0.5),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True, key="donut_chart")

    # ── Blended portfolio metrics ──
    st.divider()
    st.subheader("Metriche del Portafoglio")
    rows = []
    for c in candidates:
        cid = c["id"]; w = weights[cid]
        m = metrics_by_id.get(cid, {})
        rows.append({
            "id": cid, "weight": w,
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

    pe_val = wavg("pe"); yld_val = wavg("yield")
    vol_val = wavg("vol_252d"); dd_val = wavg("ddown")

    bm_cols = st.columns(4)
    bm_cols[0].metric("P/E Medio", f"{pe_val:.1f}" if not np.isnan(pe_val) else "n/d")
    bm_cols[1].metric("Rendimento", f"{yld_val:.2f}%" if not np.isnan(yld_val) else "n/d")
    bm_cols[2].metric("Rischio (Vol.)", f"{vol_val:.1f}%" if not np.isnan(vol_val) else "n/d")
    bm_cols[3].metric("Calo Massimo", f"{dd_val:.1f}%" if not np.isnan(dd_val) else "n/d")

    st.caption("P/E e rendimento sono calcolati solo sugli strumenti con dati disponibili (liquidità e obbligazioni escluse).")

    # ── Salva scenario ──
    st.divider()
    sc_col1, sc_col2 = st.columns([3, 1])
    scenario_name = sc_col1.text_input("Nome Scenario", value="Bozza 1")
    if sc_col2.button("💾 Salva Scenario", type="primary"):
        SCENARIOS_PATH.parent.mkdir(exist_ok=True)
        existing = json.loads(SCENARIOS_PATH.read_text()) if SCENARIOS_PATH.exists() else {}
        existing[scenario_name] = {
            "weights": weights,
            "saved_at": pd.Timestamp.now().isoformat(),
        }
        SCENARIOS_PATH.write_text(json.dumps(existing, indent=2))
        if _github_save_scenarios(existing):
            st.success(f"Scenario '{scenario_name}' salvato e sincronizzato.")
        else:
            st.success(f"Scenario '{scenario_name}' salvato.")

# ============================================================
# TAB 2 — CONFRONTO (Compare)
# ============================================================
with tab_compare:
    st.header("Confronto Candidati")
    if metrics.empty:
        st.warning("⚠️ Dati non disponibili. Premi **Aggiorna Dati** per caricare le informazioni più recenti.")
    else:
        m = metrics.copy()
        m = m.merge(
            pd.DataFrame([{"id": c["id"], "is_thematic": c.get("is_thematic", False),
                           "asset_class": c["asset_class"]} for c in candidates]),
            on="id", how="left", suffixes=("", "_policy"),
        )
        if "asset_class_policy" in m.columns:
            m["asset_class"] = m["asset_class_policy"].fillna(m.get("asset_class", ""))
            m.drop(columns=["asset_class_policy"], inplace=True)

        display_cols = [
            "name", "ticker", "asset_class", "last_price",
            "pct_vs_ma50", "pct_vs_ma200", "drawdown_from_52w_high",
            "pe_ratio", "yield_pct", "vol_252d_ann", "avg_volume_20d", "aum",
        ]
        display_cols = [c for c in display_cols if c in m.columns]
        nice = {
            "name": "Nome", "ticker": "Ticker", "asset_class": "Classe",
            "last_price": "Prezzo",
            "pct_vs_ma50": "vs MA50 %", "pct_vs_ma200": "vs MA200 %",
            "drawdown_from_52w_high": "Calo Max %",
            "pe_ratio": "P/E", "yield_pct": "Rendimento %",
            "vol_252d_ann": "Volatilità 1y %", "avg_volume_20d": "Vol(20g)", "aum": "AUM",
        }
        view = m[display_cols].rename(columns=nice)
        st.dataframe(
            view.style.format({
                "Prezzo": "{:.2f}", "vs MA50 %": "{:+.1f}", "vs MA200 %": "{:+.1f}",
                "Calo Max %": "{:.1f}", "P/E": "{:.1f}", "Rendimento %": "{:.2f}",
                "Volatilità 1y %": "{:.1f}", "Vol(20g)": "{:,.0f}", "AUM": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        # ── Bar charts ──
        st.subheader("Analisi Comparativa")

        def make_hbar(df_src, x_col, title, x_suffix=""):
            sub = df_src[["Nome", x_col, "color"]].dropna(subset=[x_col])
            fig = go.Figure(go.Bar(
                x=sub[x_col],
                y=sub["Nome"],
                orientation="h",
                marker_color=sub["color"],
                hovertemplate=f"%{{y}}: %{{x:.2f}}{x_suffix}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(size=14), x=0),
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(ticksuffix=x_suffix),
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            return fig

        mdf = m[["name", "asset_class", "yield_pct", "vol_252d_ann",
                  "drawdown_from_52w_high", "pe_ratio"]].copy()
        mdf = mdf.rename(columns={
            "name": "Nome",
            "yield_pct": "Rendimento %",
            "vol_252d_ann": "Volatilità 1y %",
            "drawdown_from_52w_high": "Calo Massimo %",
            "pe_ratio": "P/E Medio",
        })
        mdf["color"] = mdf["asset_class"].map(
            lambda ac: _CLASS_COLORS.get(_CLASS_LABELS.get(ac, ""), "#94a3b8")
        )

        row1_l, row1_r = st.columns(2)
        with row1_l:
            st.plotly_chart(
                make_hbar(mdf, "Rendimento %", "Rendimento %", "%"),
                use_container_width=True, key="bar_yield",
            )
        with row1_r:
            st.plotly_chart(
                make_hbar(mdf, "Volatilità 1y %", "Volatilità 1y %", "%"),
                use_container_width=True, key="bar_vol",
            )
        row2_l, row2_r = st.columns(2)
        with row2_l:
            st.plotly_chart(
                make_hbar(mdf, "Calo Massimo %", "Calo Massimo %", "%"),
                use_container_width=True, key="bar_dd",
            )
        with row2_r:
            st.plotly_chart(
                make_hbar(mdf, "P/E Medio", "P/E Medio"),
                use_container_width=True, key="bar_pe",
            )

# ============================================================
# TAB 3 — SCENARIOS
# ============================================================
with tab_scenarios:
    st.header("Scenari Salvati")
    if not SCENARIOS_PATH.exists():
        st.info("Nessuno scenario salvato. Crea uno scenario dalla scheda Allocazione.")
    else:
        scenarios = json.loads(SCENARIOS_PATH.read_text())
        if not scenarios:
            st.info("Nessuno scenario salvato. Crea uno scenario dalla scheda Allocazione.")
        else:
            chosen = st.multiselect("Confronta scenari", list(scenarios.keys()),
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
            del_choice = st.selectbox("Elimina scenario", ["—"] + list(scenarios.keys()))
            if del_choice != "—" and st.button("🗑️ Elimina", type="primary"):
                del scenarios[del_choice]
                SCENARIOS_PATH.write_text(json.dumps(scenarios, indent=2))
                _github_save_scenarios(scenarios)
                st.rerun()
