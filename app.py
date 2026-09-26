"""Catalyst Scanner — Google Finance-style Streamlit UI.

    streamlit run app.py      (or simply: python app.py — it relaunches itself under Streamlit)

Reads cached analyses from SQLite on load (instant). LLM calls only happen when you press
"Run analysis" in the sidebar. Price charts use yfinance (DEMO-ONLY data source).
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    # Started as a plain script (python app.py / IDE "Run" button)? Relaunch under `streamlit run`.
    from streamlit.runtime import exists as _streamlit_running
    if not _streamlit_running():
        from streamlit.web import cli as _stcli
        sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
        sys.exit(_stcli.main())

import html
from dataclasses import replace
from datetime import date, timedelta

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit Community Cloud: secrets live in st.secrets (TOML), not .env. Mirror top-level string secrets into
# env vars BEFORE scanner.config reads them. Locally there's no secrets file, so this is a no-op.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from scanner import evaluation
from scanner.config import WATCHLIST, get_settings
from scanner.ingestion import FixtureNewsSource
from scanner.models import Assessment
from scanner.pipeline import Pipeline
from scanner.prices import YFinancePriceSource, parse_event_date, verify_move
from scanner.storage import Store

st.set_page_config(page_title="Catalyst Scanner", page_icon="📈", layout="wide")

ALL_PROVIDERS = {"anthropic": "Claude Sonnet 5 (API)", "lmstudio": "Gemma 4 12B (local)",
                 "nim": "Gemma 4 31B (NVIDIA NIM)"}
_S = get_settings()
# Only providers enabled via ENABLED_PROVIDERS are ever offered (hosted deploy: "nim" only).
PROVIDERS = {k: v for k, v in ALL_PROVIDERS.items() if k in _S.enabled_providers} or {"nim": ALL_PROVIDERS["nim"]}
DEFAULT_PROVIDER = next(iter(PROVIDERS))
GREEN, GREEN_BG, RED, RED_BG = "#137333", "#e6f4ea", "#a50e0e", "#fce8e6"
AMBER, AMBER_BG, GREY, GREY_BG = "#b06000", "#fef7e0", "#5f6368", "#f1f3f4"
BADGE = {
    Assessment.STRONG_CATALYST: (GREEN, GREEN_BG, "STRONG CATALYST"),
    Assessment.PUFFY_WEAK: (RED, RED_BG, "PUFFY / WEAK CATALYST"),
    Assessment.MIXED_UNCLEAR: (AMBER, AMBER_BG, "MIXED / UNCLEAR"),
    Assessment.NO_FRESH_CATALYST: (GREY, GREY_BG, "NO FRESH CATALYST"),
}
RANGES = {"1M": 30, "6M": 182, "YTD": None, "1Y": 365, "5Y": 365 * 5}

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&family=Google+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Google Sans', Roboto, Arial, sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
.strip { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
.tcard { flex:1 1 200px; border:1px solid #dadce0; border-radius:8px; padding:10px 14px; text-decoration:none!important;
         color:#202124!important; background:#fff; transition:box-shadow .15s; }
.tcard:hover { box-shadow:0 1px 6px rgba(32,33,36,.28); }
.tcard.sel { border-color:#1a73e8; box-shadow:0 0 0 1px #1a73e8; }
.tcard .row { display:flex; justify-content:space-between; align-items:center; }
.tcard .sym { font-weight:500; font-size:14px; }
.tcard .name { color:#5f6368; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.chip { display:inline-block; border-radius:6px; padding:2px 8px; font-size:13px; font-weight:500; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; }
.h-name { font-size:26px; color:#202124; margin:4px 0 0 0; }
.h-sub { color:#5f6368; font-size:13px; }
.h-price { font-size:36px; color:#202124; margin-right:12px; }
.card { border:1px solid #dadce0; border-radius:8px; padding:16px 18px; background:#fff; margin-bottom:12px; }
.card h4 { margin:0 0 10px 0; font-size:15px; font-weight:500; color:#202124; }
.kv { display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid #f1f3f4; font-size:14px; }
.kv:last-child { border-bottom:none; }
.kv .k { color:#5f6368; } .kv .v { color:#202124; font-weight:500; text-align:right; max-width:62%; }
.banner { border-radius:8px; padding:14px 18px; margin:6px 0 14px 0; }
.banner .big { font-size:20px; font-weight:500; }
.banner .watch { font-size:14px; margin-top:2px; }
.li { font-size:14px; padding:4px 0 4px 14px; position:relative; color:#202124; }
.li:before { content:''; position:absolute; left:0; top:11px; width:6px; height:6px; border-radius:50%; background:var(--c); }
.note { font-size:12.5px; color:#5f6368; padding:2px 0; }
.guard { font-size:13px; color:#a50e0e; background:#fce8e6; border-radius:6px; padding:6px 10px; margin:4px 0; }
.news { font-size:12.5px; color:#5f6368; } .news a { color:#1a0dab; font-size:15px; text-decoration:none; }
.demo { font-size:11.5px; color:#80868b; }
.sect { font-size:13px; font-weight:500; color:#5f6368; margin:2px 0 6px 0; }
.wl-head { font-size:14px; font-weight:500; color:#202124; margin:10px 0 4px 0; }
.wlist { display:flex; flex-direction:column; }
.witem { display:flex; align-items:center; justify-content:space-between; gap:6px; padding:6px 8px;
         border-bottom:1px solid #e8eaed; border-radius:6px; text-decoration:none!important; color:#202124!important; }
.witem:hover { background:#e8f0fe; }
.witem.sel { background:#e8f0fe; box-shadow:inset 3px 0 0 #1a73e8; }
.witem .wt { font-weight:500; font-size:13px; }
.witem .wp { font-size:13px; margin-right:6px; }
.witem .chip { font-size:11.5px; padding:1px 6px; }
.witem .wna { font-size:11px; color:#80868b; }
.witem.na .wt { color:#80868b; }
</style>""", unsafe_allow_html=True)


def esc(s) -> str:
    """HTML-escape and neutralize '$' (Streamlit markdown would treat $...$ as LaTeX)."""
    return html.escape(str(s)).replace("$", "&#36;")


def chip(pct: float | None, suffix: str = "") -> str:
    if pct is None:
        return f'<span class="chip" style="color:{GREY};background:{GREY_BG}">n/a</span>'
    up = pct >= 0
    c, bg, arrow = (GREEN, GREEN_BG, "▲") if up else (RED, RED_BG, "▼")
    return f'<span class="chip" style="color:{c};background:{bg}">{arrow} {abs(pct):.2f}%{esc(suffix)}</span>'


def yn(v: str) -> str:
    color = {"Yes": GREEN, "No": RED, "Not verified": AMBER, "Unknown": AMBER}.get(v, GREY)
    return f'<span style="color:{color}">{esc(v)}</span>'


# ---------------- data ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def price_history(ticker: str) -> pd.DataFrame:
    try:
        return YFinancePriceSource().history(ticker, date.today() - timedelta(days=365 * 5), date.today())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def watchlist_quotes(tickers: tuple[str, ...]) -> dict:
    """One batched request for all watchlist tickers -> {ticker: {last, day, date} | None}."""
    try:
        import yfinance as yf
        closes = yf.download(list(tickers), period="1mo", interval="1d", auto_adjust=False,
                             progress=False, threads=True)["Close"]
    except Exception:
        return {t: None for t in tickers}
    out = {}
    for t in tickers:
        s = closes[t].dropna() if t in closes else pd.Series(dtype=float)
        out[t] = None if len(s) < 2 else {"last": float(s.iloc[-1]), "day": float((s.iloc[-1] / s.iloc[-2] - 1) * 100),
                                          "date": f"{s.index[-1]:%b %d, %Y}"}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def market_cap(ticker: str) -> float | None:
    try:
        import yfinance as yf
        return float(yf.Ticker(ticker).fast_info["market_cap"])
    except Exception:
        return None


def settings_for(provider: str):
    return replace(get_settings(), ai_provider=provider)


def load_results(provider: str, run_missing: bool = False) -> dict:
    s = settings_for(provider)
    source = FixtureNewsSource(s.fixture_path)
    cases = source.load_cases()
    store = Store(s.db_path)
    try:
        pipe = Pipeline(s, store)
        out = {}
        for c in cases:
            res = pipe.process(c.event, cache_only=not run_missing)
            ev = evaluation.evaluate(c.expected_classification, res.analysis) if res.analysis else None
            out[c.event.ticker] = (c, res, ev)
        return {"results": out, "skipped": source.skipped, "model": pipe.model}
    finally:
        store.close()


def clear_provider_cache(provider: str):
    s = settings_for(provider)
    store = Store(s.db_path)
    store.clear_provider(provider, s.model_for(provider))
    store.close()


# ---------------- sidebar ----------------
qp = st.query_params
provider = qp.get("p") if qp.get("p") in PROVIDERS else DEFAULT_PROVIDER
with st.sidebar:
    st.markdown("### 📈 Catalyst Scanner")
    st.caption("News-catalyst decision support. **Never places trades.**")
    provider = st.radio("AI provider", list(PROVIDERS), format_func=PROVIDERS.get, index=list(PROVIDERS).index(provider))
    qp["p"] = provider
    # Only analyzes cases with no cached result -> at most one API call per test case, safe to leave public.
    if st.button("▶ Run analysis (missing cases)", width="stretch"):
        with st.spinner(f"Analyzing with {PROVIDERS[provider]}…"):
            load_results(provider, run_missing=True)
    # Re-run spends credits on every click -> gated behind RERUN_PASSCODE when one is configured.
    unlocked = True
    if _S.rerun_passcode:
        code = st.text_input("Re-run passcode", type="password", help="Owner only: re-running spends API credits.")
        unlocked = code == _S.rerun_passcode
    if st.button("↻ Re-run all (clear this provider's cache)", width="stretch", disabled=not unlocked):
        clear_provider_cache(provider)
        with st.spinner(f"Re-analyzing with {PROVIDERS[provider]}…"):
            load_results(provider, run_missing=True)
    data = load_results(provider)
    with st.expander(f"Pending test cases ({len(data['skipped'])})"):
        for sid, missing in data["skipped"]:
            st.markdown(f"<div class='note'><b>{esc(sid)}</b><br>awaiting real data ({esc(', '.join(missing))})</div>",
                        unsafe_allow_html=True)

results = data["results"]
tickers = list(results)
selected = qp.get("t") if (qp.get("t") in results or qp.get("t") in WATCHLIST) else tickers[0]
qp["t"] = selected

# ---------------- sidebar watchlist (all 50 configured tickers, client order) ----------------
with st.sidebar:
    quotes = watchlist_quotes(WATCHLIST)
    n_ok = sum(1 for q in quotes.values() if q)
    st.markdown(f"<div class='wl-head'>Watchlist <span class='demo'>· {len(WATCHLIST)} tickers · "
                f"{len(WATCHLIST) - n_ok} no quote</span></div>", unsafe_allow_html=True)
    items = []
    for t in WATCHLIST:
        q = quotes.get(t)
        if q:
            body = f'<span><span class="wp">&#36;{q["last"]:,.2f}</span>{chip(q["day"])}</span>'
            cls = "witem"
        else:
            body = '<span class="wna">no quote</span>'
            cls = "witem na"
        items.append(f'<a class="{cls} {"sel" if t == selected else ""}" href="?t={t}&p={provider}" target="_self">'
                     f'<span class="wt">{t}</span>{body}</a>')
    with st.container(height=440, border=False):
        st.markdown(f'<div class="wlist">{"".join(items)}</div>', unsafe_allow_html=True)
    as_of = max(q["date"] for q in quotes.values() if q) if n_ok else "n/a"
    st.markdown(f"<div class='demo'>Last close vs previous close, as of {as_of}. \"no quote\" = Yahoo has no "
                "current data (likely delisted/renamed).<br>Prices: yfinance — unofficial, <b>demo-only</b>. "
                "Production: Alpaca market-data API.<br>News: fixture replay (data/test_cases.json).</div>",
                unsafe_allow_html=True)

tab_stock, tab_score = st.tabs(["Stocks", "Scorecard & model comparison"])

# ---------------- stock view ----------------
with tab_stock:
    # Ticker strip (Google Finance "You may be interested in")
    cards = []
    for t in tickers:
        c, res, ev = results[t]
        df = price_history(t)
        last = df["Close"].iloc[-1] if not df.empty else None
        day = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100 if len(df) > 1 else None
        dot = BADGE[res.analysis.assessment][0] if res.analysis else "#dadce0"
        price = f"&#36;{last:,.2f}" if last is not None else "—"
        cards.append(
            f'<a class="tcard {"sel" if t == selected else ""}" href="?t={t}&p={provider}" target="_self">'
            f'<div class="row"><span class="sym"><span class="dot" style="background:{dot}"></span>{t}</span>{chip(day)}</div>'
            f'<div class="row"><span class="name">{esc(c.event.company)}</span><span>{price}</span></div></a>')
    st.markdown('<div class="sect">Catalyst test cases</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="strip">{"".join(cards)}</div>', unsafe_allow_html=True)

    case, res, ev_res = results.get(selected, (None, None, None))
    event = case.event if case else None
    df = price_history(selected)

    left, right = st.columns([2.1, 1], gap="large")
    with left:
        st.markdown(f'<div class="h-name">{esc(event.company if event else selected)}</div>'
                    f'<div class="h-sub">{selected} · USD{"" if event else " · watchlist"}</div>',
                    unsafe_allow_html=True)
        rng = st.segmented_control("Range", list(RANGES), default="1Y", label_visibility="collapsed") or "1Y"
        if df.empty:
            st.warning("No price data available for this ticker.")
        else:
            start = pd.Timestamp(date(date.today().year, 1, 1)) if rng == "YTD" else df.index[-1] - pd.Timedelta(days=RANGES[rng])
            view = df.loc[start:]
            chg = (view["Close"].iloc[-1] / view["Close"].iloc[0] - 1) * 100
            st.markdown(f'<span class="h-price">&#36;{view["Close"].iloc[-1]:,.2f}</span>{chip(chg, " " + rng)}'
                        f'<div class="h-sub">Close {view.index[-1]:%b %d, %Y} · daily closes</div>', unsafe_allow_html=True)

            line = GREEN if chg >= 0 else RED
            fill = "rgba(19,115,51,0.08)" if chg >= 0 else "rgba(165,14,14,0.08)"
            fig = go.Figure(go.Scatter(x=view.index, y=view["Close"], mode="lines", line=dict(color=line, width=2),
                                       fill="tozeroy", fillcolor=fill, hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>"))
            ev_date, precision = parse_event_date(event.price_move_text) if event else (None, "none")
            if ev_date and pd.Timestamp(ev_date) >= view.index[0]:
                after = view.loc[pd.Timestamp(ev_date):]
                if not after.empty:
                    x, y = after.index[0], after["Close"].iloc[0]
                    color = BADGE[res.analysis.assessment][0] if res.analysis else GREY
                    fig.add_vline(x=x, line=dict(color=color, width=1, dash="dot"))
                    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=12, color=color, line=dict(color="white", width=2)),
                                             hovertemplate=f"Catalyst ({precision})<br>{esc(event.headline[:70])}…<extra></extra>"))
                    fig.add_annotation(x=x, y=view["Close"].max(), text="📰 Catalyst", showarrow=False, yshift=8,
                                       font=dict(color=color, size=12))
            elif ev_date:
                st.caption(f"Catalyst date {ev_date:%b %d, %Y} is outside the selected range.")
            lo, hi = view["Close"].min(), view["Close"].max()
            pad = (hi - lo) * 0.08 or hi * 0.05
            fig.update_layout(height=340, margin=dict(l=0, r=0, t=16, b=0), showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", hovermode="x unified",
                              yaxis=dict(range=[lo - pad, hi + pad], gridcolor="#f1f3f4", tickprefix="$", side="left"),
                              xaxis=dict(showgrid=False))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        rows = []
        if len(df) > 1:
            last, prev = df.iloc[-1], df.iloc[-2]
            yr = df.loc[df.index[-1] - pd.Timedelta(days=365):]
            mc = market_cap(selected)
            rows += [("Previous close", f"&#36;{prev['Close']:,.2f}"),
                     ("Day range", f"&#36;{last['Low']:,.2f} – &#36;{last['High']:,.2f}"),
                     ("Year range", f"&#36;{yr['Low'].min():,.2f} – &#36;{yr['High'].max():,.2f}"),
                     ("Market cap (today)", "—" if not mc else f"&#36;{mc / 1e12:,.2f}T" if mc >= 1e12
                      else f"&#36;{mc / 1e9:,.2f}B" if mc >= 1e9 else f"&#36;{mc / 1e6:,.1f}M"),
                     ("Volume", f"{last['Volume']:,.0f}")]
        st.markdown('<div class="card"><h4>Key stats</h4>' + "".join(
            f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in rows) + "</div>",
            unsafe_allow_html=True)

        if event:
            # Deterministic move verification: source claim vs actual closes
            ev_date, precision = parse_event_date(event.price_move_text)
            mv = verify_move(df, ev_date, res.rules.move_pct, res.rules.move_window)
            status_style = {"match": (GREEN, GREEN_BG, "✓ Matches price data"),
                            "mismatch": (RED, RED_BG, "⚠ Does not match price data"),
                            "unverifiable": (GREY, GREY_BG, "– Cannot verify")}[mv.status]
            claimed = f"{mv.claimed_pct:+.1f}%" if mv.claimed_pct is not None else "not stated"
            verified = f"{mv.verified_pct:+.1f}%" if mv.verified_pct is not None else "—"
            st.markdown(
                f'<div class="card"><h4>Move check <span class="demo">(deterministic)</span></h4>'
                f'<div class="kv"><span class="k">Claimed by source</span><span class="v">{claimed}</span></div>'
                f'<div class="kv"><span class="k">Verified from closes</span><span class="v">{verified}</span></div>'
                f'<div class="kv"><span class="k">Event date</span><span class="v">{ev_date or "—"} <span class="demo">({esc(precision)})</span></span></div>'
                f'<div style="margin-top:8px"><span class="chip" style="color:{status_style[0]};background:{status_style[1]}">{status_style[2]}</span></div>'
                f'<div class="note" style="margin-top:6px">{esc(mv.detail)}</div></div>', unsafe_allow_html=True)

    if not event:
        st.markdown(
            f'<div class="card"><h4>Catalyst analysis</h4><div style="font-size:14px">No news events for '
            f'<b>{selected}</b> in the current feed (fixture replay), so this ticker has <b>not been analyzed</b>.</div>'
            f'<div class="note" style="margin-top:6px">This is not a "NO FRESH CATALYST" verdict — that label is only '
            f'assigned after a mover is checked against live news/EDGAR sources, which arrive post-hire.</div></div>',
            unsafe_allow_html=True)
    if event:
        # News item
        st.markdown(
            f'<div class="card"><h4>In the news</h4><div class="news">{esc(event.source_publisher)} · {esc(event.source_title or "")}</div>'
            f'<a href="{esc(event.source_url or "#")}" target="_blank">{esc(event.headline)}</a>'
            f'<div class="note" style="margin-top:4px">{esc(event.price_move_text or "")}</div></div>', unsafe_allow_html=True)

        # Catalyst analysis
        st.markdown(f"#### Catalyst analysis · {PROVIDERS[provider]}")
        if res.alert_status == "NOT_RUN":
            st.info(f"No {PROVIDERS[provider]} analysis yet. Press **Run analysis** in the sidebar.")
        elif res.alert_status == "ERROR":
            st.error(f"AI provider unavailable — manual review required.\n\n{res.error}")
        else:
            a, al = res.analysis, res.alert
            c, bg, label = BADGE[a.assessment]
            grade = ""
            if ev_res:
                g = {"PASS": (GREEN, GREEN_BG), "FAIL": (RED, RED_BG)}.get(ev_res.status, (GREY, GREY_BG))
                grade = f'<span class="chip" style="float:right;color:{g[0]};background:{g[1]}">Test case: {ev_res.status}</span>'
            st.markdown(
                f'<div class="banner" style="background:{bg};border-left:4px solid {c}">{grade}'
                f'<div class="big" style="color:{c}">{label}</div>'
                f'<div class="watch">Watch: <b>{esc(al.watch)}</b> · confidence {a.confidence:.0%}</div></div>',
                unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1.25], gap="large")
            with col1:
                fields = [("Ticker", al.ticker), ("Type", al.type), ("Move", al.move), ("News age", al.news_age),
                          ("Source", al.source), ("Catalyst", al.catalyst)]
                flags = [("Financial value disclosed", al.financial_value_disclosed),
                         ("Revenue impact disclosed", al.revenue_impact_disclosed),
                         ("Binding agreement", al.binding_agreement), ("Recent dilution", al.recent_dilution)]
                body = "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{esc(v)}</span></div>' for k, v in fields)
                body += "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{yn(v)}</span></div>' for k, v in flags)
                body += (f'<div class="kv"><span class="k">Assessment</span><span class="v">{esc(al.assessment)}</span></div>'
                         f'<div class="kv"><span class="k">Watch</span><span class="v">{esc(al.watch)}</span></div>')
                st.markdown(f'<div class="card"><h4>Alert</h4>{body}</div>', unsafe_allow_html=True)
                with st.expander("Plain-text alert (as sent)"):
                    st.code(res.alert_text, language=None)

            with col2:
                ev_html = "".join(f'<div class="li" style="--c:{GREEN}">{esc(e)}</div>' for e in a.evidence)
                rk_html = "".join(f'<div class="li" style="--c:{RED}">{esc(r)}</div>' for r in a.risk_factors) or \
                    "<div class='note'>None identified</div>"
                guard = "".join(f'<div class="guard">{esc(n)}</div>' for n in res.guardrail_notes)
                notes = "".join(f'<div class="note">• {esc(n)}</div>' for n in al.notes if not n.startswith("GUARDRAIL"))
                detail = (f"Clinical/FDA: {a.clinical_fda_significance} · Promotional language: "
                          f"{'yes' if a.promotional_or_vague_language else 'no'} · Reaction disproportionate: {a.reaction_disproportionate}")
                st.markdown(
                    f'<div class="card"><h4>Why</h4><div style="font-size:14px;color:#202124">{esc(a.reasoning)}</div>'
                    f'<div class="note" style="margin:6px 0 10px 0">{esc(detail)}</div>{guard}'
                    f'<h4 style="margin-top:10px">Evidence</h4>{ev_html}<h4 style="margin-top:12px">Risk factors</h4>{rk_html}'
                    f'<h4 style="margin-top:12px">Data caveats</h4>{notes}</div>', unsafe_allow_html=True)
            if ev_res:
                st.caption(f"Expected: {case.expected_classification}  —  " +
                           "; ".join(f"{'✓' if ch.passed else '✗'} {ch.description} (actual {ch.actual})" for ch in ev_res.checks))

# ---------------- scorecard ----------------
with tab_score:
    all_data = {p: load_results(p) for p in PROVIDERS}
    rows = []
    for t in tickers:
        case = results[t][0]
        row = {"Case": case.case_id, "Ticker": t, "Expected": case.expected_classification.split(" - ")[0][:40]}
        for p, label in PROVIDERS.items():
            _, r, ev = all_data[p]["results"][t]
            short = label.split(" (")[0]
            row[f"{short}"] = r.analysis.assessment.value.replace("_", " ") if r.analysis else "not run"
            row[f"{short} result"] = ev.status if ev else "—"
        rows.append(row)
    score = pd.DataFrame(rows)
    *pcols, m3 = st.columns(len(PROVIDERS) + 1)
    for col, (p, label) in zip(pcols, PROVIDERS.items()):
        short = label.split(" (")[0]
        n = (score[f"{short} result"] == "PASS").sum()
        ran = (score[f"{short} result"] != "—").sum()
        pending = len(score) - ran
        col.metric(label, f"{n}/{ran} pass" if ran else "not run",
                   delta=f"{pending} not run yet" if pending and ran else None, delta_color="off",
                   help=f"{ran} of {len(score)} cases analyzed")
    m3.metric("Pending test cases", len(data["skipped"]), help="Null-data fixtures awaiting real sourced examples")

    def color(v):
        return {"PASS": f"color:{GREEN};background:{GREEN_BG}", "FAIL": f"color:{RED};background:{RED_BG}"}.get(v, "")
    st.dataframe(score.style.map(color), width="stretch", hide_index=True)
    st.caption("Same pipeline, same deterministic rules and guardrails — only the AI provider differs.")
