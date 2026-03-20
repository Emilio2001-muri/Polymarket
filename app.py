"""
PolyBot Dashboard — Streamlit App
==================================
Premium trading dashboard with 4 themes (Light / Dark / System / SPLENDID),
live P&L curve, sortable trades table, per‑market charts, Claude AI logs,
and full mobile‑responsive layout.

Run:  streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from bot_core import get_bot, get_state, load_state, add_log, save_state, BotState
import config

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="PolyBot — Quantitative Trader",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CSS THEMES                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

THEMES = {
    "Light": """
    <style>
        :root {
            --bg: #f5f7fa; --card: #ffffff; --text: #1a1a2e;
            --accent: #2962ff; --positive: #00c853; --negative: #ff1744;
            --border: #e0e0e0; --muted: #757575;
        }
        .stApp { background: var(--bg) !important; }
        .metric-card {
            background: var(--card); border: 1px solid var(--border);
            border-radius: 16px; padding: 1.2rem 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            text-align: center; transition: transform .2s;
        }
        .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
        .metric-label { font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: var(--text); margin: .3rem 0; }
        .metric-value.positive { color: var(--positive); }
        .metric-value.negative { color: var(--negative); }
        .header-bar {
            background: linear-gradient(135deg, #2962ff 0%, #448aff 100%);
            color: white; padding: 1.2rem 2rem; border-radius: 16px;
            margin-bottom: 1.5rem; display: flex; align-items: center;
            justify-content: space-between; flex-wrap: wrap;
        }
        .header-bar h1 { margin: 0; font-size: 1.6rem; }
        .status-badge {
            padding: .3rem .8rem; border-radius: 20px; font-size: .75rem;
            font-weight: 600; display: inline-block;
        }
        .badge-live { background: #00c853; color: white; }
        .badge-paper { background: #ff9800; color: white; }
        .badge-stopped { background: #9e9e9e; color: white; }
        div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
        @media (max-width: 768px) {
            .metric-value { font-size: 1.3rem; }
            .header-bar { padding: .8rem 1rem; }
            .header-bar h1 { font-size: 1.1rem; }
        }
    </style>
    """,

    "Dark": """
    <style>
        :root {
            --bg: #0e1117; --card: #1a1a2e; --text: #e8e8ff;
            --accent: #7c4dff; --positive: #00e676; --negative: #ff1744;
            --border: #2a2a3e; --muted: #9e9eb3;
        }
        .stApp { background: var(--bg) !important; color: var(--text) !important; }
        .metric-card {
            background: var(--card); border: 1px solid var(--border);
            border-radius: 16px; padding: 1.2rem 1.5rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            text-align: center; transition: transform .2s;
        }
        .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(124,77,255,0.15); }
        .metric-label { font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: var(--text); margin: .3rem 0; }
        .metric-value.positive { color: var(--positive); }
        .metric-value.negative { color: var(--negative); }
        .header-bar {
            background: linear-gradient(135deg, #7c4dff 0%, #b47cff 100%);
            color: white; padding: 1.2rem 2rem; border-radius: 16px;
            margin-bottom: 1.5rem; display: flex; align-items: center;
            justify-content: space-between; flex-wrap: wrap;
        }
        .header-bar h1 { margin: 0; font-size: 1.6rem; }
        .status-badge { padding: .3rem .8rem; border-radius: 20px; font-size: .75rem; font-weight: 600; display: inline-block; }
        .badge-live { background: #00e676; color: #0e1117; }
        .badge-paper { background: #ff9800; color: #0e1117; }
        .badge-stopped { background: #555; color: #ccc; }
        @media (max-width: 768px) {
            .metric-value { font-size: 1.3rem; }
        }
    </style>
    """,

    "System": """
    <style>
        @media (prefers-color-scheme: dark) {
            :root { --bg:#0e1117;--card:#1a1a2e;--text:#e8e8ff;--accent:#7c4dff;--positive:#00e676;--negative:#ff1744;--border:#2a2a3e;--muted:#9e9eb3; }
        }
        @media (prefers-color-scheme: light) {
            :root { --bg:#f5f7fa;--card:#fff;--text:#1a1a2e;--accent:#2962ff;--positive:#00c853;--negative:#ff1744;--border:#e0e0e0;--muted:#757575; }
        }
        .stApp { background: var(--bg) !important; }
        .metric-card {
            background: var(--card); border: 1px solid var(--border);
            border-radius: 16px; padding: 1.2rem 1.5rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1); text-align: center;
        }
        .metric-label { font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px; }
        .metric-value { font-size:1.8rem;font-weight:700;color:var(--text);margin:.3rem 0; }
        .metric-value.positive { color: var(--positive); }
        .metric-value.negative { color: var(--negative); }
        .header-bar {
            background: linear-gradient(135deg, var(--accent), #448aff);
            color: white; padding: 1.2rem 2rem; border-radius: 16px;
            margin-bottom: 1.5rem; display: flex; align-items: center;
            justify-content: space-between; flex-wrap: wrap;
        }
        .header-bar h1 { margin:0;font-size:1.6rem; }
        .status-badge { padding:.3rem .8rem;border-radius:20px;font-size:.75rem;font-weight:600;display:inline-block; }
        .badge-live { background:#00c853;color:white; }
        .badge-paper { background:#ff9800;color:white; }
        .badge-stopped { background:#9e9e9e;color:white; }
    </style>
    """,

    "SPLENDID": """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        :root {
            --bg-start: #070b16; --bg-mid: #0f1628; --bg-end: #0a0f1e;
            --card: rgba(255,255,255,0.04); --card-hover: rgba(255,255,255,0.07);
            --text: #eef0ff; --text-muted: #8b8fa8;
            --gold: #ffd700; --accent: #a855f7; --accent2: #ec4899;
            --positive: #22ff88; --negative: #ff3366;
            --border: rgba(168,85,247,0.2); --glow: rgba(168,85,247,0.15);
        }

        .stApp {
            background: linear-gradient(160deg, var(--bg-start) 0%, var(--bg-mid) 50%, var(--bg-end) 100%) !important;
            color: var(--text) !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-start); }
        ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }

        .metric-card {
            background: var(--card);
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 20px; padding: 1.4rem 1.6rem;
            text-align: center; position: relative; overflow: hidden;
            transition: all .3s cubic-bezier(.4,0,.2,1);
        }
        .metric-card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, var(--accent), var(--accent2), var(--gold));
        }
        .metric-card:hover {
            background: var(--card-hover);
            transform: translateY(-4px);
            box-shadow: 0 12px 40px var(--glow);
            border-color: var(--accent);
        }
        .metric-label {
            font-size: .72rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;
        }
        .metric-value {
            font-size: 2rem; font-weight: 800; margin: .4rem 0;
            background: linear-gradient(135deg, var(--gold), #ffed4e);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .metric-value.positive {
            background: linear-gradient(135deg, var(--positive), #44ffaa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .metric-value.negative {
            background: linear-gradient(135deg, var(--negative), #ff6688);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-bar {
            background: linear-gradient(135deg, rgba(168,85,247,0.25), rgba(236,72,153,0.2));
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            color: white; padding: 1.4rem 2.2rem; border-radius: 20px;
            margin-bottom: 1.5rem; display: flex; align-items: center;
            justify-content: space-between; flex-wrap: wrap;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .header-bar h1 {
            margin: 0; font-size: 1.6rem; font-weight: 800;
            background: linear-gradient(90deg, var(--gold), var(--accent2), var(--accent));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .status-badge {
            padding: .35rem 1rem; border-radius: 20px; font-size: .75rem;
            font-weight: 700; display: inline-block; letter-spacing: .5px;
        }
        .badge-live {
            background: linear-gradient(135deg, var(--positive), #00cc66);
            color: #070b16; box-shadow: 0 0 16px rgba(34,255,136,0.3);
        }
        .badge-paper {
            background: linear-gradient(135deg, #ff9800, #ffb74d);
            color: #070b16; box-shadow: 0 0 16px rgba(255,152,0,0.3);
        }
        .badge-stopped { background: rgba(255,255,255,0.1); color: var(--text-muted); }

        /* Glass table */
        div[data-testid="stDataFrame"] {
            border-radius: 16px; overflow: hidden;
            border: 1px solid var(--border);
        }

        /* Sidebar glass */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,22,40,0.95), rgba(7,11,22,0.98)) !important;
            border-right: 1px solid var(--border) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--card); border-radius: 12px; padding: 4px;
            border: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px; font-weight: 600; color: var(--text-muted);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
            color: white !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 12px; font-weight: 700; letter-spacing: .5px;
            border: 1px solid var(--border); transition: all .3s;
        }
        .stButton > button:hover {
            border-color: var(--accent);
            box-shadow: 0 0 20px var(--glow);
        }

        @media (max-width: 768px) {
            .metric-value { font-size: 1.4rem; }
            .header-bar { padding: .8rem 1rem; }
            .header-bar h1 { font-size: 1.1rem; }
            .metric-card { padding: .8rem 1rem; }
        }
    </style>
    """,
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def pnl_class(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return ""


def metric_card(label: str, value: str, css_class: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {css_class}">{value}</div>'
        f"</div>"
    )


def plotly_layout(title: str = "", dark: bool = True) -> dict:
    bg = "rgba(0,0,0,0)" if dark else "#ffffff"
    text_color = "#e8e8ff" if dark else "#1a1a2e"
    grid = "rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.06)"
    return dict(
        title=dict(text=title, font=dict(size=16, color=text_color)),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=text_color, family="Inter, sans-serif"),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## 🤖 PolyBot")
    st.caption("Hybrid Quant Trader for Polymarket")
    st.divider()

    theme = st.selectbox("🎨 Theme", list(THEMES.keys()), index=3)
    is_dark = theme in ("Dark", "SPLENDID", "System")

    st.divider()
    st.markdown("### ⚙️ Controls")

    # Bot instance (persisted in session_state)
    if "bot" not in st.session_state:
        st.session_state.bot = get_bot()
        load_state()

    bot = st.session_state.bot
    state = get_state()

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶ Start", width="stretch", type="primary"):
            if not bot.running:
                bot.start()
                st.toast("Bot started!", icon="🚀")
    with col_stop:
        if st.button("⏹ Stop", width="stretch"):
            if bot.running:
                bot.stop()
                st.toast("Bot stopped", icon="🛑")

    st.divider()

    paper = st.toggle("📝 Paper Trading", value=config.PAPER_TRADING, key="paper_toggle")
    config.PAPER_TRADING = paper
    state.paper_mode = paper
    if not paper and bot.clob is None:
        state.paper_mode = True
        st.warning("⚠️ No CLOB connection — forced paper")
    elif not paper:
        st.success("🔴 LIVE MODE — real orders enabled")

    config.ORDER_SIZE_USDC = st.slider("💰 Order Size ($)", 1.0, 50.0, config.ORDER_SIZE_USDC, 1.0)
    config.MAX_CONCURRENT_MARKETS = st.slider("📊 Max Markets", 5, 30, config.MAX_CONCURRENT_MARKETS)
    config.SCAN_INTERVAL_SECONDS = st.slider("⏱ Scan Interval (s)", 10, 120, config.SCAN_INTERVAL_SECONDS)
    config.CLAUDE_CONFIDENCE_MIN = st.slider("🧠 Claude Min Conf", 0.3, 0.95, config.CLAUDE_CONFIDENCE_MIN, 0.05)

    st.divider()
    auto_refresh = st.toggle("🔄 Auto‑refresh (5s)", value=True)

    st.divider()
    st.caption("MACD / CVD / Claude · Liquidity Farming")
    st.caption(f"v1.0 · {len(state.markets)} markets tracked")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  INJECT THEME CSS                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

st.markdown(THEMES[theme], unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HEADER BAR                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if state.running and not state.paper_mode:
    badge = '<span class="status-badge badge-live">● LIVE</span>'
elif state.running and state.paper_mode:
    badge = '<span class="status-badge badge-paper">● PAPER</span>'
else:
    badge = '<span class="status-badge badge-stopped">● STOPPED</span>'

st.markdown(
    f'<div class="header-bar">'
    f'<h1>🤖 PolyBot Dashboard</h1>'
    f'<div>{badge}</div>'
    f"</div>",
    unsafe_allow_html=True,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  METRICS ROW                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    bal_cls = pnl_class(state.balance - state.initial_balance) if state.initial_balance > 0 else ""
    bal_label = "Balance (LIVE)" if not state.paper_mode else "Balance (PAPER)"
    st.markdown(metric_card(bal_label, f"${state.balance:,.2f}", bal_cls), unsafe_allow_html=True)
with c2:
    pnl_val = state.total_pnl
    sign = "+" if pnl_val >= 0 else ""
    st.markdown(
        metric_card("Total P&L", f"{sign}${pnl_val:,.2f}", pnl_class(pnl_val)),
        unsafe_allow_html=True,
    )
with c3:
    wr = (state.winning_trades / state.total_trades * 100) if state.total_trades > 0 else 0
    st.markdown(metric_card("Win Rate", f"{wr:.1f}%", "positive" if wr > 50 else ""), unsafe_allow_html=True)
with c4:
    total_exp = sum(m.get("position_size", 0) for m in state.markets.values())
    st.markdown(metric_card("Exposure", f"${total_exp:,.2f}"), unsafe_allow_html=True)
with c5:
    st.markdown(metric_card("Markets", str(len(state.markets))), unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  P&L CURVE                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if state.pnl_history:
    pnl_df = pd.DataFrame(state.pnl_history)
    pnl_df["timestamp"] = pd.to_datetime(pnl_df["timestamp"])

    fig_pnl = go.Figure()

    # Separate positive / negative fill
    fig_pnl.add_trace(
        go.Scatter(
            x=pnl_df["timestamp"],
            y=pnl_df["pnl"],
            mode="lines",
            line=dict(width=2.5, color="#a855f7" if is_dark else "#2962ff"),
            fill="tozeroy",
            fillcolor="rgba(168,85,247,0.15)" if is_dark else "rgba(41,98,255,0.1)",
            name="P&L",
            hovertemplate="%{y:$.2f}<extra></extra>",
        )
    )
    fig_pnl.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)" if is_dark else "rgba(0,0,0,0.15)")
    fig_pnl.update_layout(**plotly_layout("Cumulative P&L", is_dark), height=380)
    st.plotly_chart(fig_pnl, width="stretch")
else:
    st.info("P&L chart will appear once the bot executes trades.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TABS                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

tab_trades, tab_charts, tab_claude, tab_logs = st.tabs(
    ["📈 Live Trades", "📊 Market Charts", "🧠 Claude Analysis", "📋 Logs"]
)


# ── TAB 1: Live Trades ──────────────────────────────────────────────────────

with tab_trades:
    if state.trades:
        trades_df = pd.DataFrame(state.trades)
        trades_df = trades_df.sort_values("timestamp", ascending=False).head(100)

        display_cols = ["timestamp", "market", "side", "price", "size", "confidence", "mode", "source"]
        available_cols = [c for c in display_cols if c in trades_df.columns]
        trades_df = trades_df[available_cols]

        # Format
        if "price" in trades_df.columns:
            trades_df["price"] = trades_df["price"].map(lambda x: f"${x:.4f}")
        if "size" in trades_df.columns:
            trades_df["size"] = trades_df["size"].map(lambda x: f"${x:.2f}")
        if "confidence" in trades_df.columns:
            trades_df["confidence"] = trades_df["confidence"].map(lambda x: f"{x:.0%}")
        if "timestamp" in trades_df.columns:
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"]).dt.strftime("%m/%d %H:%M:%S")

        st.dataframe(trades_df, width="stretch", hide_index=True, height=400)
        st.caption(f"Showing last {len(trades_df)} of {state.total_trades} total trades")
    else:
        st.info("No trades yet. Start the bot and wait for signals.")


# ── TAB 2: Market Charts ────────────────────────────────────────────────────

with tab_charts:
    market_items = list(state.markets.items())

    if market_items:
        selected_cid = st.selectbox(
            "Select Market",
            [cid for cid, _ in market_items],
            format_func=lambda cid: state.markets[cid].get("question", cid)[:70],
        )
        ms = state.markets.get(selected_cid, {})

        if ms and ms.get("prices"):
            prices = ms["prices"]
            timestamps = ms.get("timestamps", list(range(len(prices))))
            ts_dt = pd.to_datetime(timestamps, errors="coerce")

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.65, 0.35],
                vertical_spacing=0.08,
                subplot_titles=("Price (YES probability)", "MACD"),
            )

            # Price line
            fig.add_trace(
                go.Scatter(
                    x=ts_dt, y=prices,
                    mode="lines",
                    line=dict(width=2, color="#a855f7" if is_dark else "#2962ff"),
                    fill="tozeroy",
                    fillcolor="rgba(168,85,247,0.08)" if is_dark else "rgba(41,98,255,0.06)",
                    name="YES Price",
                ),
                row=1, col=1,
            )

            # Entry marker
            if ms.get("entry_price") and ms["position_size"] > 0:
                fig.add_hline(
                    y=ms["entry_price"], line_dash="dash", line_color="#ffd700",
                    annotation_text=f"Entry {ms['entry_price']:.4f}", row=1, col=1,
                )

            # MACD subplot (manual EWM — no pandas_ta)
            if len(prices) >= config.MACD_SLOW + config.MACD_SIGNAL:
                s = pd.Series(prices)
                ema_fast = s.ewm(span=config.MACD_FAST, adjust=False).mean()
                ema_slow = s.ewm(span=config.MACD_SLOW, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
                hist_vals = macd_line - signal_line

                macd_ts = ts_dt[-len(macd_line):]

                fig.add_trace(
                    go.Scatter(x=macd_ts, y=macd_line, name="MACD",
                               line=dict(color="#a855f7", width=1.5)),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=macd_ts, y=signal_line, name="Signal",
                               line=dict(color="#ec4899", width=1.5)),
                    row=2, col=1,
                )
                colors = ["#22ff88" if v >= 0 else "#ff3366" for v in hist_vals.fillna(0)]
                fig.add_trace(
                    go.Bar(x=macd_ts, y=hist_vals, name="Histogram",
                           marker_color=colors, opacity=0.6),
                    row=2, col=1,
                    )

            fig.update_layout(**plotly_layout("", is_dark), height=520, showlegend=True)
            st.plotly_chart(fig, width="stretch")

            # Position info
            info_cols = st.columns(5)
            info_cols[0].metric("Price", f"{ms['current_price']:.4f}")
            info_cols[1].metric("Position", f"{ms['position_side'] or '—'} ${ms['position_size']:.2f}")
            info_cols[2].metric("Unrealized", f"${ms.get('unrealized_pnl', 0):.4f}")
            info_cols[3].metric("MACD Str", f"{ms.get('macd_value', 0):.3f}")
            info_cols[4].metric("Signal", ms.get("last_signal", "—"))

        else:
            st.info("Collecting price data… chart will appear soon.")

        # All markets overview
        with st.expander("📋 All Markets Overview", expanded=False):
            overview = []
            for cid, m in market_items:
                overview.append({
                    "Market": m.get("question", "")[:50],
                    "Price": f"{m.get('current_price', 0):.4f}",
                    "Position": m.get("position_side", "—"),
                    "Size": f"${m.get('position_size', 0):.2f}",
                    "P&L": f"${m.get('unrealized_pnl', 0):.4f}",
                    "Signal": m.get("last_signal", "—"),
                    "Points": len(m.get("prices", [])),
                })
            st.dataframe(pd.DataFrame(overview), width="stretch", hide_index=True)
    else:
        st.info("No markets tracked yet. Start the bot to begin scanning.")


# ── TAB 3: Claude Analysis ──────────────────────────────────────────────────

with tab_claude:
    st.markdown("### 🧠 Claude AI Signal Confirmations")
    if state.claude_analyses:
        analyses = list(reversed(state.claude_analyses[-50:]))

        for a in analyses:
            conf = a.get("confidence", 0)
            color = "#22ff88" if conf >= 0.7 else "#ff9800" if conf >= 0.5 else "#ff3366"
            icon = "✅" if conf >= config.CLAUDE_CONFIDENCE_MIN else "❌"
            action = a.get('action', 'HOLD')
            market = a.get('market', '')
            reasoning = a.get('reasoning', '')
            signal_in = a.get('signal_in', '—')
            ts = a.get('timestamp', '')[:19]

            with st.container():
                st.markdown(
                    f"### {icon} {action} — "
                    f"<span style='color:{color};font-size:1.3rem;font-weight:700'>{conf:.0%}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Market:** {market}")
                st.markdown(f"**Input signal:** {signal_in} → **Claude says:** {reasoning}")
                st.caption(f"{ts}")
                st.divider()

        st.caption(f"Showing last {len(analyses)} analyses")
    else:
        st.info("No Claude analyses yet. Signals pending…")


# ── TAB 4: Logs ─────────────────────────────────────────────────────────────

with tab_logs:
    if state.logs:
        log_df = pd.DataFrame(list(reversed(state.logs[-200:])))
        log_df.columns = ["Timestamp", "Level", "Message"]
        log_df["Timestamp"] = pd.to_datetime(log_df["Timestamp"]).dt.strftime("%m/%d %H:%M:%S")
        st.dataframe(log_df, width="stretch", hide_index=True, height=500)
    else:
        st.info("Logs will appear when the bot starts.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  AUTO‑REFRESH                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if auto_refresh:
    time.sleep(5)
    st.rerun()
