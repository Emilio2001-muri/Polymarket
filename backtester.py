"""
PolyBot Multi‑Market Backtester
================================
Simulates the MACD + CVD hybrid strategy across multiple synthetic
or historical price paths.  Claude AI calls are optional (for cost
control during heavy backtesting).

Usage:
    python backtester.py            # run with defaults
    python backtester.py --markets 10 --days 60 --with-claude
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtester")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DATA GENERATION                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def generate_price_path(
    days: int = 60,
    interval_minutes: int = 30,
    start_price: float = 0.50,
    volatility: float = 0.015,
    drift: float = 0.0001,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic prediction‑market price path bounded [0.01, 0.99].
    Uses geometric Brownian motion with mean‑reversion toward 0.50.
    """
    rng = np.random.default_rng(seed)
    steps = int(days * 24 * 60 / interval_minutes)
    prices = np.empty(steps)
    prices[0] = start_price

    for i in range(1, steps):
        revert = 0.001 * (0.50 - prices[i - 1])  # gentle pull to 0.50
        shock = rng.normal(0, volatility)
        prices[i] = prices[i - 1] + drift + revert + shock
        prices[i] = np.clip(prices[i], 0.01, 0.99)

    timestamps = pd.date_range(
        start=datetime.now(timezone.utc) - timedelta(days=days),
        periods=steps,
        freq=f"{interval_minutes}min",
    )

    # Synthetic volume (correlated with absolute returns)
    returns = np.abs(np.diff(prices, prepend=prices[0]))
    volume = 100 + returns * 5000 + rng.uniform(0, 50, steps)

    return pd.DataFrame({"timestamp": timestamps, "close": prices, "volume": volume})


def generate_trade_flow(df: pd.DataFrame, seed: int | None = None) -> List[dict]:
    """Generate synthetic trade‑level data for CVD calculation."""
    rng = np.random.default_rng(seed)
    trades = []
    for _, row in df.iterrows():
        n_trades = rng.integers(3, 15)
        for _ in range(n_trades):
            side = "BUY" if rng.random() > 0.48 else "SELL"  # slight buy bias
            trades.append({"size": float(rng.uniform(1, 20)), "side": side})
    return trades


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STRATEGY SIGNALS  (mirrors bot_core.SignalEngine)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def macd_signal(prices: List[float]) -> Tuple[str, float]:
    needed = config.MACD_SLOW + config.MACD_SIGNAL + 1
    if len(prices) < needed:
        return "NEUTRAL", 0.0

    df = pd.DataFrame({"close": prices})
    macd_df = df.ta.macd(fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)
    if macd_df is None or macd_df.empty:
        return "NEUTRAL", 0.0

    hist_col = [c for c in macd_df.columns if "MACDh" in c or "Hist" in c]
    if not hist_col:
        hist_col = [macd_df.columns[1]]
    hist = macd_df[hist_col[0]].dropna()
    if len(hist) < 2:
        return "NEUTRAL", 0.0

    cur, prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    if cur > 0 and prev <= 0:
        return "BUY", min(abs(cur) * 10, 1.0)
    if cur < 0 and prev >= 0:
        return "SELL", min(abs(cur) * 10, 1.0)
    if cur > 0:
        return "BUY", min(abs(cur) * 5, 0.6)
    if cur < 0:
        return "SELL", min(abs(cur) * 5, 0.6)
    return "NEUTRAL", 0.0


def cvd_signal(trades: List[dict]) -> Tuple[str, float]:
    if len(trades) < 10:
        return "NEUTRAL", 0.0
    buy_v = sum(t["size"] for t in trades if t["side"] == "BUY")
    sell_v = sum(t["size"] for t in trades if t["side"] == "SELL")
    total = buy_v + sell_v
    if total == 0:
        return "NEUTRAL", 0.0
    cvd = (buy_v - sell_v) / total
    if cvd > config.CVD_THRESHOLD:
        return "BUY", cvd
    if cvd < -config.CVD_THRESHOLD:
        return "SELL", abs(cvd)
    return "NEUTRAL", abs(cvd)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BACKTEST ENGINE                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class BacktestTrade:
    entry_time: str
    exit_time: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    market_id: int = 0


@dataclass
class BacktestResult:
    market_id: int
    market_name: str
    trades: List[BacktestTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0


class BacktestEngine:
    def __init__(
        self,
        num_markets: int = 10,
        days: int = 60,
        order_size: float = 5.0,
        use_claude: bool = False,
    ):
        self.num_markets = num_markets
        self.days = days
        self.order_size = order_size
        self.use_claude = use_claude
        self.results: List[BacktestResult] = []
        self.claude_advisor = None

        if use_claude and config.ANTHROPIC_API_KEY:
            from bot_core import ClaudeAdvisor
            self.claude_advisor = ClaudeAdvisor()

    def run(self) -> List[BacktestResult]:
        logger.info(
            "Starting backtest: %d markets × %d days | Claude=%s",
            self.num_markets, self.days, self.use_claude,
        )
        self.results = []

        for mid in range(self.num_markets):
            seed = 42 + mid
            start_price = np.random.default_rng(seed).uniform(0.2, 0.8)
            vol = np.random.default_rng(seed).uniform(0.008, 0.025)

            df = generate_price_path(
                days=self.days,
                start_price=start_price,
                volatility=vol,
                seed=seed,
            )
            trades_flow = generate_trade_flow(df, seed=seed)
            result = self._backtest_market(mid, df, trades_flow)
            self.results.append(result)
            logger.info(
                "  Market %d: %d trades, P&L $%.2f, WR %.1f%%",
                mid, result.num_trades, result.total_pnl, result.win_rate,
            )

        total_pnl = sum(r.total_pnl for r in self.results)
        total_trades = sum(r.num_trades for r in self.results)
        avg_wr = np.mean([r.win_rate for r in self.results if r.num_trades > 0]) if self.results else 0

        logger.info("═" * 50)
        logger.info("TOTAL  P&L: $%.2f  |  Trades: %d  |  Avg WR: %.1f%%", total_pnl, total_trades, avg_wr)
        logger.info("═" * 50)

        return self.results

    def _backtest_market(
        self, mid: int, df: pd.DataFrame, trade_flow: List[dict]
    ) -> BacktestResult:
        result = BacktestResult(market_id=mid, market_name=f"Synthetic Market #{mid}")
        prices = df["close"].tolist()
        timestamps = df["timestamp"].tolist()

        position_side = ""
        position_entry = 0.0
        position_size = 0.0
        open_trades: List[BacktestTrade] = []

        window = config.MACD_SLOW + config.MACD_SIGNAL + 5

        for i in range(window, len(prices)):
            price_window = prices[: i + 1]
            current_price = prices[i]

            # Signals
            m_sig, m_str = macd_signal(price_window)
            # Use a sliding window of trade flow
            flow_start = max(0, i * 5 - 50)
            flow_end = min(len(trade_flow), i * 5 + 5)
            c_sig, c_val = cvd_signal(trade_flow[flow_start:flow_end])

            if m_sig == "NEUTRAL":
                continue

            agree = (m_sig == c_sig) or (c_sig == "NEUTRAL")
            if not agree:
                continue

            # Claude gate (optional)
            if self.claude_advisor:
                confirmed, conf, _ = self.claude_advisor.confirm_signal(
                    market_question=f"Synthetic Market #{mid}",
                    current_price=current_price,
                    macd_signal=m_sig,
                    macd_strength=m_str,
                    cvd_signal=c_sig,
                    cvd_value=c_val,
                    recent_prices=price_window[-15:],
                )
                if not confirmed:
                    continue

            # If we have a position in opposite direction → close
            if position_side and (
                (position_side == "YES" and m_sig == "SELL")
                or (position_side == "NO" and m_sig == "BUY")
            ):
                pnl = 0.0
                if position_side == "YES":
                    pnl = (current_price - position_entry) * position_size
                else:
                    pnl = (position_entry - current_price) * position_size

                for ot in open_trades:
                    ot.exit_time = str(timestamps[i])
                    ot.exit_price = current_price
                    ot.pnl = pnl / max(len(open_trades), 1)
                    result.trades.append(ot)

                open_trades = []
                position_side = ""
                position_size = 0.0

            # Open new position
            if not position_side:
                position_side = "YES" if m_sig == "BUY" else "NO"
                position_entry = current_price
                position_size = self.order_size

                open_trades.append(
                    BacktestTrade(
                        entry_time=str(timestamps[i]),
                        side=m_sig,
                        entry_price=current_price,
                        size=self.order_size,
                        market_id=mid,
                    )
                )

        # Close open positions at last price
        if position_side and open_trades:
            last_price = prices[-1]
            if position_side == "YES":
                pnl = (last_price - position_entry) * position_size
            else:
                pnl = (position_entry - last_price) * position_size
            for ot in open_trades:
                ot.exit_time = str(timestamps[-1])
                ot.exit_price = last_price
                ot.pnl = pnl / max(len(open_trades), 1)
                result.trades.append(ot)

        # Metrics
        result.num_trades = len(result.trades)
        result.total_pnl = round(sum(t.pnl for t in result.trades), 4)

        if result.num_trades > 0:
            wins = sum(1 for t in result.trades if t.pnl > 0)
            result.win_rate = round(wins / result.num_trades * 100, 1)

            # Max drawdown
            cum_pnl = np.cumsum([t.pnl for t in result.trades])
            running_max = np.maximum.accumulate(cum_pnl)
            drawdowns = running_max - cum_pnl
            result.max_drawdown = round(float(np.max(drawdowns)) if len(drawdowns) > 0 else 0, 4)

            # Sharpe (annualised, assuming 30‑min intervals)
            returns = [t.pnl / self.order_size for t in result.trades]
            if np.std(returns) > 0:
                intervals_per_year = 365 * 24 * 2
                result.sharpe = round(
                    (np.mean(returns) / np.std(returns)) * np.sqrt(intervals_per_year), 2
                )

        return result


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  VISUALISATION                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_backtest(results: List[BacktestResult]) -> go.Figure:
    """Create a comprehensive backtest results figure."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Cumulative P&L (all markets)",
            "P&L by Market",
            "Win Rate Distribution",
            "Trade Count by Market",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # 1) Cumulative P&L
    all_trades = []
    for r in results:
        for t in r.trades:
            all_trades.append({"time": t.entry_time, "pnl": t.pnl, "market": r.market_name})

    if all_trades:
        all_df = pd.DataFrame(all_trades).sort_values("time")
        all_df["cum_pnl"] = all_df["pnl"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(all_df))),
                y=all_df["cum_pnl"],
                mode="lines",
                line=dict(width=2, color="#a855f7"),
                fill="tozeroy",
                fillcolor="rgba(168,85,247,0.12)",
                name="Cum P&L",
            ),
            row=1, col=1,
        )

    # 2) P&L by market  (bar chart)
    names = [r.market_name for r in results]
    pnls = [r.total_pnl for r in results]
    colors = ["#22ff88" if p >= 0 else "#ff3366" for p in pnls]
    fig.add_trace(
        go.Bar(x=names, y=pnls, marker_color=colors, name="Market P&L"),
        row=1, col=2,
    )

    # 3) Win rate histogram
    win_rates = [r.win_rate for r in results if r.num_trades > 0]
    fig.add_trace(
        go.Histogram(x=win_rates, nbinsx=10, marker_color="#7c4dff", name="Win Rate"),
        row=2, col=1,
    )

    # 4) Trade counts
    counts = [r.num_trades for r in results]
    fig.add_trace(
        go.Bar(x=names, y=counts, marker_color="#448aff", name="Trades"),
        row=2, col=2,
    )

    fig.update_layout(
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8e8ff", family="Inter, sans-serif"),
        showlegend=False,
        title=dict(text="Backtest Results — Multi‑Market MACD + CVD Strategy", font=dict(size=18)),
    )
    for ax in ["xaxis", "xaxis2", "xaxis3", "xaxis4", "yaxis", "yaxis2", "yaxis3", "yaxis4"]:
        fig.update_layout(
            **{ax: dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")}
        )

    return fig


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="PolyBot Multi‑Market Backtester")
    parser.add_argument("--markets", type=int, default=10, help="Number of synthetic markets (default 10)")
    parser.add_argument("--days", type=int, default=60, help="Days of history (default 60)")
    parser.add_argument("--size", type=float, default=5.0, help="Order size in USDC (default 5)")
    parser.add_argument("--with-claude", action="store_true", help="Use Claude AI for confirmation")
    args = parser.parse_args()

    engine = BacktestEngine(
        num_markets=args.markets,
        days=args.days,
        order_size=args.size,
        use_claude=args.with_claude,
    )
    results = engine.run()

    # Save results
    summary = []
    for r in results:
        summary.append({
            "market": r.market_name,
            "trades": r.num_trades,
            "pnl": r.total_pnl,
            "win_rate": r.win_rate,
            "max_dd": r.max_drawdown,
            "sharpe": r.sharpe,
        })
    with open("backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to backtest_results.json")

    # Plot
    fig = plot_backtest(results)
    fig.write_html("backtest_results.html")
    logger.info("Chart saved to backtest_results.html — open in browser")


if __name__ == "__main__":
    main()
