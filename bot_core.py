"""
PolyBot Core — Claude-First Quantitative Trading Engine for Polymarket
=======================================================================
Strategy (inspired by the viral $68→$1.6M approach):
    1. Claude AI  →  PRIMARY: analyzes market question + context to find
       mispriced probabilities (the real edge on prediction markets)
    2. Momentum   →  Gamma API 1h/1d/1w price changes for confirmation
    3. MACD       →  Optional tertiary signal on accumulated price data

The key insight: prediction markets are about INFORMATION, not charts.
Claude evaluates whether the current price reflects the true probability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd

import config

# ── Robust HTTP session with retry ──────────────────────────────────────────
_http_session = requests.Session()
_retry_strategy = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_http_session.mount("https://", _adapter)
_http_session.headers.update({
    "User-Agent": "PolyBot/2026",
    "Accept": "application/json"
})

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("polybot")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SHARED STATE  (thread‑safe, JSON‑serialisable)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class BotState:
    running: bool = False
    paper_mode: bool = True
    balance: float = 0.0
    initial_balance: float = 0.0
    real_balance: float = 0.0          # live wallet balance
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    pnl_history: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    markets: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)
    claude_analyses: list = field(default_factory=list)
    last_scan: str = ""
    uptime_start: str = ""
    cooldowns: dict = field(default_factory=dict)  # cid -> iso timestamp


_state_lock = threading.Lock()
_bot_state = BotState()


def get_state() -> BotState:
    return _bot_state


def save_state():
    with _state_lock:
        try:
            data = asdict(_bot_state)
            config.STATE_FILE.write_text(json.dumps(data, default=str), encoding="utf-8")
        except Exception as exc:
            logger.error("save_state failed: %s", exc)


def load_state() -> BotState:
    global _bot_state
    if config.STATE_FILE.exists():
        try:
            raw = json.loads(config.STATE_FILE.read_text(encoding="utf-8"))
            valid = {k: v for k, v in raw.items() if k in BotState.__dataclass_fields__}
            _bot_state = BotState(**valid)
        except Exception:
            _bot_state = BotState()
    return _bot_state


def add_log(message: str, level: str = "INFO"):
    with _state_lock:
        _bot_state.logs.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "message": message,
            }
        )
        if len(_bot_state.logs) > 500:
            _bot_state.logs = _bot_state.logs[-500:]
    logger.log(getattr(logging, level, logging.INFO), message)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SIGNAL ENGINE  —  Momentum + optional MACD                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SignalEngine:
    """Stateless helpers that use Gamma API data and accumulated prices."""

    @staticmethod
    def momentum_signal(market: dict) -> Tuple[str, float]:
        """Use Gamma API 1h/1d/1w price changes. Returns (direction, strength)."""
        ch1h = float(market.get("change_1h", 0) or 0)
        ch1d = float(market.get("change_1d", 0) or 0)
        ch1w = float(market.get("change_1w", 0) or 0)

        # Weighted momentum score
        score = ch1h * 0.5 + ch1d * 0.3 + ch1w * 0.2

        if abs(score) < 0.005:
            return "NEUTRAL", abs(score)
        if score > 0:
            return "BUY", min(abs(score) * 5, 1.0)
        return "SELL", min(abs(score) * 5, 1.0)

    @staticmethod
    def macd_signal(prices: List[float]) -> Tuple[str, float]:
        """Return (BUY | SELL | NEUTRAL, strength 0-1)."""
        needed = config.MACD_SLOW + config.MACD_SIGNAL + 1
        if len(prices) < needed:
            return "NEUTRAL", 0.0

        s = pd.Series(prices)
        ema_fast = s.ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = s.ewm(span=config.MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        hist = (macd_line - signal_line).dropna()

        if len(hist) < 2:
            return "NEUTRAL", 0.0

        cur = float(hist.iloc[-1])
        prev = float(hist.iloc[-2])

        if cur > 0 and prev <= 0:
            return "BUY", min(abs(cur) * 10, 1.0)
        if cur < 0 and prev >= 0:
            return "SELL", min(abs(cur) * 10, 1.0)
        if cur > 0:
            return "BUY", min(abs(cur) * 5, 0.6)
        if cur < 0:
            return "SELL", min(abs(cur) * 5, 0.6)
        return "NEUTRAL", 0.0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLAUDE AI — PRIMARY STRATEGY ENGINE                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ClaudeAdvisor:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def analyze_market(
        self,
        market_question: str,
        description: str,
        current_price: float,
        end_date: str,
        volume_24h: float,
        liquidity: float,
        change_1h: float,
        change_1d: float,
        change_1w: float,
        spread: float,
    ) -> Tuple[bool, str, float, float, str]:
        """
        PRIMARY strategy: Claude evaluates if a market is mispriced.
        Returns (should_trade, action, confidence, estimated_prob, reasoning).
        """
        prompt = (
            "You are an expert quantitative trader on Polymarket prediction markets.\n"
            "Your job: find MISPRICED markets where the current price doesn't reflect true probability.\n\n"
            f'Market: "{market_question}"\n'
            f"Description: {description[:400]}\n"
            f"Current YES price: {current_price:.3f} (= {current_price*100:.1f}% implied probability)\n"
            f"End date: {end_date}\n"
            f"24h volume: ${volume_24h:,.0f}\n"
            f"Liquidity: ${liquidity:,.0f}\n"
            f"Price changes: 1h={change_1h:+.3f}, 1d={change_1d:+.3f}, 1w={change_1w:+.3f}\n"
            f"Spread: {spread}\n"
            f"Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
            "ANALYSIS REQUIRED:\n"
            "1. What is the TRUE probability this event happens? (your expert estimate)\n"
            "2. Is the market mispriced? By how much?\n"
            "3. Which side should we trade? BUY_YES if true prob > market price, BUY_NO if true prob < market price\n"
            "4. How confident are you in this analysis?\n\n"
            "RULES:\n"
            "- Be conservative. Only recommend trades when you see clear mispricing of at least 4+ cents.\n"
            "- Consider time until resolution, current events, and market context.\n"
            "- High volume + large price changes suggest new information — analyze carefully.\n"
            "- Markets near 0.50 have the most edge potential. Near 0/1 have less.\n"
            "- NEVER trade if you are unsure — HOLD is always valid.\n\n"
            'Respond with ONLY valid JSON (no markdown fences):\n'
            '{"should_trade": bool, "action": "BUY_YES"|"BUY_NO"|"HOLD", '
            '"confidence": 0.0-1.0, "estimated_probability": 0.0-1.0, '
            '"mispricing": float_cents, "reasoning": "2-3 sentences"}'
        )

        try:
            resp = self.client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            should_trade = bool(result.get("should_trade", False))
            action = result.get("action", "HOLD")
            confidence = float(result.get("confidence", 0))
            est_prob = float(result.get("estimated_probability", current_price))
            mispricing = float(result.get("mispricing", 0))
            reasoning = result.get("reasoning", "")

            analysis = f"[{action}] conf {confidence:.0%} | est={est_prob:.1%} vs mkt={current_price:.1%} | {reasoning}"

            with _state_lock:
                _bot_state.claude_analyses.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "market": market_question[:60],
                    "signal_in": f"price={current_price:.3f}",
                    "action": action,
                    "confidence": confidence,
                    "estimated_prob": est_prob,
                    "mispricing": mispricing,
                    "reasoning": reasoning,
                })
                if len(_bot_state.claude_analyses) > 200:
                    _bot_state.claude_analyses = _bot_state.claude_analyses[-200:]

            # Trade if Claude is confident AND finds significant mispricing
            trade_signal = (
                should_trade
                and confidence >= config.CLAUDE_CONFIDENCE_MIN
                and abs(est_prob - current_price) >= config.MISPRICING_MIN
                and action in ("BUY_YES", "BUY_NO")
            )

            return trade_signal, action, confidence, est_prob, analysis

        except Exception as exc:
            logger.error("Claude API error: %s", exc)
            return False, "HOLD", 0.0, current_price, f"Error: {exc}"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  POLYBOT ENGINE                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PolyBot:
    """Main orchestrator — market scanning, analysis, execution, liquidity."""

    def __init__(self):
        self.clob = None
        self.clob_error: str = ""
        self.signal_engine = SignalEngine()
        self.claude: Optional[ClaudeAdvisor] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None

    # ── CLOB client initialisation ──────────────────────────────────────────

    def _load_secret(self, key: str, fallback: str = "") -> str:
        """Load a secret from st.secrets (Streamlit Cloud) or config/env."""
        try:
            import streamlit as st
            val = st.secrets.get(key, "")
            if val:
                return str(val)
        except Exception:
            pass
        return getattr(config, key, "") or fallback

    def init_clob(self):
        """Initialise the CLOB client with proper auth. Safe to call repeatedly."""
        self.clob_error = ""
        pk = self._load_secret("PRIVATE_KEY", config.PRIVATE_KEY)
        funder = self._load_secret("FUNDER_ADDRESS", config.FUNDER_ADDRESS)

        if not pk:
            self.clob_error = "PRIVATE_KEY not set"
            add_log(f"⚠️ CLOB skip: {self.clob_error}", "WARNING")
            return

        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            self.clob = None
            self.clob_error = "py_clob_client not installed"
            add_log(f"⚠️ CLOB skip: {self.clob_error}", "WARNING")
            return

        try:
            client = ClobClient(
                host=config.CLOB_HOST,
                key=pk,
                chain_id=config.CHAIN_ID,
                signature_type=1,       # POLY_PROXY (Polymarket browser wallets)
                funder=funder or None,
            )
            creds = client.derive_api_key()
            client.set_api_creds(creds)
            self.clob = client
            add_log("✅ CLOB client connected — LIVE mode available")
        except Exception as exc:
            self.clob = None
            self.clob_error = str(exc)
            add_log(f"⚠️ CLOB init failed: {exc}", "WARNING")

    # ── full initialisation ─────────────────────────────────────────────────

    def initialize(self) -> bool:
        # Claude advisor
        api_key = self._load_secret("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY)
        if api_key:
            try:
                config.ANTHROPIC_API_KEY = api_key
                self.claude = ClaudeAdvisor()
                add_log("✅ Claude advisor ready")
            except Exception as exc:
                add_log(f"⚠️ Claude init failed: {exc}", "WARNING")

        # CLOB client
        self.init_clob()

        with _state_lock:
            _bot_state.running = True
            _bot_state.paper_mode = config.PAPER_TRADING
            if not config.PAPER_TRADING and self.clob is None:
                _bot_state.paper_mode = True
                add_log("⚠️ Forced paper mode — no CLOB connection", "WARNING")
            mode_label = "PAPER 📝" if _bot_state.paper_mode else "LIVE 🔴"
            source = "cloud detected" if config._is_cloud() else "local machine"
            add_log(f"🚀 Mode: {mode_label} ({source})")
            _bot_state.uptime_start = datetime.now(timezone.utc).isoformat()

        # Fetch real balance on init
        self.sync_balance()
        return True

    # ── balance sync ────────────────────────────────────────────────────────

    def sync_balance(self, log: bool = True):
        """Read real wallet balance from CLOB or set paper default."""
        if self.clob and not _bot_state.paper_mode:
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

                params = BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=1,   # must match ClobClient signature_type
                )
                bal_resp = self.clob.get_balance_allowance(params)
                if isinstance(bal_resp, dict):
                    real_bal = float(bal_resp.get("balance", 0)) / 1e6  # USDC 6 decimals
                else:
                    real_bal = float(getattr(bal_resp, "balance", 0)) / 1e6
                prev = _bot_state.real_balance
                with _state_lock:
                    _bot_state.real_balance = round(real_bal, 2)
                    _bot_state.balance = _bot_state.real_balance
                    if _bot_state.initial_balance == 0:
                        _bot_state.initial_balance = _bot_state.real_balance
                # Only log if balance changed or explicitly requested
                if log and abs(prev - _bot_state.real_balance) >= 0.01:
                    add_log(f"💰 Real balance: ${_bot_state.real_balance:.2f}", "INFO")
            except Exception as exc:
                if log:
                    add_log(f"⚠️ Balance fetch failed: {exc}", "WARNING")
        else:
            with _state_lock:
                if _bot_state.balance == 0:
                    _bot_state.balance = 1000.0
                    _bot_state.initial_balance = 1000.0

    # ── market discovery (Gamma API) ────────────────────────────────────────

    async def fetch_markets(self) -> List[dict]:
        """Fetch active markets from Gamma API with full metadata."""
        try:
            resp = _http_session.get(
                f"{config.GAMMA_HOST}/markets",
                params={
                    "limit": 50,
                    "active": "true",
                    "closed": "false",
                    "sortBy": "volume24hr",
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            markets = data if isinstance(data, list) else data.get("data", [])
            add_log(f"✅ Fetched {len(markets)} markets from Gamma API")
            return markets
        except Exception as exc:
            add_log(f"Market fetch error: {exc}", "ERROR")
            await asyncio.sleep(5)
            return []

    # ── order-book (public, no auth) ────────────────────────────────────────

    async def fetch_order_book(self, token_id: str) -> dict:
        try:
            resp = _http_session.get(
                f"{config.CLOB_HOST}/book",
                params={"token_id": token_id},
                timeout=10,
            )
            if resp.status_code == 200:
                book = resp.json()
                return {
                    "bids": [{"price": float(b["price"]), "size": float(b["size"])}
                             for b in (book.get("bids") or [])[:5]],
                    "asks": [{"price": float(a["price"]), "size": float(a["size"])}
                             for a in (book.get("asks") or [])[:5]],
                    "last_trade_price": book.get("last_trade_price", ""),
                }
        except Exception:
            pass
        return {"bids": [], "asks": [], "last_trade_price": ""}

    # ── market selection & state ────────────────────────────────────────────

    def select_markets(self, raw: List[dict]) -> List[dict]:
        """Filter and rank markets by tradability."""
        scored: List[Tuple[float, dict]] = []
        for m in raw:
            try:
                vol24 = float(m.get("volume24hr", 0) or 0)
                liquidity = float(m.get("liquidityNum", 0) or m.get("liquidity", 0) or 0)

                if vol24 < config.MIN_VOLUME_24H:
                    continue
                if liquidity < config.MIN_LIQUIDITY:
                    continue

                # Parse prices from outcomePrices JSON string
                prices_str = m.get("outcomePrices", "[]")
                prices_list = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                if not prices_list:
                    continue
                yes_price = float(prices_list[0])

                if yes_price < config.PRICE_RANGE_MIN or yes_price > config.PRICE_RANGE_MAX:
                    continue

                # Score: prefer high volume, good liquidity, prices near 0.5
                edge_potential = 1.0 - abs(yes_price - 0.5) * 2  # peaks at 0.5
                score = vol24 * 0.4 + liquidity * 0.3 + edge_potential * 10000 * 0.3
                scored.append((score, m))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [m for _, m in scored[:config.MAX_CONCURRENT_MARKETS]]
        add_log(f"🎯 Selected {len(selected)} tradeable markets")
        return selected

    def upsert_market(self, m: dict):
        """Parse Gamma API market data and upsert into state."""
        cid = m.get("conditionId", "") or m.get("condition_id", "")
        if not cid:
            return

        # Parse token IDs from clobTokenIds JSON string
        clob_tokens_str = m.get("clobTokenIds", "[]")
        try:
            token_ids = json.loads(clob_tokens_str) if isinstance(clob_tokens_str, str) else clob_tokens_str
        except (json.JSONDecodeError, TypeError):
            token_ids = []

        # Parse prices from outcomePrices JSON string
        prices_str = m.get("outcomePrices", "[]")
        try:
            prices_list = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        except (json.JSONDecodeError, TypeError):
            prices_list = []

        if not token_ids or not prices_list:
            return

        yes_price = float(prices_list[0]) if prices_list else 0.5

        with _state_lock:
            if cid not in _bot_state.markets:
                _bot_state.markets[cid] = {
                    "condition_id": cid,
                    "question": (m.get("question") or "Unknown")[:80],
                    "description": (m.get("description") or "")[:500],
                    "end_date": m.get("endDate", ""),
                    "token_id_yes": token_ids[0] if token_ids else "",
                    "token_id_no": token_ids[1] if len(token_ids) > 1 else "",
                    "current_price": 0.5,
                    "prices": [],
                    "timestamps": [],
                    "position_side": "",
                    "position_size": 0.0,
                    "entry_price": 0.0,
                    "unrealized_pnl": 0.0,
                    "liquidity_ids": [],
                    "last_signal": "",
                    "macd_value": 0.0,
                    "momentum": 0.0,
                    "peak_price": 0.0,
                    "trailing_stop": 0.0,
                    "take_profit": 0.0,
                    # Gamma API metadata
                    "volume_24h": 0.0,
                    "liquidity": 0.0,
                    "change_1h": 0.0,
                    "change_1d": 0.0,
                    "change_1w": 0.0,
                    "spread": 0.0,
                    "best_bid": 0.0,
                    "best_ask": 0.0,
                }

            ms = _bot_state.markets[cid]
            ms["current_price"] = yes_price
            ms["prices"].append(yes_price)
            ms["timestamps"].append(datetime.now(timezone.utc).isoformat())
            if len(ms["prices"]) > 300:
                ms["prices"] = ms["prices"][-300:]
                ms["timestamps"] = ms["timestamps"][-300:]

            # Update Gamma API metadata each scan
            ms["volume_24h"] = float(m.get("volume24hr", 0) or 0)
            ms["liquidity"] = float(m.get("liquidityNum", 0) or m.get("liquidity", 0) or 0)
            ms["change_1h"] = float(m.get("oneHourPriceChange", 0) or 0)
            ms["change_1d"] = float(m.get("oneDayPriceChange", 0) or 0)
            ms["change_1w"] = float(m.get("oneWeekPriceChange", 0) or 0)
            ms["spread"] = float(m.get("spread", 0) or 0)
            ms["best_bid"] = float(m.get("bestBid", 0) or 0)
            ms["best_ask"] = float(m.get("bestAsk", 0) or 0)
            ms["description"] = (m.get("description") or ms.get("description", ""))[:500]
            ms["end_date"] = m.get("endDate", "") or ms.get("end_date", "")

    # ── cooldown check ───────────────────────────────────────────────────────

    def is_on_cooldown(self, cid: str) -> bool:
        cd = _bot_state.cooldowns.get(cid)
        if not cd:
            return False
        try:
            cd_time = datetime.fromisoformat(cd)
            elapsed = (datetime.now(timezone.utc) - cd_time).total_seconds()
            return elapsed < config.COOLDOWN_SECONDS
        except Exception:
            return False

    def set_cooldown(self, cid: str):
        _bot_state.cooldowns[cid] = datetime.now(timezone.utc).isoformat()

    # ── trailing stop / take-profit check ───────────────────────────────────

    async def check_exit_conditions(self, cid: str):
        ms = _bot_state.markets.get(cid)
        if not ms or ms["position_size"] <= 0:
            return

        price = ms["current_price"]
        entry = ms["entry_price"]
        side = ms["position_side"]

        # Calculate current return
        if side == "YES":
            ret = (price - entry) / entry if entry > 0 else 0
            # Update peak
            if price > ms.get("peak_price", 0):
                ms["peak_price"] = price
            trailing_ref = ms["peak_price"]
            trail_hit = (trailing_ref - price) / trailing_ref >= config.TRAILING_STOP_PCT if trailing_ref > 0 else False
        else:
            ret = (entry - price) / entry if entry > 0 else 0
            if price < ms.get("peak_price", 999) or ms.get("peak_price", 0) == 0:
                ms["peak_price"] = price
            trailing_ref = ms["peak_price"]
            trail_hit = (price - trailing_ref) / trailing_ref >= config.TRAILING_STOP_PCT if trailing_ref > 0 else False

        tp_hit = ret >= config.TAKE_PROFIT_PCT

        if trail_hit or tp_hit:
            reason = "TRAILING STOP" if trail_hit else "TAKE PROFIT"
            pnl = ret * ms["position_size"]
            add_log(f"🔔 {reason} | {ms['question'][:35]} | PnL ${pnl:.2f}", "INFO")

            # Close position
            with _state_lock:
                _bot_state.trades.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "market": ms["question"][:60],
                    "side": f"CLOSE ({reason})",
                    "price": price,
                    "size": ms["position_size"],
                    "pnl": round(pnl, 4),
                    "source": reason,
                    "confidence": 1.0,
                })
                _bot_state.total_trades += 1
                if pnl > 0:
                    _bot_state.winning_trades += 1
                ms["position_side"] = ""
                ms["position_size"] = 0.0
                ms["entry_price"] = 0.0
                ms["peak_price"] = 0.0
                ms["unrealized_pnl"] = 0.0
            self.set_cooldown(cid)
            save_state()

    # ── analysis pipeline (Claude-First) ──────────────────────────────────────

    def _should_call_claude(self, cid: str, ms: dict) -> bool:
        """Pre-filter: decide if this market warrants a Claude API call.
        Returns False to SKIP Claude (saves money)."""
        # Never re-analyze within cooldown period
        last_ts = ms.get("claude_last_ts", "")
        if last_ts:
            try:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_ts)).total_seconds()
                if elapsed < config.CLAUDE_COOLDOWN_MARKET:
                    # Exception: re-analyze if price moved significantly
                    last_price = ms.get("claude_last_price", 0)
                    if last_price > 0:
                        change = abs(ms["current_price"] - last_price) / last_price
                        if change < config.CLAUDE_PRICE_CHANGE_TRIGGER:
                            return False  # skip — nothing changed
                    else:
                        return False
            except Exception:
                pass

        # Pre-filter: require at least ONE reason to analyze
        mom_dir, _ = self.signal_engine.momentum_signal(ms)
        has_momentum = mom_dir != "NEUTRAL"
        has_volume_spike = ms.get("volume_24h", 0) > 5000
        price_near_half = 0.25 <= ms["current_price"] <= 0.75
        has_price_move = abs(ms.get("change_1d", 0)) > 0.02

        # Need at least one trigger to justify spending on Claude
        return has_momentum or has_volume_spike or price_near_half or has_price_move

    async def analyze_market(self, cid: str) -> Optional[dict]:
        ms = _bot_state.markets.get(cid)
        if not ms:
            return None

        # Cooldown check (trade cooldown)
        if self.is_on_cooldown(cid):
            return None

        # Skip if already have a position (wait for exit)
        if ms["position_size"] > 0:
            return None

        prices = ms["prices"]
        if len(prices) < config.DATA_POINTS_MIN:
            return None

        # 1) Momentum from Gamma API (quick, no API call)
        mom_dir, mom_str = self.signal_engine.momentum_signal(ms)
        ms["momentum"] = mom_str

        # 2) MACD if enough data (optional, tertiary)
        macd_dir, macd_str = self.signal_engine.macd_signal(prices)
        ms["macd_value"] = macd_str

        # 3) Claude AI — PRIMARY STRATEGY (with pre-filter to save $$$)
        if self.claude:
            if not self._should_call_claude(cid, ms):
                ms["last_signal"] = ms.get("last_signal", "WAIT (filtered)")
                return None

            trade_signal, action, confidence, est_prob, analysis = self.claude.analyze_market(
                market_question=ms["question"],
                description=ms.get("description", ""),
                current_price=ms["current_price"],
                end_date=ms.get("end_date", ""),
                volume_24h=ms.get("volume_24h", 0),
                liquidity=ms.get("liquidity", 0),
                change_1h=ms.get("change_1h", 0),
                change_1d=ms.get("change_1d", 0),
                change_1w=ms.get("change_1w", 0),
                spread=ms.get("spread", 0),
            )
            # Record when we last called Claude for this market
            ms["claude_last_ts"] = datetime.now(timezone.utc).isoformat()
            ms["claude_last_price"] = ms["current_price"]

            add_log(f"🤖 Claude → {ms['question'][:35]}… {analysis[:80]}")

            if not trade_signal:
                ms["last_signal"] = f"HOLD (conf {confidence:.0%})"
                return None

            # Momentum confirmation (bonus, not required)
            if config.MOMENTUM_ENABLED and mom_dir != "NEUTRAL":
                expected_dir = "BUY" if action == "BUY_YES" else "SELL"
                if mom_dir == expected_dir:
                    add_log(f"📈 Momentum confirms {action}")

            ms["last_signal"] = f"{action} ✅ (conf {confidence:.0%})"
            return {
                "cid": cid,
                "action": action,
                "confidence": confidence,
                "est_prob": est_prob,
                "macd_str": macd_str,
                "mom_str": mom_str,
            }
        else:
            # No Claude — use momentum + MACD agreement (fallback)
            if mom_dir == "NEUTRAL" and macd_dir == "NEUTRAL":
                ms["last_signal"] = "NEUTRAL"
                return None
            direction = mom_dir if mom_dir != "NEUTRAL" else macd_dir
            if direction == "NEUTRAL":
                ms["last_signal"] = "NEUTRAL"
                return None
            action = "BUY_YES" if direction == "BUY" else "BUY_NO"
            ms["last_signal"] = f"{action} (no-AI, momentum)"
            return {
                "cid": cid,
                "action": action,
                "confidence": 0.5,
                "est_prob": ms["current_price"],
                "macd_str": macd_str,
                "mom_str": mom_str,
            }

    # ── execution ───────────────────────────────────────────────────────────

    async def execute_signal(self, sig: dict):
        ms = _bot_state.markets.get(sig["cid"])
        if not ms:
            return

        action = sig["action"]  # BUY_YES or BUY_NO
        price = ms["current_price"]

        # Dynamic size: configured order size, capped at % of balance
        max_risk = _bot_state.balance * config.MAX_EXPOSURE_PCT
        size = min(config.ORDER_SIZE_USDC, max_risk)
        if size < 1.0:
            add_log("⚠️ Order too small — balance too low", "WARNING")
            return

        # Exposure guards
        total_exp = sum(m["position_size"] for m in _bot_state.markets.values())
        if total_exp + size > config.TOTAL_MAX_EXPOSURE:
            add_log("⚠️ Total exposure limit — skip", "WARNING")
            return
        if ms["position_size"] + size > config.MAX_POSITION_PER_MARKET:
            add_log(f"⚠️ Market limit — {ms['question'][:30]}", "WARNING")
            return

        # Determine token and side
        if action == "BUY_YES":
            token_id = ms["token_id_yes"]
            order_price = price
            position_side = "YES"
        else:  # BUY_NO
            token_id = ms["token_id_no"]
            order_price = 1.0 - price  # NO token price
            position_side = "NO"

        success = False

        if _bot_state.paper_mode:
            success = True
            add_log(f"📝 PAPER {action} | {ms['question'][:40]} @ {price:.3f} × ${size:.2f}")
        else:
            if not self.clob:
                add_log("⚠️ No CLOB connection — queuing for next cycle", "WARNING")
                return
            add_log(f"🔴 LIVE {action} | {ms['question'][:40]} @ {order_price:.3f} × ${size:.2f}", "INFO")
            try:
                from py_clob_client.clob_types import OrderArgs

                # Resolve tick size for proper price rounding
                try:
                    tick = float(self.clob.get_tick_size(token_id))
                except Exception:
                    tick = 0.01
                # Round price to tick precision (0.01 → 2 decimals, 0.001 → 3)
                decimals = max(2, len(str(tick).rstrip('0').split('.')[-1]))
                rounded_price = round(order_price, decimals)
                # Ensure price in valid range
                rounded_price = max(tick, min(rounded_price, 1.0 - tick))

                order_args = OrderArgs(
                    price=rounded_price,
                    size=round(size, 2),
                    side="BUY",
                    token_id=token_id,
                )
                signed = self.clob.create_order(order_args)
                resp = self.clob.post_order(signed)
                if resp and (resp.get("success") or resp.get("orderID")):
                    success = True
                    add_log(f"🎯 LIVE FILLED {action} | {ms['question'][:40]} | orderID={resp.get('orderID','')[:20]}", "INFO")
                else:
                    add_log(f"❌ Order rejected: {resp}", "ERROR")
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                detail = getattr(exc, "error_msg", "")
                if status == 403:
                    add_log(
                        "❌ Geo-blocked (403) — Polymarket blocks cloud IPs. Run the bot locally (INICIAR_BOT.bat).",
                        "ERROR",
                    )
                else:
                    add_log(f"❌ Execution error (status={status}): {detail or exc}", "ERROR")

        if success:
            with _state_lock:
                ms["position_side"] = position_side
                ms["position_size"] += size
                ms["entry_price"] = price
                ms["peak_price"] = price

                _bot_state.trades.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "market": ms["question"][:60],
                    "side": action,
                    "price": price,
                    "size": round(size, 2),
                    "pnl": 0.0,
                    "source": "Claude AI",
                    "confidence": sig["confidence"],
                    "mode": "PAPER" if _bot_state.paper_mode else "LIVE",
                })
                _bot_state.total_trades += 1
                if len(_bot_state.trades) > 1000:
                    _bot_state.trades = _bot_state.trades[-1000:]
            self.set_cooldown(sig["cid"])
            save_state()

    # ── liquidity farming ───────────────────────────────────────────────────

    async def refresh_liquidity(self, cid: str):
        if not config.LIQ_ENABLED or _bot_state.paper_mode or not self.clob:
            return
        ms = _bot_state.markets.get(cid)
        if not ms:
            return

        try:
            # Cancel stale orders
            for oid in ms.get("liquidity_ids", []):
                try:
                    self.clob.cancel(oid)
                except Exception:
                    pass

            from py_clob_client.clob_types import OrderArgs

            mid = ms["current_price"]
            bid_px = max(0.01, round(mid - config.LIQ_SPREAD, 2))
            ask_px = min(0.99, round(mid + config.LIQ_SPREAD, 2))
            new_ids = []

            for px, side in [(bid_px, "BUY"), (ask_px, "SELL")]:
                args = OrderArgs(
                    price=px, size=config.LIQ_SIZE, side=side, token_id=ms["token_id_yes"]
                )
                signed = self.clob.create_order(args)
                resp = self.clob.post_order(signed)
                if resp and resp.get("orderID"):
                    new_ids.append(resp["orderID"])

            ms["liquidity_ids"] = new_ids
            add_log(f"💧 Liq refresh {ms['question'][:25]} bid={bid_px} ask={ask_px}")

        except Exception as exc:
            logger.debug("Liquidity error: %s", exc)

    # ── P&L ─────────────────────────────────────────────────────────────────

    def update_pnl(self):
        total = 0.0
        wins = 0
        for ms in _bot_state.markets.values():
            if ms["position_size"] > 0:
                if ms["position_side"] == "YES":
                    unr = (ms["current_price"] - ms["entry_price"]) * ms["position_size"]
                else:
                    unr = (ms["entry_price"] - ms["current_price"]) * ms["position_size"]
                ms["unrealized_pnl"] = round(unr, 4)
                total += unr
                if unr > 0:
                    wins += 1

        with _state_lock:
            _bot_state.total_pnl = round(total, 4)
            _bot_state.winning_trades = wins
            _bot_state.balance = _bot_state.initial_balance + total
            _bot_state.pnl_history.append(
                {"timestamp": datetime.now(timezone.utc).isoformat(), "pnl": total}
            )
            if len(_bot_state.pnl_history) > 2000:
                _bot_state.pnl_history = _bot_state.pnl_history[-2000:]

    # ── main cycle ──────────────────────────────────────────────────────────

    async def run_cycle(self):
        try:
            # Sync paper_mode from config each cycle
            with _state_lock:
                _bot_state.paper_mode = config.PAPER_TRADING

            # If live mode requested but no CLOB yet, retry connection
            if not config.PAPER_TRADING and self.clob is None:
                self.init_clob()

            # 0  Sync balance (log=True so changes are reported once per cycle)
            self.sync_balance(log=True)

            # 1  Discover markets from Gamma API
            raw = await self.fetch_markets()
            if raw:
                for m in self.select_markets(raw):
                    self.upsert_market(m)

            _bot_state.last_scan = datetime.now(timezone.utc).isoformat()

            # 2  Check exit conditions (trailing stop / TP)
            for cid in list(_bot_state.markets.keys()):
                if not self.running:
                    break
                await self.check_exit_conditions(cid)

            # 3  Analyse + Execute (Claude analyzes each market)
            analyzed = 0
            for cid in list(_bot_state.markets.keys()):
                if not self.running:
                    break
                sig = await self.analyze_market(cid)
                if sig:
                    await self.execute_signal(sig)
                analyzed += 1
                # Brief pause between market analyses to avoid rate limits
                if analyzed % 5 == 0:
                    await asyncio.sleep(1)

            # 4  P&L
            self.update_pnl()
            save_state()
            add_log(f"📊 Cycle done: {len(_bot_state.markets)} markets, {_bot_state.total_trades} trades, P&L: ${_bot_state.total_pnl:+.2f}")

        except Exception as exc:
            add_log(f"Cycle error: {exc}", "ERROR")

    async def _loop(self):
        add_log("🚀 PolyBot engine started")
        while self.running:
            await self.run_cycle()
            add_log(f"⏳ Next scan in {config.SCAN_INTERVAL_SECONDS}s …")
            await asyncio.sleep(config.SCAN_INTERVAL_SECONDS)
        add_log("🛑 PolyBot engine stopped")

    # ── thread management ───────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self.running = True
        self.initialize()

        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._loop())
            finally:
                loop.close()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        add_log("🤖 Bot thread launched")

    def stop(self):
        self.running = False
        with _state_lock:
            _bot_state.running = False
        save_state()
        add_log("Bot stopping …")


# ── singleton accessor ──────────────────────────────────────────────────────

_instance: Optional[PolyBot] = None


def get_bot() -> PolyBot:
    global _instance
    if _instance is None:
        _instance = PolyBot()
    return _instance
