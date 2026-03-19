"""
PolyBot Core — Hybrid Quantitative Trading Engine for Polymarket
=================================================================
Edge stack:
    1. MACD crossover  →  trend / momentum signal
    2. CVD (Cumulative Volume Delta)  →  order‑flow confirmation
    3. Liquidity farming  →  passive rewards from resting limit orders
    4. Claude AI  →  final confirmation gate with contextual reasoning

Designed for 15‑20 simultaneous markets with async I/O and a
thread‑safe state object consumed by the Streamlit dashboard.
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
    balance: float = 1000.0
    initial_balance: float = 1000.0
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
# ║  SIGNAL ENGINE  —  MACD + CVD                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SignalEngine:
    """Stateless helpers that take raw data and return (direction, strength)."""

    @staticmethod
    def macd_signal(prices: List[float]) -> Tuple[str, float]:
        """Return (BUY | SELL | NEUTRAL, strength 0‑1)."""
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

        # Crossover detection
        if cur > 0 and prev <= 0:
            return "BUY", min(abs(cur) * 10, 1.0)
        if cur < 0 and prev >= 0:
            return "SELL", min(abs(cur) * 10, 1.0)
        if cur > 0:
            return "BUY", min(abs(cur) * 5, 0.6)
        if cur < 0:
            return "SELL", min(abs(cur) * 5, 0.6)
        return "NEUTRAL", 0.0

    @staticmethod
    def cvd_signal(trades: List[dict]) -> Tuple[str, float]:
        """Return (BUY | SELL | NEUTRAL, normalised CVD –1…1)."""
        if len(trades) < 10:
            return "NEUTRAL", 0.0

        buy_vol = sum(float(t.get("size", 0)) for t in trades if str(t.get("side", "")).upper() == "BUY")
        sell_vol = sum(float(t.get("size", 0)) for t in trades if str(t.get("side", "")).upper() == "SELL")
        total = buy_vol + sell_vol
        if total == 0:
            return "NEUTRAL", 0.0

        cvd = (buy_vol - sell_vol) / total
        if cvd > config.CVD_THRESHOLD:
            return "BUY", cvd
        if cvd < -config.CVD_THRESHOLD:
            return "SELL", abs(cvd)
        return "NEUTRAL", abs(cvd)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLAUDE AI ADVISOR                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ClaudeAdvisor:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # ── public ──────────────────────────────────────────────────────────────
    def confirm_signal(
        self,
        market_question: str,
        current_price: float,
        macd_signal: str,
        macd_strength: float,
        cvd_signal: str,
        cvd_value: float,
        recent_prices: List[float],
    ) -> Tuple[bool, float, str]:
        """Ask Claude to confirm a signal. Returns (confirmed, confidence, analysis)."""

        prompt = (
            "You are a quantitative trading analyst for Polymarket prediction markets.\n"
            "Analyze this signal and respond with ONLY valid JSON — no markdown fences.\n\n"
            f'Market: "{market_question}"\n'
            f"Current YES price: {current_price:.4f}\n"
            f"MACD signal: {macd_signal} (strength {macd_strength:.2f})\n"
            f"CVD signal: {cvd_signal} (value {cvd_value:.2f})\n"
            f"Recent prices (last 10): {recent_prices[-10:]}\n"
            f'Trend: {"UP" if len(recent_prices) > 5 and recent_prices[-1] > recent_prices[-5] else "DOWN" if len(recent_prices) > 5 else "FLAT"}\n\n'
            "Rules:\n"
            "- Price is a probability 0‑1. BUY_YES = probability should rise. BUY_NO = should fall.\n"
            "- Both MACD and CVD should broadly agree for high confidence.\n"
            "- Avoid trades when price < 0.06 or > 0.94 (low remaining edge).\n\n"
            'Respond ONLY with JSON: {"confirmed": bool, "confidence": 0.0-1.0, '
            '"action": "BUY_YES"|"BUY_NO"|"HOLD", "reasoning": "one sentence"}'
        )

        try:
            resp = self.client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Strip markdown fences if model added them
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            confirmed = bool(result.get("confirmed", False))
            confidence = float(result.get("confidence", 0))
            reasoning = result.get("reasoning", "")
            action = result.get("action", "HOLD")

            analysis = f"[{action}] conf {confidence:.0%} — {reasoning}"

            with _state_lock:
                _bot_state.claude_analyses.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "market": market_question[:60],
                        "signal_in": macd_signal,
                        "action": action,
                        "confidence": confidence,
                        "reasoning": reasoning,
                    }
                )
                if len(_bot_state.claude_analyses) > 200:
                    _bot_state.claude_analyses = _bot_state.claude_analyses[-200:]

            return (
                confirmed and confidence >= config.CLAUDE_CONFIDENCE_MIN,
                confidence,
                analysis,
            )

        except Exception as exc:
            logger.error("Claude API error: %s", exc)
            return False, 0.0, f"Error: {exc}"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  POLYBOT ENGINE                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PolyBot:
    """Main orchestrator — market scanning, analysis, execution, liquidity."""

    def __init__(self):
        self.clob = None
        self.signal_engine = SignalEngine()
        self.claude: Optional[ClaudeAdvisor] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None

    # ── initialisation ──────────────────────────────────────────────────────

    def initialize(self) -> bool:
        # Claude advisor
        if config.ANTHROPIC_API_KEY:
            try:
                self.claude = ClaudeAdvisor()
                add_log("✅ Claude advisor ready")
            except Exception as exc:
                add_log(f"⚠️ Claude init failed: {exc}", "WARNING")

        # CLOB client
        try:
            from py_clob_client.client import ClobClient

            self.clob = ClobClient(
                host=config.CLOB_HOST,
                key=config.PRIVATE_KEY,
                chain_id=config.CHAIN_ID,
                funder=config.FUNDER_ADDRESS,
            )
            creds = self.clob.derive_api_key()
            self.clob.set_api_creds(creds)
            add_log("✅ CLOB client connected")
        except Exception as exc:
            add_log(f"⚠️ CLOB init failed ({exc}). Paper‑mode only.", "WARNING")
            self.clob = None

        with _state_lock:
            _bot_state.running = True
            _bot_state.paper_mode = config.PAPER_TRADING or (self.clob is None)
            _bot_state.uptime_start = datetime.now(timezone.utc).isoformat()
        return True

    # ── market discovery (Gamma API) ────────────────────────────────────────

    async def fetch_markets(self) -> List[dict]:
        try:
            resp = _http_session.get(
                f"{config.GAMMA_HOST}/markets",
                params={"limit": 30, "active": "true", "closed": "false", "sortBy": "volume"},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            markets = data if isinstance(data, list) else data.get("data", [])
            add_log(f"✅ Fetch exitoso: {len(markets)} mercados cargados")
            return markets
        except Exception as exc:
            add_log(f"Market fetch error: {exc}", "ERROR")
            await asyncio.sleep(5)
            return []

    # ── order‑book / trades --------------------------------------------------

    async def fetch_order_book(self, token_id: str) -> dict:
        if self.clob and not _bot_state.paper_mode:
            try:
                book = self.clob.get_order_book(token_id)
                return {
                    "bids": [
                        {"price": float(o.price), "size": float(o.size)}
                        for o in (book.bids or [])
                    ],
                    "asks": [
                        {"price": float(o.price), "size": float(o.size)}
                        for o in (book.asks or [])
                    ],
                }
            except Exception:
                pass
        return {"bids": [], "asks": []}

    async def fetch_trades(self, token_id: str) -> List[dict]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{config.CLOB_HOST}/trades",
                    params={"token_id": token_id, "limit": 100},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data if isinstance(data, list) else data.get("data", [])
        except Exception:
            pass
        return []

    # ── market selection & state ────────────────────────────────────────────

    def select_markets(self, raw: List[dict]) -> List[dict]:
        scored: List[Tuple[float, dict]] = []
        for m in raw:
            try:
                volume = float(m.get("volume", 0) or 0)
                liquidity = float(m.get("liquidity", 0) or 0)
                tokens = m.get("tokens", [])
                if not tokens:
                    continue
                yes_price = float(tokens[0].get("price", 0.5) or 0.5)
                if yes_price < 0.08 or yes_price > 0.92:
                    continue
                score = volume * 0.6 + liquidity * 0.4
                scored.append((score, m))
            except (ValueError, TypeError):
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [m for _, m in scored[: config.MAX_CONCURRENT_MARKETS]]
        add_log(f"🎯 Tracking {len(selected)} markets")
        return selected

    def upsert_market(self, m: dict):
        cid = m.get("condition_id", "")
        tokens = m.get("tokens", [])
        if not cid or not tokens:
            return

        with _state_lock:
            if cid not in _bot_state.markets:
                _bot_state.markets[cid] = {
                    "condition_id": cid,
                    "question": (m.get("question") or "Unknown")[:80],
                    "token_id_yes": tokens[0].get("token_id", ""),
                    "token_id_no": tokens[1].get("token_id", "") if len(tokens) > 1 else "",
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
                    "cvd_value": 0.0,
                }

            ms = _bot_state.markets[cid]
            price = float(tokens[0].get("price", 0.5) or 0.5)
            ms["current_price"] = price
            ms["prices"].append(price)
            ms["timestamps"].append(datetime.now(timezone.utc).isoformat())
            if len(ms["prices"]) > 300:
                ms["prices"] = ms["prices"][-300:]
                ms["timestamps"] = ms["timestamps"][-300:]

    # ── analysis pipeline ───────────────────────────────────────────────────

    async def analyze_market(self, cid: str) -> Optional[dict]:
        ms = _bot_state.markets.get(cid)
        if not ms:
            return None

        prices = ms["prices"]
        if len(prices) < config.DATA_POINTS_MIN:
            return None

        # 1) MACD
        macd_dir, macd_str = self.signal_engine.macd_signal(prices)
        ms["macd_value"] = macd_str

        # 2) CVD
        trades = await self.fetch_trades(ms["token_id_yes"])
        cvd_dir, cvd_val = self.signal_engine.cvd_signal(trades)
        ms["cvd_value"] = cvd_val

        if macd_dir == "NEUTRAL":
            ms["last_signal"] = "NEUTRAL"
            return None

        # Agreement check
        agree = (macd_dir == cvd_dir) or cvd_dir == "NEUTRAL"
        if not agree:
            ms["last_signal"] = f"CONFLICT {macd_dir}/{cvd_dir}"
            return None

        # 3) Claude gate
        if self.claude:
            confirmed, conf, analysis = self.claude.confirm_signal(
                market_question=ms["question"],
                current_price=ms["current_price"],
                macd_signal=macd_dir,
                macd_strength=macd_str,
                cvd_signal=cvd_dir,
                cvd_value=cvd_val,
                recent_prices=prices,
            )
            add_log(f"🤖 Claude → {ms['question'][:35]}… {analysis}")
            if not confirmed:
                ms["last_signal"] = f"REJECTED (conf {conf:.0%})"
                return None
            ms["last_signal"] = f"{macd_dir} ✅ (conf {conf:.0%})"
            return {
                "cid": cid,
                "action": macd_dir,
                "confidence": conf,
                "macd_str": macd_str,
                "cvd_val": cvd_val,
            }
        else:
            # No Claude — require both MACD + CVD agreement
            if macd_dir != cvd_dir:
                ms["last_signal"] = f"NO‑AI CONFLICT"
                return None
            ms["last_signal"] = f"{macd_dir} ✅ (no‑AI)"
            return {
                "cid": cid,
                "action": macd_dir,
                "confidence": 0.5,
                "macd_str": macd_str,
                "cvd_val": cvd_val,
            }

    # ── execution ───────────────────────────────────────────────────────────

    async def execute_signal(self, sig: dict):
        ms = _bot_state.markets.get(sig["cid"])
        if not ms:
            return

        size = config.ORDER_SIZE_USDC
        price = ms["current_price"]

        # Exposure guards
        total_exp = sum(m["position_size"] for m in _bot_state.markets.values())
        if total_exp + size > config.TOTAL_MAX_EXPOSURE:
            add_log("⚠️ Total exposure limit — skip", "WARNING")
            return
        if ms["position_size"] + size > config.MAX_POSITION_PER_MARKET:
            add_log(f"⚠️ Market limit — {ms['question'][:30]}", "WARNING")
            return

        action = sig["action"]
        token_id = ms["token_id_yes"] if action == "BUY" else ms["token_id_no"]
        success = False

        if _bot_state.paper_mode:
            success = True
            add_log(f"📝 PAPER {action} | {ms['question'][:40]} @ {price:.4f} × ${size}")
        else:
            try:
                from py_clob_client.clob_types import OrderArgs

                order_args = OrderArgs(
                    price=round(price, 2),
                    size=size,
                    side="BUY",
                    token_id=token_id,
                )
                signed = self.clob.create_order(order_args)
                resp = self.clob.post_order(signed)
                if resp and (resp.get("success") or resp.get("orderID")):
                    success = True
                    add_log(f"🎯 LIVE {action} | {ms['question'][:40]} @ {price:.4f} × ${size}")
                else:
                    add_log(f"❌ Order rejected: {resp}", "ERROR")
            except Exception as exc:
                add_log(f"❌ Execution error: {exc}", "ERROR")

        if success:
            with _state_lock:
                ms["position_side"] = "YES" if action == "BUY" else "NO"
                ms["position_size"] += size
                ms["entry_price"] = price

                _bot_state.trades.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "market": ms["question"][:60],
                        "side": action,
                        "price": price,
                        "size": size,
                        "pnl": 0.0,
                        "source": "MACD+CVD+Claude",
                        "confidence": sig["confidence"],
                    }
                )
                _bot_state.total_trades += 1
                if len(_bot_state.trades) > 1000:
                    _bot_state.trades = _bot_state.trades[-1000:]
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
            # 1  Discover
            raw = await self.fetch_markets()
            if raw:
                for m in self.select_markets(raw):
                    self.upsert_market(m)

            _bot_state.last_scan = datetime.now(timezone.utc).isoformat()

            # 2  Analyse + Execute
            for cid in list(_bot_state.markets.keys()):
                if not self.running:
                    break
                sig = await self.analyze_market(cid)
                if sig:
                    await self.execute_signal(sig)
                await self.refresh_liquidity(cid)
                await asyncio.sleep(0.3)

            # 3  P&L
            self.update_pnl()
            save_state()

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
