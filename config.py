"""
PolyBot Configuration
Loads environment variables and defines all trading parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
FUNDER_ADDRESS: str = os.getenv("FUNDER_ADDRESS", "")

# ---------------------------------------------------------------------------
# Polymarket Endpoints
# ---------------------------------------------------------------------------
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

# ---------------------------------------------------------------------------
# Strategy — MACD Crossover
# ---------------------------------------------------------------------------
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ---------------------------------------------------------------------------
# Strategy — Cumulative Volume Delta (CVD)
# ---------------------------------------------------------------------------
CVD_LOOKBACK = 50
CVD_THRESHOLD = 0.35  # normalized delta threshold to confirm direction

# ---------------------------------------------------------------------------
# Strategy — Claude AI Confirmation
# ---------------------------------------------------------------------------
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_CONFIDENCE_MIN = 0.68  # minimum confidence to accept signal
SIGNAL_AGREEMENT_MIN = 2      # at least 2/3 indicators must agree

# ---------------------------------------------------------------------------
# Trading Parameters
# ---------------------------------------------------------------------------
MAX_CONCURRENT_MARKETS = 20
ORDER_SIZE_USDC = 5.0           # base order size
MAX_POSITION_PER_MARKET = 50.0  # max exposure per market
TOTAL_MAX_EXPOSURE = 500.0      # total portfolio max
SLIPPAGE_TOLERANCE = 0.02

# ---------------------------------------------------------------------------
# Liquidity Farming
# ---------------------------------------------------------------------------
LIQ_ENABLED = True
LIQ_SPREAD = 0.03            # 3 cents from mid price
LIQ_SIZE = 2.0               # USDC per side
LIQ_REFRESH_SECONDS = 120

# ---------------------------------------------------------------------------
# Intervals & Limits
# ---------------------------------------------------------------------------
SCAN_INTERVAL_SECONDS = 30
DATA_POINTS_MIN = 30          # minimum prices before MACD fires

# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
PAPER_TRADING = True  # True = paper‑trade, False = live orders

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATE_FILE = Path(__file__).resolve().parent / "bot_state.json"
LOG_FILE = Path(__file__).resolve().parent / "bot.log"
