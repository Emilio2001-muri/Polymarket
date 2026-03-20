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
# Strategy — Claude AI (Primary Strategy)
# ---------------------------------------------------------------------------
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_CONFIDENCE_MIN = 0.70    # minimum confidence to accept trade
MISPRICING_MIN = 0.04           # minimum 4 cent mispricing to trade

# ---------------------------------------------------------------------------
# Strategy — Momentum (Secondary Confirmation)
# ---------------------------------------------------------------------------
MOMENTUM_ENABLED = True         # use Gamma API price changes as confirmation

# ---------------------------------------------------------------------------
# Strategy — MACD (Optional, tertiary)
# ---------------------------------------------------------------------------
MACD_FAST = 8
MACD_SLOW = 17
MACD_SIGNAL = 6

# ---------------------------------------------------------------------------
# Trading Parameters
# ---------------------------------------------------------------------------
MAX_CONCURRENT_MARKETS = 15
ORDER_SIZE_USDC = 5.0           # base order size
MAX_POSITION_PER_MARKET = 25.0  # max exposure per market
TOTAL_MAX_EXPOSURE = 200.0      # total portfolio max
SLIPPAGE_TOLERANCE = 0.02

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
MAX_EXPOSURE_PCT = 0.10         # max 10% of balance per trade
TRAILING_STOP_PCT = 0.10        # 10% trailing stop
TAKE_PROFIT_PCT = 0.20          # 20% take-profit
COOLDOWN_SECONDS = 180          # 3 min cooldown per market after trade

# ---------------------------------------------------------------------------
# Liquidity Farming
# ---------------------------------------------------------------------------
LIQ_ENABLED = False             # disabled by default for safety
LIQ_SPREAD = 0.03
LIQ_SIZE = 2.0
LIQ_REFRESH_SECONDS = 120

# ---------------------------------------------------------------------------
# Intervals & Limits
# ---------------------------------------------------------------------------
SCAN_INTERVAL_SECONDS = 30
DATA_POINTS_MIN = 3             # minimum prices before analysis

# ---------------------------------------------------------------------------
# Market Filters
# ---------------------------------------------------------------------------
MIN_VOLUME_24H = 500            # minimum 24h volume to consider
MIN_LIQUIDITY = 200             # minimum liquidity
PRICE_RANGE_MIN = 0.08          # skip markets below 8%
PRICE_RANGE_MAX = 0.92          # skip markets above 92%

# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
PAPER_TRADING = True  # True = paper-trade, False = live orders

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATE_FILE = Path(__file__).resolve().parent / "bot_state.json"
LOG_FILE = Path(__file__).resolve().parent / "bot.log"
