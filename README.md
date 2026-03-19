# 🤖 PolyBot — Hybrid Quantitative Trader for Polymarket

> **MACD crossover + CVD order‑flow + Liquidity farming + Claude AI confirmation**
> Multi‑market (15‑20 simultaneous) · Premium Streamlit dashboard · Free deployment

---

## 1. Philosophy & Edge (compiled from the best viral bots)

The most profitable Polymarket bots don't rely on a single signal.
They **stack edges**:

| Layer | What it does | Why it works |
|-------|-------------|--------------|
| **MACD Crossover** | Detects momentum shifts before the crowd | Classic trend‑following on probability curves |
| **CVD (Cumulative Volume Delta)** | Confirms real buying/selling pressure | Filters out false MACD signals caused by noise |
| **Claude AI Gate** | LLM reasoning on each signal | Catches context a pure quant model misses (e.g. news, market structure) |
| **Liquidity Farming** | Passive rewards from resting limit orders | Free yield while you wait for signals |
| **Multi‑Market** | Scan 15‑20 markets concurrently | Diversification + more signal opportunities |

**Paper‑trade first.** The bot ships in paper mode by default — no real
money touches the chain until you explicitly switch to live and start
with $1.

---

## 2. Project Structure

```
Polymarket/
├── .env                  # Your real secrets (never commit)
├── .env.example          # Template
├── requirements.txt      # Dependencies
├── config.py             # All parameters in one place
├── bot_core.py           # Async trading engine
├── app.py                # Streamlit dashboard (4 themes)
├── backtester.py         # Multi‑market backtester
├── deployment_guide.md   # Free deployment walkthrough
└── README.md             # This file
```

---

## 3. Quick Start

### A. Install dependencies

```bash
cd Polymarket
pip install -r requirements.txt
```

### B. Configure secrets

Copy `.env.example` → `.env` and fill in your 3 keys:

```
ANTHROPIC_API_KEY=sk-ant-...
PRIVATE_KEY=0x...
FUNDER_ADDRESS=0x...
```

### C. Launch the dashboard

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

### D. Start the bot

1. In the dashboard sidebar, make sure **📝 Paper Trading** is ON.
2. Click **▶ Start**.
3. Watch markets populate, signals fire, and the P&L curve grow.

---

## 4. Dashboard Themes

| Theme | Description |
|-------|-------------|
| **Light** | Clean white + blue accents |
| **Dark** | Deep dark + purple accents |
| **System** | Follows your OS preference |
| **SPLENDID** | Premium glass‑morphism, gold gradients, animated borders |

Switch themes from the sidebar dropdown — changes apply instantly.

---

## 5. Backtesting

Run the multi‑market backtester locally:

```bash
# 10 synthetic markets, 60 days
python backtester.py

# More markets, with Claude confirmation
python backtester.py --markets 15 --days 90 --with-claude
```

Results are saved to `backtest_results.json` and an interactive HTML chart
is generated at `backtest_results.html`.

---

## 6. Going Live — Paper → $1

> **CRITICAL: Only risk money you can afford to lose. Prediction markets
> carry real financial risk.**

1. **Paper‑trade for at least 1 week.** Verify positive P&L on the
   dashboard with real Polymarket data.
2. In `config.py`, set `PAPER_TRADING = False`.
3. Fund your wallet (`FUNDER_ADDRESS`) with **$1 USDC** on Polygon.
4. Set `ORDER_SIZE_USDC = 1.0` to start with minimal risk.
5. Restart the bot. It will now place real orders on Polymarket CLOB.
6. Scale up gradually: $1 → $5 → $10 → your comfort level.

---

## 7. Key Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAPER_TRADING` | `True` | Paper mode on/off |
| `ORDER_SIZE_USDC` | `5.0` | Base order size |
| `MAX_CONCURRENT_MARKETS` | `20` | How many markets to scan |
| `SCAN_INTERVAL_SECONDS` | `30` | Seconds between full cycles |
| `CLAUDE_CONFIDENCE_MIN` | `0.68` | Minimum Claude confidence |
| `MACD_FAST / SLOW / SIGNAL` | `12/26/9` | MACD parameters |
| `CVD_THRESHOLD` | `0.35` | CVD delta threshold |
| `LIQ_ENABLED` | `True` | Liquidity farming on/off |
| `LIQ_SPREAD` | `0.03` | Spread from mid for liq orders |

---

## 8. Deployment

See [deployment_guide.md](deployment_guide.md) for a step‑by‑step guide
to deploy on **Streamlit Community Cloud** (100% free).

---

## 9. Security Notes

- **Never commit `.env`** — add it to `.gitignore`.
- Your `PRIVATE_KEY` controls your wallet. Keep it secret.
- The bot only reads from Polymarket APIs and places orders with your
  explicit keys. No data is sent anywhere except the Polymarket CLOB
  and Anthropic API.
- Start in paper mode. Verify everything before going live.

---

## License

Personal use. Not financial advice. Trade at your own risk.
