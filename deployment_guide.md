# PolyBot — Deployment Guide (100 % Free)

Deploy the PolyBot dashboard and trading bot to **Streamlit Community Cloud**
at zero cost.  This covers everything from repo creation to a live URL you can
open on your phone.

---

## Prerequisites

| What | Where |
|------|-------|
| GitHub account | <https://github.com> |
| Streamlit Cloud account | <https://share.streamlit.io> (sign in with GitHub) |
| Your 3 secrets | `ANTHROPIC_API_KEY`, `PRIVATE_KEY`, `FUNDER_ADDRESS` |

---

## Step 1 — Push project to GitHub

```bash
# From the Polymarket/ folder
git init
git add .
git commit -m "PolyBot initial commit"
```

Create a **private** repo on GitHub (e.g. `polybot`), then:

```bash
git remote add origin https://github.com/YOUR_USER/polybot.git
git branch -M main
git push -u origin main
```

> **Important:** make sure `.env` is in `.gitignore` so your keys never touch
> GitHub.  Streamlit Cloud has its own secrets manager.

Create a `.gitignore`:

```
.env
bot_state.json
bot.log
__pycache__/
*.pyc
backtest_results.*
```

---

## Step 2 — Connect Streamlit Community Cloud

1. Go to **<https://share.streamlit.io>** and sign in with GitHub.
2. Click **"New app"**.
3. Select your **repo** (`polybot`), branch `main`, and main file `app.py`.
4. Click **"Advanced settings…"**.

---

## Step 3 — Add Secrets

In the **Advanced settings** box, paste your secrets in TOML format:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-..."
PRIVATE_KEY = "0x..."
FUNDER_ADDRESS = "0x..."
```

Streamlit Cloud injects these as environment variables — your `config.py`
reads them automatically via `python-dotenv` / `os.getenv`.

Click **Deploy!**

---

## Step 4 — Verify

1. Wait 2‑3 minutes for the build.
2. Your app will be live at `https://YOUR_USER-polybot-app-XXXXX.streamlit.app`.
3. Open it on your phone — the dashboard is fully responsive.
4. Click **▶ Start** to begin paper trading.

---

## Step 5 — Custom Domain (Optional)

Streamlit supports custom subdomains:

- Go to **Settings → General → Custom subdomain**.
- Enter something like `polybot`.
- Your URL becomes `https://polybot.streamlit.app`.

---

## Keeping It Free

| Resource | Free Tier |
|----------|-----------|
| Streamlit Community Cloud | Unlimited public apps, 1 private app |
| GitHub | Unlimited private repos |
| Anthropic API | Pay‑per‑use (≈ $0.003 per signal check with Sonnet) |
| Polymarket CLOB | Free API, no trading fees on limit orders |

The only variable cost is Claude API calls.  At ~30 scans/hour with 20 markets,
a signal fires on a small minority — expect **< $1/day** in API costs.

---

## Updating

Push changes to `main` and Streamlit auto‑redeploys:

```bash
git add -A
git commit -m "update strategy params"
git push
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App crashes on start | Check **Manage app → Logs** on Streamlit Cloud |
| `py-clob-client` install fails | Add `--extra-index-url` or pin version in `requirements.txt` |
| Claude errors | Verify `ANTHROPIC_API_KEY` in Secrets and billing is active |
| No markets appear | Polymarket API may be rate‑limiting; wait and retry |

---

That's it — your PolyBot dashboard is live, free, and accessible from
any device. 🚀
