# WC 2026 Winner Forecast

A live Monte Carlo tournament simulator for the 2026 FIFA World Cup, deployed on Google Cloud Run.

**Live demo → https://wc2026-forecast-117489527781.us-central1.run.app**

---

## What it does

Runs 50,000 simulations of the remainder of the tournament every 6 hours and serves a live dashboard showing:

- Championship win probabilities for all 48 teams
- Stage-by-stage breakdown (R32 → R16 → QF → SF → Final → Win)
- W/D/L predictions for every upcoming group fixture
- Live match results (polls ESPN every 5 minutes)
- Group standings, biggest upsets, and odds shifts
- AI chatbot backed by the live forecast data

---

## How the model works

A 5-layer Bayesian pipeline feeds into a Poisson goal model (`λ = BASE × atk × def`):

| Layer | Source | Learning rate |
|---|---|---|
| 1. Live Elo ratings | eloratings.net (244 teams) | Prior |
| 2. Squad / pedigree | StatsBomb xG, WC history, manager score | ±10% multiplier |
| 3. 392 qualifying matches | ESPN (all 6 confederations) | lr = 0.04 |
| 4. WC 2026 live results | ESPN scoreboard API | lr = 0.08 |
| 5. Form + injuries (AI) | Tavily web search → Groq LLaMA 3.3 70B | 0.90–1.04× |

Each simulation plays out all remaining group matches (Poisson draws), advances the top 2 + 8 best third-place teams to the 32-team knockout, then runs the bracket.

### Validation

Backtested against WC 2022 (`backtest_2022.py`). Argentina was ranked #3 pre-tournament at 11.8%, correctly collapsed to 3.4% after the Saudi Arabia upset, recovered through the group stage, and reached 49.8% heading into the Final. Brier score beat the random baseline at all 7 checkpoints.

---

## Stack

| Component | Tech |
|---|---|
| Simulation | Python · NumPy · Poisson |
| Web server | Flask · Gunicorn |
| Frontend | Vanilla JS · CSS animations |
| AI layer | Groq (LLaMA 3.3 70B) · Tavily |
| Live data | ESPN public scoreboard API |
| Deployment | Google Cloud Run · Cloud Scheduler |

---

## Running locally

```bash
git clone https://github.com/Adityaiyer3004/wc2026-forecast.git
cd wc2026-forecast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional — needed for the AI form/injury layer and chatbot
export GROQ_API_KEY="..."
export TAVILY_API_KEY="..."

python app.py
# → http://localhost:8080
```

The server pre-warms the cache in a background thread on startup. The first page load may take 30–60 seconds while the simulation runs.

### Run the WC 2022 backtest

```bash
python backtest_2022.py
```

Prints Argentina's probability trajectory across 7 tournament checkpoints and Brier scores vs. a random baseline.

---

## Deploying to Cloud Run

```bash
gcloud run deploy wc2026-forecast \
  --source . \
  --region us-central1 \
  --set-env-vars GROQ_API_KEY="...",TAVILY_API_KEY="..." \
  --timeout 600
```

Schedule a refresh every 6 hours via Cloud Scheduler pointing `POST` at `/refresh`.

---

## Key files

| File | Purpose |
|---|---|
| `wc2026_sim.py` | Monte Carlo engine — Poisson model, Bayesian update, tournament bracket |
| `fetch.py` | ESPN API client — live WC results + 392 qualifying matches |
| `squad.py` | AI squad layer — Tavily search + Groq scoring for all 48 teams |
| `ratings.py` | Elo ratings fetcher (eloratings.net) |
| `app.py` | Flask server — cache management, `/api/odds`, `/api/scores`, `/api/chat` |
| `wc2026_forecast.html` | Single-file dashboard — animations, charts, chatbot |
| `backtest_2022.py` | WC 2022 validation — 7-checkpoint Brier score analysis |

---

## API endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Dashboard HTML |
| `/api/odds` | GET | Full forecast — probabilities, stage breakdown, fixtures (1h cache) |
| `/api/scores` | GET | Live match scores from ESPN (no cache, ~2s) |
| `/api/chat` | POST | AI chatbot with live forecast context |
| `/refresh` | POST | Force re-simulation (called by Cloud Scheduler) |
| `/health` | GET | Health check |

---

*Analysis only — not betting advice.*
