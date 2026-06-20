"""
WC 2026 Forecast — Flask web server for Cloud Run.

Routes:
  GET  /           → HTML dashboard (fetches /api/odds on load)
  GET  /api/odds   → JSON probabilities (cached, refreshes if stale)
  POST /refresh    → force re-run (called by Cloud Scheduler every 6h)
  GET  /health     → Cloud Run health check
"""

import os
import time
import threading
from flask import Flask, jsonify, send_file, request, abort
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

from fetch import fetch_results, fetch_qualifying_results
from wc2026_sim import run_with_results
from squad import build_squad_adjustments

app = Flask(__name__)

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: dict = {
    "probs": None, "prev_probs": None,
    "stage_probs": None, "fixtures": None, "shifts": [],
    "matches": [], "updated_at": None, "ts": 0.0,
}
_lock = threading.Lock()
_refreshing = False
CACHE_TTL = 3600  # seconds; Cloud Scheduler refreshes every 6h anyway


def _run_simulation() -> None:
    """Full pipeline: fetch → squad → simulate → write cache."""
    global _refreshing
    _refreshing = True
    try:
        # 1. Live match data
        try:
            matches = fetch_results()
        except Exception as e:
            app.logger.warning(f"ESPN fetch failed: {e}. Using empty results.")
            matches = []

        # 2. Squad intelligence
        groq_key   = os.environ.get("GROQ_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")
        try:
            squad_adj = build_squad_adjustments(groq_key=groq_key, tavily_key=tavily_key)
        except Exception as e:
            app.logger.warning(f"Squad layer failed: {e}. Skipping.")
            squad_adj = None

        # 3. Qualifying results (7-day cache)
        try:
            qualifying = fetch_qualifying_results()
        except Exception as e:
            app.logger.warning(f"Qualifying fetch failed: {e}")
            qualifying = []

        # Snapshot previous probs for shift detection before overwriting
        with _lock:
            old_probs = _cache.get("probs")

        # 4. Monte Carlo (50k for scheduled runs, tunable via env var)
        n_sims = int(os.environ.get("N_SIMS", "50000"))
        raw = run_with_results(
            matches, n=n_sims, squad_adjustments=squad_adj,
            qualifying_matches=qualifying or None,
            return_stages=True,
        )
        win_probs    = raw["win"]
        stage_probs  = raw["stage_probs"]
        fixtures     = raw["fixtures"]

        probs = {t: round(p * 100, 2) for t, p in win_probs.items() if p >= 0.001}
        stage_probs_pct = {
            t: {s: round(v * 100, 1) for s, v in sp.items()}
            for t, sp in stage_probs.items()
        }

        # Odds shifts: teams that moved ≥2pp since the last run
        shifts = []
        if old_probs:
            for team, curr in probs.items():
                prev = old_probs.get(team, 0.0)
                delta = curr - prev
                if abs(delta) >= 2.0:
                    shifts.append({"team": team, "prev": round(prev, 2),
                                   "curr": curr, "delta": round(delta, 2)})
            shifts.sort(key=lambda x: -abs(x["delta"]))

        match_list = [
            {
                "home": m["home"], "away": m["away"],
                "home_goals": m["home_goals"], "away_goals": m["away_goals"],
                "date": m.get("date", ""),
            }
            for m in matches
        ]

        with _lock:
            _cache["prev_probs"]  = old_probs
            _cache["probs"]       = probs
            _cache["stage_probs"] = stage_probs_pct
            _cache["fixtures"]    = fixtures
            _cache["shifts"]      = shifts
            _cache["matches"]     = match_list
            _cache["updated_at"]  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _cache["ts"]          = time.time()

        app.logger.info(f"Cache refreshed: {len(matches)} matches, {n_sims:,} sims, "
                        f"{len(fixtures)} upcoming fixtures, {len(shifts)} movers")

    finally:
        _refreshing = False


def _get_or_refresh() -> dict:
    """Return cached data, refreshing synchronously if stale."""
    with _lock:
        age = time.time() - _cache["ts"]
        ready = _cache["probs"] is not None and age < CACHE_TTL

    if not ready:
        if _refreshing:
            # Background warmup already in progress — wait for it instead of
            # spawning a second simulation that would race on the cache.
            while _refreshing:
                time.sleep(0.5)
        else:
            _run_simulation()

    with _lock:
        return dict(_cache)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("wc2026_forecast.html")


@app.route("/api/odds")
def api_odds():
    data = _get_or_refresh()
    return jsonify({
        "probs":        data["probs"],
        "stage_probs":  data.get("stage_probs"),
        "fixtures":     data.get("fixtures"),
        "shifts":       data.get("shifts", []),
        "matches":      data["matches"],
        "match_count":  len(data["matches"]),
        "updated_at":   data["updated_at"],
    })


@app.route("/refresh", methods=["POST"])
def refresh():
    """Triggered by Cloud Scheduler — runs a fresh simulation."""
    _run_simulation()
    with _lock:
        return jsonify({"ok": True, "updated_at": _cache["updated_at"],
                        "match_count": len(_cache["matches"])})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return jsonify({"error": "Chat unavailable — GROQ_API_KEY not set"}), 503

    body = request.get_json(silent=True) or {}
    user_msg = (body.get("message") or "").strip()
    history  = body.get("history") or []   # list of {role, content}
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    with _lock:
        probs       = _cache.get("probs") or {}
        stage_probs = _cache.get("stage_probs") or {}
        fixtures    = _cache.get("fixtures") or []
        shifts      = _cache.get("shifts") or []
        matches     = _cache.get("matches") or []
        updated_at  = _cache.get("updated_at") or "unknown"

    # Build a compact forecast summary for the system prompt
    top10 = sorted(probs.items(), key=lambda x: -x[1])[:10]
    top10_str = "\n".join(f"  {t}: {p:.1f}%" for t, p in top10)

    recent = matches[-8:] if matches else []
    results_str = "\n".join(
        f"  {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']}"
        for m in recent
    ) or "  (none yet)"

    fixtures_str = "\n".join(
        f"  {f['home']} vs {f['away']} (Group {f['group']}): "
        f"W {f['home_win']*100:.0f}% / D {f['draw']*100:.0f}% / L {f['away_win']*100:.0f}%"
        for f in (fixtures or [])[:8]
    ) or "  (none)"

    shifts_str = "\n".join(
        f"  {s['team']}: {s['prev']:.1f}% → {s['curr']:.1f}% ({'+' if s['delta']>0 else ''}{s['delta']:.1f}pp)"
        for s in (shifts or [])[:5]
    ) or "  (none)"

    system = f"""You are a football analyst assistant for the WC 2026 Forecast dashboard.
You have access to live Monte Carlo simulation data (50,000 simulations, 5-layer Bayesian model).
Today's date: 2026-06-20. Last updated: {updated_at}.

TOP 10 CHAMPIONSHIP WIN PROBABILITIES:
{top10_str}

RECENT MATCH RESULTS (last 8):
{results_str}

UPCOMING FIXTURES (model predictions):
{fixtures_str}

NOTABLE ODDS SHIFTS SINCE LAST RUN:
{shifts_str}

Answer questions about the tournament, probabilities, and model predictions.
Be concise (2-4 sentences max unless asked for detail). Use the data above.
If asked about a team not in the top 10, you can still discuss them based on general knowledge.
Never make up specific probabilities you don't have — say "I don't have that breakdown" instead."""

    messages = []
    for h in history[-6:]:   # keep last 6 turns for context
        role = h.get("role")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_msg})

    try:
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=300,
            temperature=0.6,
        )
        reply = resp.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        app.logger.error(f"Groq chat error: {e}")
        return jsonify({"error": "Chat failed, try again"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# Pre-warm cache on module import so gunicorn workers don't block the first
# HTTP request waiting for a 50 k-sim cold start.
threading.Thread(target=_run_simulation, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
