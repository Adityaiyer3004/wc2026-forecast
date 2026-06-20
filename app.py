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
from flask import Flask, jsonify, send_file, abort
from dotenv import load_dotenv
load_dotenv()

from fetch import fetch_results
from wc2026_sim import run_with_results
from squad import build_squad_adjustments

app = Flask(__name__)

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: dict = {"probs": None, "matches": [], "updated_at": None, "ts": 0.0}
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

        # 3. Monte Carlo (50k for scheduled runs, tunable via env var)
        n_sims = int(os.environ.get("N_SIMS", "50000"))
        raw_probs = run_with_results(matches, n=n_sims, squad_adjustments=squad_adj)

        probs = {t: round(p * 100, 2) for t, p in raw_probs.items() if p >= 0.001}
        match_list = [
            {
                "home": m["home"], "away": m["away"],
                "home_goals": m["home_goals"], "away_goals": m["away_goals"],
                "date": m.get("date", ""),
            }
            for m in matches
        ]

        with _lock:
            _cache["probs"] = probs
            _cache["matches"] = match_list
            _cache["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _cache["ts"] = time.time()

        app.logger.info(f"Cache refreshed: {len(matches)} matches, {n_sims:,} sims")

    finally:
        _refreshing = False


def _get_or_refresh() -> dict:
    """Return cached data, refreshing synchronously if stale."""
    with _lock:
        age = time.time() - _cache["ts"]
        ready = _cache["probs"] is not None and age < CACHE_TTL

    if not ready:
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
        "probs":       data["probs"],
        "matches":     data["matches"],
        "match_count": len(data["matches"]),
        "updated_at":  data["updated_at"],
    })


@app.route("/refresh", methods=["POST"])
def refresh():
    """Triggered by Cloud Scheduler — runs a fresh simulation."""
    _run_simulation()
    with _lock:
        return jsonify({"ok": True, "updated_at": _cache["updated_at"],
                        "match_count": len(_cache["matches"])})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Local dev: pre-warm cache on startup
    threading.Thread(target=_run_simulation, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
