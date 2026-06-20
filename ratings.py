"""
Live team ratings from World Football Elo (eloratings.net — free, no key).

Replaces the hardcoded TEAMS atk/def values in wc2026_sim.py with data-driven
ratings derived from current Elo points.

Conversion (calibrated against WC 2022 betting priors):
  atk =  0.001532 * elo - 1.481   clamped [0.60, 2.00]
  def = -0.000809 * elo + 2.383   clamped [0.55, 1.40]
Calibration anchors: Spain (2129 → 1.78 atk / 0.66 def)
                     Qatar  (1437 → 0.72 atk / 1.22 def)
"""

import urllib.request
import json
import time
from pathlib import Path

CACHE_FILE = Path(__file__).parent / ".squad_cache" / "elo_ratings.json"
CACHE_TTL  = 86400  # 24 hours — rankings update monthly

# eloratings.net country code → our team name
ELO_CODE_MAP = {
    "AR": "Argentina",
    "AT": "Austria",
    "AU": "Australia",
    "BA": "Bosnia",
    "BE": "Belgium",
    "BR": "Brazil",
    "CA": "Canada",
    "CD": "DR Congo",
    "CH": "Switzerland",
    "CI": "Ivory Coast",
    "CO": "Colombia",
    "CW": "Curacao",
    "CV": "Cabo Verde",
    "CZ": "Czechia",
    "DE": "Germany",
    "DZ": "Algeria",
    "EC": "Ecuador",
    "EG": "Egypt",
    "EN": "England",
    "ES": "Spain",
    "FR": "France",
    "GH": "Ghana",
    "HR": "Croatia",
    "HT": "Haiti",
    "IQ": "Iraq",
    "IR": "Iran",
    "JP": "Japan",
    "JO": "Jordan",
    "KR": "South Korea",
    "MA": "Morocco",
    "MX": "Mexico",
    "NL": "Netherlands",
    "NO": "Norway",
    "NZ": "New Zealand",
    "PA": "Panama",
    "PT": "Portugal",
    "PY": "Paraguay",
    "QA": "Qatar",
    "SA": "Saudi Arabia",
    "SE": "Sweden",
    "SN": "Senegal",
    "SQ": "Scotland",
    "TN": "Tunisia",
    "TR": "Turkey",
    "US": "USA",
    "UY": "Uruguay",
    "UZ": "Uzbekistan",
    "ZA": "South Africa",
}


def _elo_to_atk(elo: float) -> float:
    return round(max(0.60, min(2.00,  0.001532 * elo - 1.481)), 4)


def _elo_to_def(elo: float) -> float:
    return round(max(0.55, min(1.40, -0.000809 * elo + 2.383)), 4)


def fetch_elo_ratings() -> dict[str, dict]:
    """
    Fetch current Elo ratings from eloratings.net and convert to atk/def.
    Returns {team: {"atk": float, "def": float, "elo": int}}.
    Cached for 24 hours.
    """
    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text())
        if time.time() - cached.get("_ts", 0) < CACHE_TTL:
            return {k: v for k, v in cached.items() if k != "_ts"}

    print("[ratings] Fetching Elo ratings from eloratings.net...", end=" ", flush=True)
    req = urllib.request.Request(
        "https://www.eloratings.net/World.tsv",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        tsv = r.read().decode("utf-8")

    ratings: dict[str, dict] = {}
    for line in tsv.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        code = parts[2].strip()
        try:
            elo = int(parts[3].strip())
        except ValueError:
            continue
        team = ELO_CODE_MAP.get(code)
        if team:
            ratings[team] = {
                "atk": _elo_to_atk(elo),
                "def": _elo_to_def(elo),
                "elo": elo,
            }

    print(f"OK — {len(ratings)} WC 2026 teams rated")
    CACHE_FILE.write_text(json.dumps({**ratings, "_ts": time.time()}))
    return ratings


def apply_elo_ratings(teams: dict) -> dict:
    """
    Overlay live Elo-derived atk/def onto the TEAMS dict.
    Keeps group assignments; falls back to hardcoded values for any missing team.
    """
    import copy
    live = fetch_elo_ratings()
    updated = copy.deepcopy(teams)
    for team, data in live.items():
        if team in updated:
            updated[team]["atk"] = data["atk"]
            updated[team]["def"] = data["def"]
    return updated


if __name__ == "__main__":
    ratings = fetch_elo_ratings()
    ranked = sorted(ratings.items(), key=lambda x: -x[1]["elo"])
    print(f"\n{'Team':<22} {'Elo':>5}  {'atk':>6}  {'def':>6}")
    print("─" * 46)
    for team, r in ranked:
        print(f"  {team:<20} {r['elo']:>5}  {r['atk']:>6.3f}  {r['def']:>6.3f}")
