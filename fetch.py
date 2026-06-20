"""
Live WC 2026 match data + qualifying results from ESPN's public API.
Returns normalized results with goal-quality metadata for the model.
"""

import urllib.request
import json
import time
from pathlib import Path

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20260601-20261231"

# Qualifying competitions by confederation
QUALIFIER_SLUGS = [
    "fifa.worldq.conmebol",
    "fifa.worldq.uefa",
    "fifa.worldq.concacaf",
    "fifa.worldq.caf",
    "fifa.worldq.afc",
    "fifa.worldq.ofc",
]
# Qualifying ran from mid-2023 through early 2026
QUALIFIER_WINDOWS = [
    "20230601-20231231",
    "20240101-20240630",
    "20240701-20241231",
    "20250101-20260601",
]

QUAL_CACHE = Path(__file__).parent / ".squad_cache" / "qualifying.json"
QUAL_CACHE_TTL = 86400 * 7  # 7 days — historical results don't change

# ESPN name → model name
NAME_MAP = {
    "United States":       "USA",
    "Bosnia-Herzegovina":  "Bosnia",
    "Congo DR":            "DR Congo",
    "Cape Verde":          "Cabo Verde",
    "Curaçao":             "Curacao",
    "Türkiye":             "Turkey",
    "Ivory Coast":         "Ivory Coast",
    "South Korea":         "South Korea",
    "Saudi Arabia":        "Saudi Arabia",
    "New Zealand":         "New Zealand",
    "South Africa":        "South Africa",
    "DR Congo":            "DR Congo",
}

def _normalize(name: str) -> str:
    return NAME_MAP.get(name, name)

def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def fetch_results() -> list[dict]:
    """
    Returns completed WC 2026 matches with goal-quality metadata:
      home, away, home_goals, away_goals,
      effective_home/away (penalties 0.6×, own goals 0×),
      home_red_cards, away_red_cards, date
    """
    data = _fetch_json(ESPN_URL)
    matches = []

    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status not in ("STATUS_FULL_TIME", "STATUS_FINAL"):
            continue

        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})

        h_name = _normalize(home.get("team", {}).get("displayName", ""))
        a_name = _normalize(away.get("team", {}).get("displayName", ""))
        h_score = int(home.get("score", 0) or 0)
        a_score = int(away.get("score", 0) or 0)

        details = comp.get("details", [])
        home_team_id = home.get("team", {}).get("id", "")
        h_eff = a_eff = 0.0
        h_red = a_red = 0

        for d in details:
            team_id = d.get("team", {}).get("id", "")
            is_home = team_id == home_team_id
            if d.get("redCard"):
                if is_home: h_red += 1
                else: a_red += 1
            if d.get("scoringPlay"):
                own_goal = d.get("ownGoal", False)
                penalty  = d.get("penaltyKick", False)
                weight = 0.0 if own_goal else (0.6 if penalty else 1.0)
                if own_goal:
                    if is_home: a_eff += weight
                    else:       h_eff += weight
                else:
                    if is_home: h_eff += weight
                    else:       a_eff += weight

        if not details:
            h_eff, a_eff = float(h_score), float(a_score)

        matches.append({
            "home": h_name, "away": a_name,
            "home_goals": h_score, "away_goals": a_score,
            "effective_home": h_eff, "effective_away": a_eff,
            "home_red_cards": h_red, "away_red_cards": a_red,
            "date": event.get("date", "")[:10],
        })

    return matches


def _parse_events(data: dict) -> list[dict]:
    """Parse ESPN scoreboard events into match dicts (shared by WC + qualifying)."""
    matches = []
    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status not in ("STATUS_FULL_TIME", "STATUS_FINAL"):
            continue
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        h_name = _normalize(home.get("team", {}).get("displayName", ""))
        a_name = _normalize(away.get("team", {}).get("displayName", ""))
        if not h_name or not a_name:
            continue
        h_score = int(home.get("score", 0) or 0)
        a_score = int(away.get("score", 0) or 0)
        matches.append({
            "home": h_name, "away": a_name,
            "home_goals": h_score, "away_goals": a_score,
            "effective_home": float(h_score), "effective_away": float(a_score),
            "home_red_cards": 0, "away_red_cards": 0,
            "date": event.get("date", "")[:10],
        })
    return matches


def fetch_qualifying_results() -> list[dict]:
    """
    Fetch WC 2026 qualifying results from all 6 confederations via ESPN.
    Returns ~700+ completed matches. Cached for 7 days (historical data).
    Used as low-weight training data (lr=0.04) to calibrate base ratings.
    """
    if QUAL_CACHE.exists():
        cached = json.loads(QUAL_CACHE.read_text())
        if time.time() - cached.get("_ts", 0) < QUAL_CACHE_TTL:
            return cached.get("matches", [])

    print("[fetch] Fetching qualifying results (6 confederations × 4 windows)...", end=" ", flush=True)
    all_matches: list[dict] = []
    seen: set = set()  # deduplicate

    for slug in QUALIFIER_SLUGS:
        for window in QUALIFIER_WINDOWS:
            url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/"
                   f"{slug}/scoreboard?dates={window}")
            try:
                data = _fetch_json(url)
                for m in _parse_events(data):
                    key = (m["home"], m["away"], m["date"])
                    if key not in seen:
                        seen.add(key)
                        all_matches.append(m)
            except Exception:
                pass

    print(f"OK — {len(all_matches)} qualifying matches")
    QUAL_CACHE.write_text(json.dumps({"matches": all_matches, "_ts": time.time()}))
    return all_matches


def to_simple_results(matches: list[dict]) -> list[tuple]:
    """Convert to the (home, away, hg, ag) tuple format the model uses."""
    return [(m["home"], m["away"], m["home_goals"], m["away_goals"]) for m in matches]


if __name__ == "__main__":
    results = fetch_results()
    print(f"Fetched {len(results)} completed matches\n")
    for m in results:
        flag = ""
        if m["home_red_cards"] or m["away_red_cards"]:
            flag = f"  [RC: {m['home_red_cards']}/{m['away_red_cards']}]"
        eff = f"  (eff {m['effective_home']:.1f}-{m['effective_away']:.1f})" if (
            abs(m["effective_home"] - m["home_goals"]) > 0.1 or
            abs(m["effective_away"] - m["away_goals"]) > 0.1
        ) else ""
        print(f"  {m['date']}  {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']}{eff}{flag}")
