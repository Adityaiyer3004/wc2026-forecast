"""
Live WC 2026 match data from ESPN's public API.
Returns normalized results with goal-quality metadata for the model.
"""

import urllib.request
import json

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20260601-20261231"

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
    Returns completed matches as a list of dicts:
      home, away, home_goals, away_goals,
      effective_home_goals, effective_away_goals,  # penalties/OGs discounted
      home_red_cards, away_red_cards,
      date
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

        # Parse match events for goal quality and discipline
        details = comp.get("details", [])
        home_team_id = home.get("team", {}).get("id", "")
        away_team_id = away.get("team", {}).get("id", "")

        h_eff = 0.0  # effective goals (signal for Bayesian update)
        a_eff = 0.0
        h_red = 0
        a_red = 0

        for event_detail in details:
            etype = event_detail.get("type", {}).get("text", "")
            team_id = event_detail.get("team", {}).get("id", "")
            is_home = team_id == home_team_id

            if event_detail.get("redCard"):
                if is_home:
                    h_red += 1
                else:
                    a_red += 1

            if event_detail.get("scoringPlay"):
                own_goal = event_detail.get("ownGoal", False)
                penalty = event_detail.get("penaltyKick", False)
                # Weight: open play = 1.0, header = 1.0, penalty = 0.6, own goal = 0.0 (luck)
                weight = 0.0 if own_goal else (0.6 if penalty else 1.0)
                # Credit goes to the team that scored (own goals credit the opposition)
                if own_goal:
                    if is_home:
                        a_eff += weight  # own goal by home = credit to away (weight=0)
                    else:
                        h_eff += weight
                else:
                    if is_home:
                        h_eff += weight
                    else:
                        a_eff += weight

        # If details unavailable (shouldn't happen with ESPN), fall back to raw score
        if not details:
            h_eff = float(h_score)
            a_eff = float(a_score)

        matches.append({
            "home":               h_name,
            "away":               a_name,
            "home_goals":         h_score,
            "away_goals":         a_score,
            "effective_home":     h_eff,
            "effective_away":     a_eff,
            "home_red_cards":     h_red,
            "away_red_cards":     a_red,
            "date":               event.get("date", "")[:10],
        })

    return matches


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
