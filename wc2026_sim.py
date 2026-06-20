#!/usr/bin/env python3
"""
2026 FIFA World Cup — Monte Carlo winner forecast
Method: Poisson goal model + quality-weighted Bayesian update + 50k simulations.

Can be run standalone (uses hardcoded results) or imported by engine.py
which feeds it live ESPN data.
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
import copy

BASE = 1.35  # avg international goals per team per 90 mins

# Pre-tournament attack/defense ratings from betting market implied probs.
# Formula: xG_home = BASE * atk_home * def_away  (average team = 1.0 / 1.0)
TEAMS = {
    # Group A
    "Mexico":        {"atk": 1.28, "def": 0.85, "group": "A"},
    "South Korea":   {"atk": 1.05, "def": 0.96, "group": "A"},
    "Czechia":       {"atk": 1.05, "def": 0.96, "group": "A"},
    "South Africa":  {"atk": 0.85, "def": 1.12, "group": "A"},
    # Group B
    "Canada":        {"atk": 1.22, "def": 0.88, "group": "B"},
    "Switzerland":   {"atk": 1.18, "def": 0.88, "group": "B"},
    "Bosnia":        {"atk": 0.95, "def": 1.05, "group": "B"},
    "Qatar":         {"atk": 0.72, "def": 1.22, "group": "B"},
    # Group C
    "Brazil":        {"atk": 1.62, "def": 0.70, "group": "C"},
    "Morocco":       {"atk": 1.22, "def": 0.82, "group": "C"},
    "Scotland":      {"atk": 0.95, "def": 1.00, "group": "C"},
    "Haiti":         {"atk": 0.68, "def": 1.28, "group": "C"},
    # Group D
    "USA":           {"atk": 1.30, "def": 0.88, "group": "D"},
    "Australia":     {"atk": 1.05, "def": 0.96, "group": "D"},
    "Turkey":        {"atk": 1.12, "def": 0.92, "group": "D"},
    "Paraguay":      {"atk": 0.90, "def": 1.05, "group": "D"},
    # Group E
    "Germany":       {"atk": 1.62, "def": 0.72, "group": "E"},
    "Ivory Coast":   {"atk": 1.15, "def": 0.92, "group": "E"},
    "Ecuador":       {"atk": 1.00, "def": 1.00, "group": "E"},
    "Curacao":       {"atk": 0.68, "def": 1.32, "group": "E"},
    # Group F
    "Netherlands":   {"atk": 1.52, "def": 0.76, "group": "F"},
    "Japan":         {"atk": 1.18, "def": 0.90, "group": "F"},
    "Sweden":        {"atk": 1.22, "def": 0.88, "group": "F"},
    "Tunisia":       {"atk": 0.82, "def": 1.10, "group": "F"},
    # Group G
    "Belgium":       {"atk": 1.42, "def": 0.78, "group": "G"},
    "Egypt":         {"atk": 0.95, "def": 1.00, "group": "G"},
    "Iran":          {"atk": 0.90, "def": 1.02, "group": "G"},
    "New Zealand":   {"atk": 0.78, "def": 1.14, "group": "G"},
    # Group H
    "Spain":         {"atk": 1.78, "def": 0.66, "group": "H"},
    "Uruguay":       {"atk": 1.28, "def": 0.86, "group": "H"},
    "Saudi Arabia":  {"atk": 0.90, "def": 1.02, "group": "H"},
    "Cabo Verde":    {"atk": 0.72, "def": 1.20, "group": "H"},
    # Group I
    "France":        {"atk": 1.72, "def": 0.68, "group": "I"},
    "Norway":        {"atk": 1.35, "def": 0.84, "group": "I"},
    "Senegal":       {"atk": 1.10, "def": 0.93, "group": "I"},
    "Iraq":          {"atk": 0.72, "def": 1.24, "group": "I"},
    # Group J
    "Argentina":     {"atk": 1.72, "def": 0.68, "group": "J"},
    "Austria":       {"atk": 1.22, "def": 0.88, "group": "J"},
    "Jordan":        {"atk": 0.78, "def": 1.14, "group": "J"},
    "Algeria":       {"atk": 0.84, "def": 1.08, "group": "J"},
    # Group K
    "Portugal":      {"atk": 1.68, "def": 0.70, "group": "K"},
    "Colombia":      {"atk": 1.28, "def": 0.86, "group": "K"},
    "DR Congo":      {"atk": 0.90, "def": 1.02, "group": "K"},
    "Uzbekistan":    {"atk": 0.78, "def": 1.14, "group": "K"},
    # Group L
    "England":       {"atk": 1.62, "def": 0.72, "group": "L"},
    "Croatia":       {"atk": 1.22, "def": 0.88, "group": "L"},
    "Ghana":         {"atk": 0.95, "def": 1.00, "group": "L"},
    "Panama":        {"atk": 0.72, "def": 1.20, "group": "L"},
}

# Co-hosts get a crowd/travel advantage in their group-stage home games
HOME_ADVANTAGE_TEAMS = {"USA", "Canada", "Mexico"}
HOME_ADVANTAGE_FACTOR = 1.12  # +12% to attack when playing at home venue

GROUPS = defaultdict(list)
for _name, _info in TEAMS.items():
    GROUPS[_info["group"]].append(_name)


def bayesian_update(teams: dict, matches: list, lr: float = 0.08) -> dict:
    """
    Update attack/defense ratings from match results.

    `matches` can be either:
      - simple tuples: (home, away, home_goals, away_goals)
      - rich dicts from fetch.py with effective_home/effective_away and red card counts

    Uses effective goals (penalties/own-goals discounted) when available.
    Scales the update down when red cards made the scoreline misleading.
    """
    t = copy.deepcopy(teams)

    for m in matches:
        if isinstance(m, dict):
            home, away = m["home"], m["away"]
            # Use effective goals (quality-weighted) for the Bayesian signal
            gh = m.get("effective_home", m["home_goals"])
            ga = m.get("effective_away", m["away_goals"])
            # Red cards distort scorelines — discount the update
            total_red = m.get("home_red_cards", 0) + m.get("away_red_cards", 0)
            red_scale = 1.0 if total_red == 0 else (0.7 if total_red == 1 else 0.35)
        else:
            home, away, gh, ga = m
            red_scale = 1.0

        if home not in t or away not in t:
            continue

        mu_h = BASE * t[home]["atk"] * t[away]["def"]
        mu_a = BASE * t[away]["atk"] * t[home]["def"]

        effective_lr = lr * red_scale

        # Gradient step on log scale: nudge toward what this scoreline implies
        t[home]["atk"] = max(0.3, min(3.5, t[home]["atk"] * (1 + effective_lr * (gh - mu_h) / mu_h)))
        t[away]["def"] = max(0.3, min(3.5, t[away]["def"] * (1 + effective_lr * (gh - mu_h) / mu_h)))
        t[away]["atk"] = max(0.3, min(3.5, t[away]["atk"] * (1 + effective_lr * (ga - mu_a) / mu_a)))
        t[home]["def"] = max(0.3, min(3.5, t[home]["def"] * (1 + effective_lr * (ga - mu_a) / mu_a)))

    return t


def _xg(t1: str, t2: str, teams: dict, t1_is_host: bool = False) -> tuple[float, float]:
    atk1 = teams[t1]["atk"] * (HOME_ADVANTAGE_FACTOR if t1_is_host else 1.0)
    mu1 = max(0.1, BASE * atk1 * teams[t2]["def"])
    mu2 = max(0.1, BASE * teams[t2]["atk"] * teams[t1]["def"])
    return mu1, mu2


def sim_match(t1: str, t2: str, teams: dict, t1_is_host: bool = False) -> tuple[int, int]:
    mu1, mu2 = _xg(t1, t2, teams, t1_is_host)
    return np.random.poisson(mu1), np.random.poisson(mu2)


def knockout_winner(t1: str, t2: str, teams: dict) -> str:
    """Simulate a knockout match; draw → weighted penalty shootout."""
    g1, g2 = sim_match(t1, t2, teams)
    if g1 > g2:
        return t1
    if g2 > g1:
        return t2
    s1 = teams[t1]["atk"] / teams[t1]["def"]
    s2 = teams[t2]["atk"] / teams[t2]["def"]
    return t1 if np.random.random() < s1 / (s1 + s2) else t2


def sim_group(group_teams: list, teams: dict, played: list) -> tuple[list, dict, dict]:
    """Simulate remaining group matches; return final standings."""
    played_pairs = {frozenset([h, a]) for m in played for h, a in [(
        (m["home"], m["away"]) if isinstance(m, dict) else (m[0], m[1])
    )]}

    pts = defaultdict(int)
    gd  = defaultdict(int)
    gf  = defaultdict(int)

    for m in played:
        if isinstance(m, dict):
            h, a, gh, ga = m["home"], m["away"], m["home_goals"], m["away_goals"]
        else:
            h, a, gh, ga = m

        if h in group_teams and a in group_teams:
            gf[h] += gh; gf[a] += ga
            gd[h] += gh - ga; gd[a] += ga - gh
            if gh > ga:   pts[h] += 3
            elif ga > gh: pts[a] += 3
            else:         pts[h] += 1; pts[a] += 1

    for t1, t2 in combinations(group_teams, 2):
        if frozenset([t1, t2]) not in played_pairs:
            t1_host = t1 in HOME_ADVANTAGE_TEAMS
            g1, g2 = sim_match(t1, t2, teams, t1_is_host=t1_host)
            gf[t1] += g1; gf[t2] += g2
            gd[t1] += g1 - g2; gd[t2] += g2 - g1
            if g1 > g2:   pts[t1] += 3
            elif g2 > g1: pts[t2] += 3
            else:         pts[t1] += 1; pts[t2] += 1

    standings = sorted(
        group_teams,
        key=lambda t: (pts[t], gd[t], gf[t], np.random.random()),
        reverse=True,
    )
    return standings, pts, gd


def sim_tournament_once(teams: dict, group_played: dict) -> str:
    qualifiers = []
    all_third = []

    for grp in sorted(GROUPS.keys()):
        group_teams = GROUPS[grp]
        played = group_played.get(grp, [])
        standings, pts, gd = sim_group(group_teams, teams, played)
        qualifiers.extend(standings[:2])
        all_third.append((pts[standings[2]], gd[standings[2]], standings[2]))

    # Best 8 of 12 third-place teams advance
    all_third.sort(key=lambda x: (x[0], x[1]), reverse=True)
    qualifiers.extend(t[2] for t in all_third[:8])

    # 32-team knockout (5 rounds)
    bracket = list(qualifiers)
    np.random.shuffle(bracket)
    while len(bracket) > 1:
        next_rd = []
        for i in range(0, len(bracket), 2):
            next_rd.append(knockout_winner(bracket[i], bracket[i + 1], teams))
        bracket = next_rd

    return bracket[0]


def run_with_results(
    matches: list,
    n: int = 50_000,
    seed: int = 2026,
    squad_adjustments: dict[str, float] | None = None,
    use_live_ratings: bool = True,
) -> dict[str, float]:
    """
    Core simulation. Returns {team: win_probability} dict.

    `matches` can be tuples or dicts from fetch.py.
    `squad_adjustments` is an optional {team: multiplier} from squad.py that
    scales each team's attack rating before the Bayesian update is applied.
    `use_live_ratings` fetches current Elo ratings to replace hardcoded priors.
    """
    np.random.seed(seed)

    # Layer 1: start from live Elo ratings (falls back to hardcoded if unavailable)
    if use_live_ratings:
        try:
            from ratings import apply_elo_ratings
            base_teams = apply_elo_ratings(TEAMS)
        except Exception as e:
            print(f"[sim] Elo fetch failed ({e}), using hardcoded ratings")
            base_teams = TEAMS
    else:
        base_teams = TEAMS

    # Layer 2: squad/form/manager multipliers from squad.py
    if squad_adjustments:
        import copy
        base_teams = copy.deepcopy(base_teams)
        for team, mult in squad_adjustments.items():
            if team in base_teams:
                base_teams[team]["atk"] = round(base_teams[team]["atk"] * mult, 4)
                # Better squads also defend marginally better
                base_teams[team]["def"] = round(base_teams[team]["def"] / (1 + (mult - 1) * 0.4), 4)

    updated = bayesian_update(base_teams, matches)

    group_played: dict[str, list] = defaultdict(list)  # type: ignore[assignment]
    for m in matches:
        home = m["home"] if isinstance(m, dict) else m[0]
        if home in TEAMS:
            grp = TEAMS[home]["group"]
            group_played[grp].append(m)

    wins: dict[str, int] = defaultdict(int)
    for _ in range(n):
        wins[sim_tournament_once(updated, group_played)] += 1

    return {t: wins[t] / n for t in TEAMS}


def run(n: int = 50_000) -> dict[str, float]:
    """Standalone entry point using hardcoded results (no network)."""
    RESULTS = [
        ("Mexico","South Africa",2,0), ("South Korea","Czechia",2,1),
        ("Czechia","South Africa",1,1), ("Mexico","South Korea",1,0),
        ("Canada","Bosnia",1,1), ("Qatar","Switzerland",1,1),
        ("Switzerland","Bosnia",4,1), ("Canada","Qatar",6,0),
        ("Brazil","Morocco",1,1), ("Scotland","Haiti",1,0),
        ("USA","Paraguay",4,1), ("Australia","Turkey",2,0), ("USA","Australia",2,0),
        ("Germany","Curacao",7,1), ("Ivory Coast","Ecuador",1,0),
        ("Netherlands","Japan",2,2), ("Sweden","Tunisia",5,1),
        ("Belgium","Egypt",1,1), ("Iran","New Zealand",2,2),
        ("Spain","Cabo Verde",0,0), ("Saudi Arabia","Uruguay",1,1),
        ("France","Senegal",3,1), ("Norway","Iraq",4,1),
        ("Argentina","Algeria",3,0), ("Austria","Jordan",3,1),
        ("Portugal","DR Congo",1,1), ("Colombia","Uzbekistan",3,1),
        ("England","Croatia",4,2), ("Ghana","Panama",1,0),
        ("Scotland","Morocco",0,1),
    ]
    probs = run_with_results(RESULTS, n=n)
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    print(f"\n2026 FIFA World Cup — Championship Probabilities")
    print(f"  ({n:,} Monte Carlo simulations | quality-weighted Bayesian | home advantage for co-hosts)\n")
    print(f"{'Team':<20} {'Win %':>7}  Bar")
    print("─" * 52)
    for team, p in ranked:
        if p >= 0.004:
            bar = "█" * int(p * 100)
            print(f"  {team:<18} {p*100:>5.1f}%  {bar}")

    return {t: round(p * 100, 2) for t, p in ranked}


if __name__ == "__main__":
    run()
