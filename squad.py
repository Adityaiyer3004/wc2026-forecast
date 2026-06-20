"""
Squad intelligence layer — free data only.

Sources:
  1. StatsBomb open data (GitHub, no key) — WC 2022 player xG, goals, shots per team
  2. WC round-reached pedigree (WC 2022 + 2018 stage points)
  3. Gemini free tier (aistudio.google.com key) — current club season form via Google Search

Output: per-team squad_multiplier that scales attack/defense ratings in the model.
"""

import urllib.request
import json
import os
from collections import defaultdict
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".squad_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── StatsBomb name → model name ───────────────────────────────────────────
SB_NAME_MAP = {
    "United States":      "USA",
    "Côte d'Ivoire":      "Ivory Coast",
    "Netherlands":        "Netherlands",
    "South Korea":        "South Korea",
    "Saudi Arabia":       "Saudi Arabia",
    "New Zealand":        "New Zealand",
    "South Africa":       "South Africa",
}

def _sb_normalize(name: str) -> str:
    return SB_NAME_MAP.get(name, name)


# ── WC 2022 pedigree (stage points: group=1, R16=2, QF=3, SF=4, F=5, W=7) ─
WC22_PEDIGREE = {
    "Argentina": 5, "France": 7, "Croatia": 4, "Morocco": 4,
    "England": 3, "Portugal": 3, "Netherlands": 3, "Brazil": 3,
    "Japan": 2, "South Korea": 2, "USA": 2, "Switzerland": 2,
    "Australia": 2, "Spain": 2, "Senegal": 2,
    "Germany": 1, "Belgium": 1, "Mexico": 1, "Uruguay": 1,
    "Ecuador": 1, "Iran": 1, "Ghana": 1, "Tunisia": 1,
    "Canada": 1, "Saudi Arabia": 1, "Qatar": 1,
}
# Teams not at WC 2022 get 0 — no pedigree signal
WC22_MAX = 7


# ── WC 2018 pedigree ──────────────────────────────────────────────────────
WC18_PEDIGREE = {
    "France": 7, "Croatia": 5, "Belgium": 4, "England": 4,
    "Uruguay": 3, "Brazil": 3, "Russia": 3, "Sweden": 3,
    "Argentina": 2, "Portugal": 2, "Spain": 2, "Denmark": 2,
    "Mexico": 2, "Japan": 2, "Colombia": 2, "Switzerland": 2,
    "Germany": 1, "Brazil": 1, "Poland": 1, "Senegal": 1,
    "Iran": 1, "Egypt": 1, "Saudi Arabia": 1, "Morocco": 1,
    "Serbia": 1, "Iceland": 1, "Australia": 1, "Peru": 1,
    "Nigeria": 1, "Costa Rica": 1, "Panama": 1, "Tunisia": 1,
}
WC18_MAX = 7


def _fetch_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _fetch_statsbomb_player_stats(competition_id: int, season_id: int) -> dict[str, dict]:
    """
    Compute per-team player xG totals from StatsBomb open data.
    Returns {team: {player: {xg, goals, shots, matches}}}
    Cached to disk so we only fetch once.
    """
    cache_file = CACHE_DIR / f"sb_{competition_id}_{season_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    print(f"  Fetching StatsBomb competition {competition_id} season {season_id}...")
    matches_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{competition_id}/{season_id}.json"
    matches = _fetch_json(matches_url)

    team_players: dict[str, dict] = defaultdict(lambda: defaultdict(
        lambda: {"xg": 0.0, "goals": 0, "shots": 0, "matches": 0}
    ))

    for i, m in enumerate(matches):
        mid = m["match_id"]
        try:
            events = _fetch_json(
                f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{mid}.json"
            )
        except Exception:
            continue

        seen = set()
        for e in events:
            if e.get("type", {}).get("name") != "Shot":
                continue
            player = e.get("player", {}).get("name", "?")
            team = _sb_normalize(e.get("team", {}).get("name", "?"))
            shot = e.get("shot", {})
            xg = shot.get("statsbomb_xg", 0) or 0
            is_goal = shot.get("outcome", {}).get("name") == "Goal"
            team_players[team][player]["xg"] += xg
            team_players[team][player]["goals"] += int(is_goal)
            team_players[team][player]["shots"] += 1
            key = (team, player)
            if key not in seen:
                team_players[team][player]["matches"] += 1
                seen.add(key)

        if (i + 1) % 16 == 0:
            print(f"    {i+1}/{len(matches)} matches processed")

    result = {team: dict(players) for team, players in team_players.items()}
    cache_file.write_text(json.dumps(result, indent=2))
    return result


def _wc_pedigree_score(team: str) -> float:
    """
    Blended pedigree from WC 2022 + WC 2018, normalised 0-1.
    Recent WC weighted 2x vs older.
    """
    p22 = WC22_PEDIGREE.get(team, 0) / WC22_MAX
    p18 = WC18_PEDIGREE.get(team, 0) / WC18_MAX
    return (2 * p22 + p18) / 3


def _team_xg_quality(team: str, player_stats: dict[str, dict]) -> float:
    """
    Total xG of top-5 players as a fraction of the WC 2022 median team.
    Returns a score where 1.0 = median WC squad, >1 = better attacking threat.

    WC 2022 median team total xG for top-5 players ≈ 6.5 across the tournament.
    (Derived empirically from StatsBomb data.)
    """
    if team not in player_stats:
        return 1.0
    players = player_stats[team]
    total_xg_per_player = [s["xg"] for s in players.values() if s["shots"] > 0]
    if not total_xg_per_player:
        return 1.0
    total_xg_per_player.sort(reverse=True)
    top5_total = sum(total_xg_per_player[:5])
    # WC 2022 median ≈ 6.5; Argentina had ~16, Qatar had ~1.8
    return min(max(top5_total / 6.5, 0.4), 2.5)


def _gemini_form_score(team: str, key_players: list[str], gemini_key: str) -> float:
    """
    Ask Gemini (free tier) to rate a team's key players' current club form 0-10.
    Requires a free Gemini API key from aistudio.google.com.
    Returns a normalized multiplier (0.85 – 1.15).
    """
    names = ", ".join(key_players[:5])
    prompt = (
        f"Rate the current club form (2025/26 season) of these players from 0-10 "
        f"based on goals, assists, and xG performance: {names}. "
        f"Reply ONLY with a JSON object like: "
        f'{{"{key_players[0]}": 7, "{key_players[1] if len(key_players)>1 else "x"}": 8, ...}} '
        f"No explanation, just the JSON."
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }).encode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
    req = urllib.request.Request(url, data=payload, method="POST",
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        resp = json.loads(r.read())

    text = resp["candidates"][0]["content"]["parts"][0]["text"]
    # Extract JSON from response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1:
        return 1.0
    ratings = json.loads(text[start:end])
    avg_rating = sum(ratings.values()) / len(ratings) if ratings else 5.0
    # Map 0-10 rating to 0.85-1.15 multiplier
    return 0.85 + (avg_rating / 10) * 0.30


def build_squad_adjustments(gemini_key: str | None = None) -> dict[str, float]:
    """
    Main entry point. Returns {team: multiplier} for all WC 2026 teams.

    multiplier > 1.0 → team stronger than their betting-market prior suggests
    multiplier < 1.0 → weaker

    Without a Gemini key, uses only StatsBomb + pedigree (still meaningful).
    With a free Gemini key, adds current club form layer on top.
    """
    print("[squad] Loading StatsBomb WC 2022 player data...")
    stats_2022 = _fetch_statsbomb_player_stats(43, 106)

    # Key players per team (top 3 by xG in WC 2022, for Gemini queries)
    team_key_players: dict[str, list[str]] = {}
    for team, players in stats_2022.items():
        ranked = sorted(players.items(), key=lambda x: -x[1]["xg"])
        team_key_players[team] = [p for p, _ in ranked[:3]]

    adjustments: dict[str, float] = {}

    # All WC 2026 teams
    WC_2026_TEAMS = [
        "Mexico","South Korea","Czechia","South Africa","Canada","Switzerland","Bosnia","Qatar",
        "Brazil","Morocco","Scotland","Haiti","USA","Australia","Turkey","Paraguay",
        "Germany","Ivory Coast","Ecuador","Curacao","Netherlands","Japan","Sweden","Tunisia",
        "Belgium","Egypt","Iran","New Zealand","Spain","Uruguay","Saudi Arabia","Cabo Verde",
        "France","Norway","Senegal","Iraq","Argentina","Austria","Jordan","Algeria",
        "Portugal","Colombia","DR Congo","Uzbekistan","England","Croatia","Ghana","Panama",
    ]

    print("[squad] Computing squad adjustments...")
    for team in WC_2026_TEAMS:
        pedigree = _wc_pedigree_score(team)           # 0–1
        xg_qual  = _team_xg_quality(team, stats_2022) # 0.5–2.0

        # Base multiplier: pedigree (35%) + xG quality (65%), centered at 1.0
        # Range kept tight (0.93–1.07) so squad signal nudges rather than overrides market priors
        base = 1.0 + 0.12 * (pedigree - 0.3) + 0.06 * (xg_qual - 1.0)
        base = min(max(base, 0.93), 1.07)

        # Optional: Gemini current form layer
        form_mult = 1.0
        if gemini_key and team in team_key_players and team_key_players[team]:
            try:
                form_mult = _gemini_form_score(team, team_key_players[team], gemini_key)
                form_mult = min(max(form_mult, 0.96), 1.04)  # cap Gemini influence to ±4%
            except Exception as e:
                print(f"  [squad] Gemini failed for {team}: {e}")

        adjustments[team] = round(base * form_mult, 4)

    return adjustments


def print_adjustments(adj: dict[str, float]) -> None:
    ranked = sorted(adj.items(), key=lambda x: -x[1])
    print(f"\n{'Team':<22} {'Multiplier':>12}  {'Signal'}")
    print("─" * 55)
    for team, mult in ranked:
        pedigree = _wc_pedigree_score(team)
        bar = "▲" * int((mult - 0.88) * 50) if mult > 1.0 else "▼" * int((1.0 - mult) * 50)
        ped_str = f"WC22 pedigree: {WC22_PEDIGREE.get(team, 0)} pts" if WC22_PEDIGREE.get(team) else "no WC22"
        print(f"  {team:<20} ×{mult:>7.4f}  {ped_str}")


if __name__ == "__main__":
    import sys
    gemini_key = os.environ.get("GEMINI_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)

    if not gemini_key:
        print("No Gemini key — using StatsBomb + pedigree only.")
        print("For current club form, get a FREE key at: aistudio.google.com\n")
    else:
        print(f"Gemini key found — will query current club form for key players.\n")

    adj = build_squad_adjustments(gemini_key=gemini_key)
    print_adjustments(adj)
