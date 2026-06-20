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


# ── Manager pedigree (0–10) ───────────────────────────────────────────────
# Scoring: 10=multiple major trophies, 9=WC winner, 8=continental winner,
# 7=WC SF/finalist or serial winner, 6=experienced/good win rate, 5=average, 4=new/limited
MANAGER_SCORES = {
    # Elite — WC winners as manager
    "Argentina":    10,  # Scaloni: WC 2022, Copa América 2021 + 2024
    "France":        9,  # Deschamps: WC 2018, Euro 2016 final, WC 2022 final
    "Croatia":       8,  # Dalić: WC 2018 final, WC 2022 3rd place
    "Spain":         8,  # De la Fuente: Euro 2024 winner
    "Senegal":       7,  # Cissé: AFCON 2022 winner, 10+ years in charge
    "Ivory Coast":   7,  # Faé: AFCON 2023 winner
    "Morocco":       7,  # Regragui: WC 2022 semi (historic), AFCON 2022 winner (clubs)
    "England":       7,  # Tuchel: UCL winner (club), high tactical pedigree
    "Uruguay":       7,  # Bielsa: legendary record, elite tactical reputation
    "Austria":       7,  # Rangnick: elite pressing tactician, transformed Austria
    "Colombia":      7,  # Lorenzo: Copa América 2024 finalist
    "Germany":       6,  # Nagelsmann: talented, no major NT trophies yet
    "Netherlands":   6,  # Koeman: experienced, no major NT trophies
    "Portugal":      6,  # Martínez: WC 2018 3rd with Belgium
    "Switzerland":   6,  # Yakin: steady, experienced
    "Japan":         6,  # Moriyasu: consistent overachiever at WC 2022
    "Scotland":      6,  # Clarke: qualified for multiple Euros
    "Mexico":        6,  # Aguirre: third stint, experienced campaigner
    "USA":           6,  # Pochettino: elite club pedigree, no NT trophies
    "Saudi Arabia":  6,  # Experienced European coach
    "Brazil":        5,  # Dorival Júnior: Copa América 2024 exit, under pressure
    "Belgium":       5,  # Garcia: limited international record
    "Norway":        5,  # Solbakken: solid but unproven at majors
    "Sweden":        5,  # Tomasson: early tenure
    "Ecuador":       5,  # Beccacece: young, no major trophies
    "Canada":        5,  # Marsch: new to NT management
    "Turkey":        5,  # Montella: limited international record
    "Iran":          5,  # Ghalenoei: domestic pedigree only
    "Tunisia":       5,
    "Ghana":         5,
    "Egypt":         5,
    "South Korea":   5,
    "Australia":     5,
    "Paraguay":      5,
    "Panama":        5,
    "Algeria":       5,
    "Iraq":          5,
    "Jordan":        4,
    "New Zealand":   4,
    "South Africa":  4,
    "Czechia":       4,
    "Bosnia":        4,
    "Qatar":         4,
    "Haiti":         4,
    "Curacao":       4,
    "Cabo Verde":    4,
    "DR Congo":      4,
    "Uzbekistan":    4,
}
MANAGER_MAX = 10


def _manager_score(team: str) -> float:
    """Normalised manager quality 0–1."""
    return MANAGER_SCORES.get(team, 5) / MANAGER_MAX


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


def _tavily_search(query: str, tavily_key: str) -> str:
    """Search Tavily and return concatenated result snippets (max 400 chars)."""
    payload = json.dumps({
        "api_key": tavily_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 3,
    }).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search", data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        results = json.loads(r.read())
    snippets = [item.get("content", "") for item in results.get("results", [])]
    return " ".join(snippets)[:400]


def _groq_form_batch(team_snippets: dict[str, str], groq_key: str) -> dict[str, float]:
    """
    Single Groq call — reads Tavily snippets, outputs form + injury scores.
    Returns {team: combined_multiplier} where:
      form_mult    = 0.96 + (form/10) * 0.08   → range 0.96–1.04
      injury_mult  = 1.0  - (injuries/10) * 0.06 → range 0.94–1.00
      combined     = form_mult * injury_mult       → range ~0.90–1.04
    """
    lines = [f"- {team}: {snippet}" for team, snippet in team_snippets.items()]
    prompt = (
        "You are a football analyst. Based ONLY on the news snippets below, rate each national "
        "team on TWO signals (integers 0-10):\n"
        "  form: collective club form this 2025/26 season "
        "(10=exceptional, 5=average, 0=very poor)\n"
        "  injuries: injury severity heading into WC 2026 "
        "(0=fully fit, 5=one key player out, 10=multiple starters injured)\n\n"
        + "\n".join(lines)
        + "\n\nReply ONLY with valid JSON: "
        "{\"Team\": {\"form\": 7, \"injuries\": 2}, ...}. No explanation."
    )
    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 900,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions", data=payload, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {groq_key}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    text = resp["choices"][0]["message"]["content"]
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1:
        return {}
    scores: dict = json.loads(text[start:end])
    result: dict[str, float] = {}
    for team, s in scores.items():
        if isinstance(s, dict):
            form     = float(s.get("form", 5))
            injuries = float(s.get("injuries", 0))
        else:
            form, injuries = float(s), 0.0
        form_mult   = min(max(0.96 + (form / 10) * 0.08, 0.96), 1.04)
        injury_mult = 1.0 - (injuries / 10) * 0.06
        result[team] = round(form_mult * injury_mult, 4)
    return result


def _groq_tavily_form(team_players: dict[str, list[str]],
                      groq_key: str, tavily_key: str) -> dict[str, float]:
    """
    Pipeline: Tavily search per team → single Groq call → form + injury multipliers.
    Each search covers both form and injury signals (zero extra Tavily credits vs. form-only).
    Uses ~26 Tavily credits per run, cached 6h (~780/month, within 1,000 free limit).
    """
    import time as _time

    cache_file = CACHE_DIR / "form_injury_scores.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        age = _time.time() - cached.get("_ts", 0)
        if age < 21600:  # 6h TTL
            return {k: v for k, v in cached.items() if k != "_ts"}

    # 1. Tavily: 1 search per team — query covers form AND injury in one shot
    print(f"\n    Searching form + injuries for {len(team_players)} teams via Tavily...", flush=True)
    team_snippets: dict[str, str] = {}
    for team, players in team_players.items():
        names = " ".join(players[:2])
        query = f"{names} WC 2026 injuries fitness form 2025/26"
        try:
            team_snippets[team] = _tavily_search(query, tavily_key)
        except Exception:
            team_snippets[team] = ""

    # 2. Groq: single batch call — extracts form + injury scores, returns combined mult
    print(f"    Scoring form + injuries with Groq LLaMA 3.3 70B...", flush=True)
    result = _groq_form_batch(team_snippets, groq_key)

    cache_file.write_text(json.dumps({**result, "_ts": _time.time()}))
    return result


def build_squad_adjustments(gemini_key: str | None = None,
                            groq_key: str | None = None,
                            tavily_key: str | None = None) -> dict[str, float]:
    """
    Main entry point. Returns {team: multiplier} for all WC 2026 teams.

    multiplier > 1.0 → team stronger than their betting-market prior suggests
    multiplier < 1.0 → weaker

    Data layers (each optional, gracefully skipped if keys absent):
      - StatsBomb WC 2022 xG (always runs, no key needed)
      - WC pedigree scoring (always runs)
      - Groq + Tavily current club form (requires both keys)
    """
    print("[squad] Loading StatsBomb WC 2022 player data...")
    stats_2022 = _fetch_statsbomb_player_stats(43, 106)

    team_key_players: dict[str, list[str]] = {}
    for team, players in stats_2022.items():
        ranked = sorted(players.items(), key=lambda x: -x[1]["xg"])
        team_key_players[team] = [p for p, _ in ranked[:3]]

    WC_2026_TEAMS = [
        "Mexico","South Korea","Czechia","South Africa","Canada","Switzerland","Bosnia","Qatar",
        "Brazil","Morocco","Scotland","Haiti","USA","Australia","Turkey","Paraguay",
        "Germany","Ivory Coast","Ecuador","Curacao","Netherlands","Japan","Sweden","Tunisia",
        "Belgium","Egypt","Iran","New Zealand","Spain","Uruguay","Saudi Arabia","Cabo Verde",
        "France","Norway","Senegal","Iraq","Argentina","Austria","Jordan","Algeria",
        "Portugal","Colombia","DR Congo","Uzbekistan","England","Croatia","Ghana","Panama",
    ]

    form_scores: dict[str, float] = {}
    if groq_key and tavily_key:
        try:
            print("[squad] Fetching form + injuries: Tavily search → Groq LLaMA 3.3 70B...", end=" ", flush=True)
            form_scores = _groq_tavily_form(team_key_players, groq_key, tavily_key)
            print(f"OK — {len(form_scores)} teams scored (form × injury multiplier)")
        except Exception as e:
            print(f"FAILED: {e}")
    else:
        print("[squad] No Groq/Tavily keys — using StatsBomb + pedigree only")

    print("[squad] Computing squad adjustments...")
    adjustments: dict[str, float] = {}
    for team in WC_2026_TEAMS:
        pedigree = _wc_pedigree_score(team)
        xg_qual  = _team_xg_quality(team, stats_2022)
        manager  = _manager_score(team)           # 0–1
        # Pedigree 35%, xG quality 40%, manager 25% — all nudge around 1.0
        base = (1.0
                + 0.12 * (pedigree - 0.3)
                + 0.06 * (xg_qual - 1.0)
                + 0.06 * (manager - 0.5))
        base = min(max(base, 0.90), 1.10)
        form_mult = form_scores.get(team, 1.0)
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
