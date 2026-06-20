#!/usr/bin/env python3
"""
WC 2026 Prediction Engine
Pulls live match data → runs improved model → prints ranked probabilities + deltas.

Usage:
  python3 engine.py            # full run with 50k sims
  python3 engine.py --quick    # 10k sims, faster
  python3 engine.py --delta    # show vs pre-tournament baseline too
"""

import sys
import time
import os

from dotenv import load_dotenv
load_dotenv()

from fetch import fetch_results
from wc2026_sim import TEAMS, run_with_results
from squad import build_squad_adjustments


def format_bar(p: float, width: int = 30) -> str:
    filled = int(p * width * 4)
    full, part = divmod(filled, 4)
    blocks = "█" * full + (" ▏▎▍▌▋▊▉"[part] if part else "")
    return blocks.ljust(width)


def print_table(probs: dict, label: str, deltas: dict | None = None) -> None:
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    total = sum(probs.values())

    print(f"\n{'━'*62}")
    print(f"  {label}")
    print(f"  (probs sum to {total*100:.1f}% across {len(probs)} teams)")
    print(f"{'━'*62}")
    print(f"  {'Team':<20} {'Win%':>6}  {'Bar':<30}  {'Δ':>6}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*30}  {'─'*6}")

    for team, p in ranked:
        if p < 0.003:
            continue
        bar = format_bar(p)
        delta_str = ""
        if deltas and team in deltas:
            d = deltas[team] * 100
            delta_str = f"{d:>+6.1f}pp" if abs(d) >= 0.05 else "      —"
        print(f"  {team:<20} {p*100:>5.1f}%  {bar}  {delta_str}")

    print(f"{'━'*62}")


def main() -> None:
    quick = "--quick" in sys.argv
    show_delta = "--delta" in sys.argv
    n = 10_000 if quick else 50_000

    print("─" * 62)
    print("  WC 2026 Prediction Engine")
    print("─" * 62)

    # 1. Fetch live data
    print("\n[1/3] Fetching live match data from ESPN...", end=" ", flush=True)
    t0 = time.time()
    try:
        matches = fetch_results()
        print(f"OK — {len(matches)} completed matches ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")
        print("       Falling back to hardcoded results.")
        matches = None

    # 2. Squad intelligence (StatsBomb WC 2022 + optional Groq/Tavily current form)
    groq_key   = os.environ.get("GROQ_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    has_form = groq_key and tavily_key
    print(f"\n[2/4] Building squad adjustments (StatsBomb WC 2022", end="", flush=True)
    print(f" + Groq/Tavily form)..." if has_form else f", no form keys)...", end=" ", flush=True)
    t0 = time.time()
    try:
        squad_adj = build_squad_adjustments(groq_key=groq_key, tavily_key=tavily_key)
        print(f"OK ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e} — skipping squad layer")
        squad_adj = None

    # 3. Optional baseline (pre-tournament priors, no results)
    baseline_probs = None
    if show_delta or matches is None:
        print(f"\n[3/4] Computing baseline (pre-tournament priors, {n:,} sims)...", end=" ", flush=True)
        t0 = time.time()
        baseline_probs = run_with_results([], n=n, seed=42, squad_adjustments=squad_adj)
        print(f"done ({time.time()-t0:.1f}s)")
    else:
        print(f"\n[3/4] Baseline skipped (use --delta to include)")

    # 4. Live model
    effective_matches = matches if matches is not None else []
    print(f"\n[4/4] Running model with {len(effective_matches)} results ({n:,} sims)...", end=" ", flush=True)
    t0 = time.time()
    live_probs = run_with_results(effective_matches, n=n, seed=2026, squad_adjustments=squad_adj)
    print(f"done ({time.time()-t0:.1f}s)")

    # 4. Display
    deltas = None
    if baseline_probs:
        deltas = {t: live_probs.get(t, 0) - baseline_probs.get(t, 0) for t in TEAMS}

    squad_note = " + squad/form" if squad_adj else ""
    label = f"Championship odds — {len(effective_matches)} matches played (Bayesian + co-host boost{squad_note})"
    print_table(live_probs, label, deltas)

    # 5. Notable signals from match quality data
    if matches:
        penalties_own_goals = []
        red_card_games = []
        for m in matches:
            raw_h, raw_a = m["home_goals"], m["away_goals"]
            eff_h, eff_a = m["effective_home"], m["effective_away"]
            if abs(raw_h - eff_h) > 0.5 or abs(raw_a - eff_a) > 0.5:
                penalties_own_goals.append(m)
            if m["home_red_cards"] + m["away_red_cards"] > 0:
                red_card_games.append(m)

        if penalties_own_goals or red_card_games:
            print("\n  Quality adjustments applied:")
            for m in penalties_own_goals:
                print(f"    {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']} "
                      f"→ effective {m['effective_home']:.1f}-{m['effective_away']:.1f} "
                      f"(penalties/own goals discounted)")
            for m in red_card_games:
                total_rc = m['home_red_cards'] + m['away_red_cards']
                scale = 0.35 if total_rc >= 2 else 0.7
                print(f"    {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']} "
                      f"→ Bayesian update scaled to {scale:.0%} (red cards distorted scoreline)")


if __name__ == "__main__":
    main()
