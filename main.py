"""Football match outcome prediction - backtest of a Poisson goals model.

Examples:
    python main.py                                   # Premier League, train 2021-25, test 2025/26
    python main.py --league SP1                      # La Liga
    python main.py --predict Arsenal "Man United"    # one match, model fitted on all data incl. current season
"""
import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd
import requests

from src.data import current_season, load_seasons
from src.elo import EloModel
from src.evaluate import average, report, walk_forward
from src.poisson import PoissonModel

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def check_team(name: str, known: set[str]) -> None:
    """Exit with a helpful message if a team name is not in the data (e.g. 'Man Utd' vs 'Man United')."""
    if name in known:
        return
    hint = difflib.get_close_matches(name, known, n=3, cutoff=0.4)
    msg = f"Unknown team '{name}'."
    msg += f" Did you mean: {', '.join(hint)}?" if hint else f" Known teams: {', '.join(sorted(known))}"
    sys.exit(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0", help="E0=Premier League, SP1=La Liga, I1=Serie A, D1=Bundesliga, F1=Ligue 1")
    ap.add_argument("--train", nargs="+", default=["2122", "2223", "2324", "2425"], help="seasons used as history")
    ap.add_argument("--test", default="2526", help="season to backtest on")
    ap.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"), help="predict one match instead of backtesting")
    args = ap.parse_args()

    history = load_seasons(args.league, args.train)
    test = load_seasons(args.league, [args.test])

    if args.predict:
        data = pd.concat([history, test])
        season = current_season()
        if season not in args.train + [args.test]:
            try:
                data = pd.concat([data, load_seasons(args.league, [season])])
            except requests.RequestException:
                print(f"Warning: could not download the current season ({season}); using older seasons only.")
        print(f"Model fitted on {len(data)} matches up to {data['date'].max():%d %b %Y}")
        home, away = args.predict
        known = set(data["home"]) | set(data["away"])
        check_team(home, known)
        check_team(away, known)
        poisson = PoissonModel().fit(data)
        elo = EloModel().fit(data)
        lh, la = poisson.expected_goals(home, away)
        print(f"\n{home} vs {away}")
        print(f"Expected goals (Poisson): {lh:.2f} - {la:.2f}")
        print(f"Elo ratings: {elo.rating(home):.0f} vs {elo.rating(away):.0f}\n")
        for name, model in [("Poisson", poisson), ("Elo", elo)]:
            ph, pd_, pa = model.predict_proba(home, away)
            print(f"{name:<8} Home {ph:6.1%} | Draw {pd_:6.1%} | Away {pa:6.1%}")
        return

    print(f"Backtesting {args.league} {args.test} ({len(test)} matches, refit weekly)...")
    preds = {"Poisson model": walk_forward(history, test, PoissonModel),
             "Elo model": walk_forward(history, test, EloModel)}
    preds["Average (Poisson + Elo)"] = average(*preds.values())
    metrics = report(preds, history)
    print("\nLower log loss / Brier = better probabilities\n")
    print(metrics.to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{args.league}_{args.test}"
    out = preds["Poisson model"].drop(columns=["p_h", "p_d", "p_a"])
    for name, short in [("Poisson model", "poisson"), ("Elo model", "elo")]:
        for c in ["p_h", "p_d", "p_a"]:
            out[f"{short}_{c}"] = preds[name][c].to_numpy()
    out.to_csv(RESULTS_DIR / f"{stem}_predictions.csv", index=False)
    metrics.to_csv(RESULTS_DIR / f"{stem}_metrics.csv")
    print(f"\nSaved predictions and metrics to {RESULTS_DIR.name}/{stem}_*.csv")

    data = pd.concat([history, test])
    ratings = PoissonModel().fit(data).ratings_table().merge(EloModel().fit(data).ratings_table(), on="team")
    print("\nTeam ratings after the last match in the data (top 10 by Poisson strength):")
    print(ratings.head(10).round({"attack": 3, "defence": 3, "strength": 3, "elo": 0}).to_string(index=False))


if __name__ == "__main__":
    main()
