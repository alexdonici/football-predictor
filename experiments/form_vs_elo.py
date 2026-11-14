"""Does recent form add information beyond Elo?

Fits a multinomial logistic regression on past seasons with different feature sets and
scores it on the next season. All features are pre-match, so there is no look-ahead.

    python -m experiments.form_vs_elo
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_seasons
from src.evaluate import accuracy, brier, log_loss, one_hot
from src.features import build_features

FEATURE_SETS = {
    "Elo only": ["elo_diff"],
    "Elo + form": ["elo_diff", "form_pts_diff", "form_gd_diff"],
    "Elo + form + venue form": ["elo_diff", "form_pts_diff", "form_gd_diff", "venue_pts_diff"],
    "Form only (no Elo)": ["form_pts_diff", "form_gd_diff", "venue_pts_diff"],
}
SPLITS = [  # (test season, training seasons)
    ("2324", ["1920", "2021", "2122", "2223"]),
    ("2425", ["2021", "2122", "2223", "2324"]),
    ("2526", ["2122", "2223", "2324", "2425"]),
]
BURN_IN_DAYS = 300  # Elo needs time before its ratings mean anything


def main(league: str = "E0"):
    rows = []
    for test, train in SPLITS:
        f = build_features(load_seasons(league, train + [test]))
        start = f["date"].min() + pd.Timedelta(days=BURN_IN_DAYS)
        tr, te = f[(f["season"] != test) & (f["date"] >= start)], f[f["season"] == test]
        y = one_hot(te["result"])
        for name, cols in FEATURE_SETS.items():
            model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(tr[cols], tr["result"])
            p = pd.DataFrame(model.predict_proba(te[cols]), columns=model.classes_)[["H", "D", "A"]].to_numpy()
            rows.append({"test season": test, "features": name,
                         "log loss": log_loss(p, y), "Brier": brier(p, y), "accuracy": accuracy(p, y)})
    table = pd.DataFrame(rows).pivot(index="features", columns="test season", values="log loss")
    print("Log loss by test season (lower is better)\n")
    print(table.loc[list(FEATURE_SETS)].round(4).to_string())


if __name__ == "__main__":
    main()
