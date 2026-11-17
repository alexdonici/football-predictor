"""Can machine learning combine Poisson, Elo and form better than Poisson alone?

Training table: one row per match with pre-match information only (see src/ml.py).
For each test season, the models are trained on all earlier seasons and then predict the
whole test season. Settings were chosen on 2023/24 only; 2024/25 and 2025/26 were run once.

The 2020/21 season is left out of training: it was played without fans, home teams won less
often than away teams (38% vs 40%), so it teaches the wrong home advantage.

    python -m experiments.ml_vs_poisson
"""
import warnings

import pandas as pd

from src.data import load_seasons
from src.evaluate import accuracy, brier, log_loss, one_hot
from src.ml import ALL_FEATURES, POISSON_FEATURES, build_dataset, fit_predict, logistic_model, xgboost_model

SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
FIRST_ROW_SEASON = "1920"     # 2018/19 is only history for Poisson, Elo and form
EXCLUDED_FROM_TRAINING = ["2021"]
TEST_SEASONS = ["2324", "2425", "2526"]
XGB_SETTINGS = dict(max_depth=1, learning_rate=0.02, n_estimators=300, min_child_weight=30)


def main(league: str = "E0"):
    warnings.filterwarnings("ignore", category=UserWarning)
    data = build_dataset(load_seasons(league, SEASONS), FIRST_ROW_SEASON)
    pool = data[~data["season"].isin(EXCLUDED_FROM_TRAINING)]

    rows = []
    for test in TEST_SEASONS:
        train, te = pool[pool["season"] < test], data[data["season"] == test]
        y = one_hot(te["result"])
        preds = {
            "Poisson model": te[["pois_h", "pois_d", "pois_a"]].to_numpy(),
            "LogReg: Poisson only": fit_predict(logistic_model(), train, te, POISSON_FEATURES),
            "LogReg: Poisson + Elo + form": fit_predict(logistic_model(), train, te, ALL_FEATURES),
            "XGBoost: Poisson + Elo + form": fit_predict(xgboost_model(**XGB_SETTINGS), train, te, ALL_FEATURES),
        }
        for name, p in preds.items():
            rows.append({"model": name, "season": test, "log loss": log_loss(p, y),
                         "Brier": brier(p, y), "accuracy": accuracy(p, y)})

    table = pd.DataFrame(rows)
    order = list(dict.fromkeys(table["model"]))
    for metric in ["log loss", "Brier", "accuracy"]:
        print(f"\n{metric} by test season ({'higher' if metric == 'accuracy' else 'lower'} is better)")
        print(table.pivot(index="model", columns="season", values=metric).loc[order].round(4).to_string())


if __name__ == "__main__":
    main()
