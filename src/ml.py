"""Machine-learning models that combine Poisson, Elo and form.

The training table has one row per match with what was known *before* kick-off:
- Poisson probabilities from the walk-forward backtest (the model only saw earlier matches)
- Elo rating difference and recent-form features (see features.py)
and the target: what actually happened (H / D / A).

A model is trained on whole past seasons and then predicts a later season. The features of
that later season keep updating week by week, but the ML model itself is not retrained.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import walk_forward
from .features import build_features
from .poisson import PoissonModel

OUTCOMES = ["H", "D", "A"]

POISSON_FEATURES = ["pois_logit_h", "pois_logit_a"]
ELO_FORM_FEATURES = ["elo_diff", "form_pts_diff", "form_gd_diff", "venue_pts_diff"]
ALL_FEATURES = POISSON_FEATURES + ELO_FORM_FEATURES


def build_dataset(matches: pd.DataFrame, first_season: str) -> pd.DataFrame:
    """Training table for every match from `first_season` on.

    Earlier seasons are only used as history (for Poisson, Elo and form), never as rows.
    """
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    feats = build_features(matches)

    history, target = matches[matches["season"] < first_season], matches[matches["season"] >= first_season]
    pois = walk_forward(history, target, PoissonModel)[["date", "home", "away", "p_h", "p_d", "p_a"]]
    pois = pois.rename(columns={"p_h": "pois_h", "p_d": "pois_d", "p_a": "pois_a"})

    data = feats[feats["season"] >= first_season].merge(pois, on=["date", "home", "away"], how="left",
                                                         validate="one_to_one")
    # Log-odds against the draw: the scale on which a logistic regression combines probabilities
    data["pois_logit_h"] = np.log(data["pois_h"] / data["pois_d"])
    data["pois_logit_a"] = np.log(data["pois_a"] / data["pois_d"])
    return data


def logistic_model(C: float = 1.0):
    """Multinomial logistic regression; missing form values are filled with the training mean."""
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def xgboost_model(**params):
    from xgboost import XGBClassifier  # optional dependency, imported only when used

    defaults = dict(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0)
    return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **{**defaults, **params})


def fit_predict(model, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Fit on `train`, return probabilities for `test` as columns H, D, A."""
    y = train["result"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    model.fit(train[features], y)
    return model.predict_proba(test[features])
