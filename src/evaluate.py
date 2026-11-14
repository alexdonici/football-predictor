"""Walk-forward backtest: the model only sees matches played before each prediction."""
import numpy as np
import pandas as pd

from .poisson import PoissonModel

OUTCOMES = ["H", "D", "A"]


def implied_probs(df: pd.DataFrame) -> np.ndarray:
    """Bookmaker probabilities from decimal odds, with the margin (overround) removed."""
    raw = 1.0 / df[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    return raw / raw.sum(axis=1, keepdims=True)


def one_hot(results: pd.Series) -> np.ndarray:
    return np.stack([(results == o).to_numpy(float) for o in OUTCOMES], axis=1)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip((p * y).sum(axis=1), 1e-12, 1))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(((p - y) ** 2).sum(axis=1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(p.argmax(axis=1) == y.argmax(axis=1)))


def walk_forward(history: pd.DataFrame, test: pd.DataFrame, refit_days: int = 7, **model_kwargs) -> pd.DataFrame:
    """Predict every match in `test`, refitting the model every `refit_days` days."""
    preds = []
    start = test["date"].min()
    while start <= test["date"].max():
        end = start + pd.Timedelta(days=refit_days)
        block = test[(test["date"] >= start) & (test["date"] < end)]
        if len(block):
            seen = pd.concat([history, test[test["date"] < start]])
            model = PoissonModel(**model_kwargs).fit(seen, as_of=start)
            for _, m in block.iterrows():
                ph, pd_, pa = model.predict_proba(m["home"], m["away"])
                preds.append({**m.to_dict(), "p_h": ph, "p_d": pd_, "p_a": pa})
        start = end
    return pd.DataFrame(preds)


def report(preds: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Compare the model with the bookmakers and a naive baseline on the same matches."""
    base_rates = one_hot(history["result"]).mean(axis=0)  # how often H / D / A happened before
    y = one_hot(preds["result"])
    rows = {"Poisson model": preds[["p_h", "p_d", "p_a"]].to_numpy()}
    if {"odds_h", "odds_d", "odds_a"} <= set(preds.columns):
        ok = preds[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1).to_numpy()
        y = y[ok]
        rows = {k: v[ok] for k, v in rows.items()}
        rows["Bookmakers"] = implied_probs(preds[ok])
    rows["Baseline (past H/D/A rates)"] = np.tile(base_rates, (len(y), 1))
    return pd.DataFrame(
        {name: {"log loss": log_loss(p, y), "Brier": brier(p, y), "accuracy": accuracy(p, y)}
         for name, p in rows.items()}
    ).T.round(4)
