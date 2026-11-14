"""Offline tests: synthetic data only, no downloads."""
import numpy as np
import pandas as pd
import pytest

from datetime import date

from src.data import current_season, parse_dates
from src.evaluate import accuracy, brier, implied_probs, log_loss, one_hot, walk_forward
from src.poisson import PoissonModel


def simulate_league(n_teams=10, rounds=4, seed=0, start="2023-08-01"):
    """Double round-robin repeated `rounds` times, goals drawn from a known Poisson model."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    att -= att.mean()
    dfn = rng.normal(0, 0.3, n_teams)
    mu, home_adv = 0.1, 0.25
    rows, day = [], pd.Timestamp(start)
    for _ in range(rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                hg = rng.poisson(np.exp(mu + home_adv + att[i] + dfn[j]))
                ag = rng.poisson(np.exp(mu + att[j] + dfn[i]))
                rows.append({"date": day, "home": teams[i], "away": teams[j], "home_goals": hg, "away_goals": ag,
                             "result": "H" if hg > ag else "A" if ag > hg else "D"})
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows), att, home_adv


def test_parse_dates_handles_all_formats():
    s = pd.Series(["16/08/2024", "16/08/24", "2024-08-16"])
    assert (parse_dates(s) == pd.Timestamp("2024-08-16")).all()


def test_current_season_code():
    assert current_season(date(2026, 11, 12)) == "2627"
    assert current_season(date(2027, 5, 20)) == "2627"
    assert current_season(date(2027, 7, 1)) == "2728"
    assert current_season(date(2099, 8, 1)) == "9900"


def test_metrics_on_perfect_and_uniform_predictions():
    y = one_hot(pd.Series(["H", "D", "A"]))
    assert log_loss(y, y) == pytest.approx(0, abs=1e-9)
    assert brier(y, y) == 0
    assert accuracy(y, y) == 1
    uniform = np.full((3, 3), 1 / 3)
    assert log_loss(uniform, y) == pytest.approx(np.log(3))
    assert brier(uniform, y) == pytest.approx(2 / 3)


def test_implied_probs_remove_margin():
    odds = pd.DataFrame({"odds_h": [2.0], "odds_d": [3.4], "odds_a": [3.8]})
    p = implied_probs(odds)
    assert p.sum() == pytest.approx(1)
    assert p[0, 0] > p[0, 2]


def test_model_recovers_true_parameters():
    df, true_att, true_home = simulate_league(rounds=6)
    model = PoissonModel(xi=0.0).fit(df)
    assert model.converged_
    assert model.home_adv_ == pytest.approx(true_home, abs=0.08)
    assert np.corrcoef(model.attack_, true_att)[0, 1] > 0.9


def test_probabilities_are_valid():
    df, _, _ = simulate_league()
    model = PoissonModel().fit(df)
    for home, away in [("T0", "T1"), ("T5", "T2"), ("T0", "NewTeam")]:
        p = model.predict_proba(home, away)
        assert sum(p) == pytest.approx(1)
        assert all(0 < x < 1 for x in p)


def test_walk_forward_has_no_lookahead(monkeypatch):
    """Every model used for a prediction must be fitted only on matches played before it."""
    df, _, _ = simulate_league(rounds=2)
    split = df["date"].iloc[len(df) // 2]
    history, test = df[df["date"] < split], df[df["date"] >= split]

    seen_max_dates = []
    original_fit = PoissonModel.fit

    def spy_fit(self, matches, as_of=None):
        seen_max_dates.append((matches["date"].max(), pd.Timestamp(as_of)))
        return original_fit(self, matches, as_of)

    monkeypatch.setattr(PoissonModel, "fit", spy_fit)
    preds = walk_forward(history, test)
    assert len(preds) == len(test)
    assert all(last_seen < as_of for last_seen, as_of in seen_max_dates)
