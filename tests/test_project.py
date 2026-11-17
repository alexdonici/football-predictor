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


# ---------- Elo ----------
from src.elo import EloModel, margin_multiplier  # noqa: E402


def with_season(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(season=df["date"].dt.year.astype(str))


def test_margin_multiplier():
    assert margin_multiplier(0) == margin_multiplier(1) == margin_multiplier(-1) == 1.0
    assert margin_multiplier(2) == 1.5
    assert margin_multiplier(4) > margin_multiplier(3) > 1.5


def test_elo_ranks_strong_teams_higher_and_probabilities_are_valid():
    df, true_att, _ = simulate_league(rounds=6)
    model = EloModel().fit(with_season(df))
    elo = np.array([model.rating(f"T{i}") for i in range(len(true_att))])
    assert np.corrcoef(elo, true_att)[0, 1] > 0.6
    for home, away in [("T0", "T1"), ("T3", "T7"), ("T0", "NewTeam")]:
        p = model.predict_proba(home, away)
        assert sum(p) == pytest.approx(1)
        assert all(0 < x < 1 for x in p)
    best, worst = f"T{elo.argmax()}", f"T{elo.argmin()}"
    assert model.predict_proba(best, worst)[0] > model.predict_proba(worst, best)[0]


def test_elo_ignores_matches_after_as_of():
    df, _, _ = simulate_league(rounds=2)
    df = with_season(df)
    cutoff = df["date"].iloc[len(df) // 2]
    a = EloModel().fit(df, as_of=cutoff)
    b = EloModel().fit(df[df["date"] < cutoff])
    assert a.ratings_ == b.ratings_


# ---------- Features ----------
from src.features import FEATURE_COLUMNS, build_features  # noqa: E402


def test_form_matches_hand_calculation():
    df = with_season(simulate_league(rounds=2)[0])
    f = build_features(df, window=3)
    row = f[f["home"] == "T0"].iloc[-1]
    before = df[(df["date"] < row["date"]) & ((df["home"] == "T0") | (df["away"] == "T0"))].tail(3)
    gf = np.where(before["home"] == "T0", before["home_goals"], before["away_goals"])
    ga = np.where(before["home"] == "T0", before["away_goals"], before["home_goals"])
    pts = np.select([gf > ga, gf == ga], [3, 1], 0)
    assert row["home_form_pts"] == pytest.approx(pts.mean())
    assert row["home_form_gf"] == pytest.approx(gf.mean())


def test_features_never_see_the_match_itself_or_the_future():
    df = with_season(simulate_league(rounds=2)[0])
    k = len(df) // 2
    before = build_features(df)
    changed = df.copy()
    changed.loc[k:, "home_goals"] += 5          # rewrite the result of match k and every later match
    changed.loc[k:, "result"] = "H"
    after = build_features(changed)
    pd.testing.assert_frame_equal(before.loc[:k, FEATURE_COLUMNS], after.loc[:k, FEATURE_COLUMNS])


def test_first_matches_have_no_form():
    f = build_features(with_season(simulate_league(rounds=1)[0]), window=5)
    assert f[["home_form_pts", "away_form_pts"]].iloc[:5].isna().all().all()


def test_elo_all_teams_start_equal_in_first_season():
    day = pd.Timestamp("2024-08-17")
    df = pd.DataFrame({"date": [day, day, day + pd.Timedelta(days=7)], "home": ["A", "C", "E"],
                       "away": ["B", "D", "F"], "home_goals": [3, 0, 1], "away_goals": [0, 2, 1],
                       "result": ["H", "A", "D"], "season": "2425"})
    f = build_features(df)
    # Nobody had played before these matches, so each pair is level: the difference is just home advantage
    assert (f["elo_diff"] == EloModel().home_adv).all()


# ---------- Machine learning ----------
from src.ml import ALL_FEATURES, build_dataset, fit_predict, logistic_model  # noqa: E402


def two_season_league():
    df1, _, _ = simulate_league(rounds=2, seed=1, start="2022-08-01")
    df2, _, _ = simulate_league(rounds=2, seed=2, start="2023-08-01")
    return pd.concat([df1.assign(season="2223"), df2.assign(season="2324")], ignore_index=True)


def test_dataset_rows_start_at_first_season_and_have_poisson_probabilities():
    data = build_dataset(two_season_league(), "2324")
    assert (data["season"] == "2324").all()
    assert len(data) == 180
    assert data[["pois_h", "pois_d", "pois_a"]].sum(axis=1).to_numpy() == pytest.approx(1)


def test_dataset_never_uses_the_match_itself_or_the_future():
    df = two_season_league()
    # 180 matches per season, one per day; the second season starts at row 180 and is predicted
    # in 7-day blocks. Row 180 + 55 is the last day of a block, so earlier matches of the same
    # block would see its result if the backtest leaked.
    k = 180 + 55
    before = build_dataset(df, "2324")
    changed = df.copy()
    changed.loc[k:, "home_goals"] += 5
    changed.loc[k:, "result"] = "H"
    after = build_dataset(changed, "2324")
    cols = ALL_FEATURES + ["pois_h", "pois_d", "pois_a"]
    rows = before["date"] < df.loc[k, "date"]   # matches played strictly before the changed one
    assert rows.sum() == 55
    pd.testing.assert_frame_equal(before.loc[rows, cols], after.loc[rows, cols])


def test_logistic_model_gives_valid_probabilities():
    data = build_dataset(two_season_league(), "2324")
    p = fit_predict(logistic_model(), data.iloc[:120], data.iloc[120:], ALL_FEATURES)
    assert p.shape == (60, 3)
    assert p.sum(axis=1) == pytest.approx(np.ones(60))
