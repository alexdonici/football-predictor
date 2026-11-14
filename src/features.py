"""Pre-match features for every match, built only from matches played before it.

These are the inputs for the machine-learning models:
- Elo ratings of both teams just before kick-off
- recent form: points, goals scored and conceded per match over the last N matches
- venue form: the home team's recent home matches, the away team's recent away matches

Every rolling value is shifted by one match, so a match never sees its own result.
Teams without enough history get NaN (for example a promoted team in its first matches).
"""
import numpy as np
import pandas as pd

from .elo import EloModel


def _team_view(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (two rows per match), with that team's goals and points."""
    m = matches.reset_index(drop=True).assign(match_id=lambda d: d.index)
    home = pd.DataFrame({"match_id": m["match_id"], "date": m["date"], "team": m["home"], "venue": "home",
                         "gf": m["home_goals"], "ga": m["away_goals"]})
    away = pd.DataFrame({"match_id": m["match_id"], "date": m["date"], "team": m["away"], "venue": "away",
                         "gf": m["away_goals"], "ga": m["home_goals"]})
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.select([long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], 0)
    return long.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)


def _rolling_before(df: pd.DataFrame, by: list[str], cols: list[str], window: int) -> pd.DataFrame:
    """Mean of `cols` over the previous `window` rows of each group, excluding the current row."""
    grouped = df.groupby(by, sort=False)[cols]
    return grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())


def build_features(matches: pd.DataFrame, window: int = 5, venue_window: int = 5, **elo_kwargs) -> pd.DataFrame:
    """Return `matches` (sorted by date) with pre-match feature columns added."""
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)

    # Elo just before each match (the rating difference already includes home advantage)
    elo = EloModel(**elo_kwargs)
    elo_diff = elo._run(matches)
    out = matches.copy()
    out["elo_diff"] = elo_diff

    # Recent form over all matches, and venue-specific form
    long = _team_view(matches)
    long[["form_pts", "form_gf", "form_ga"]] = _rolling_before(long, ["team"], ["pts", "gf", "ga"], window)
    long["venue_pts"] = _rolling_before(long, ["team", "venue"], ["pts"], venue_window)["pts"]

    feats = ["form_pts", "form_gf", "form_ga", "venue_pts"]
    h = long[long["venue"] == "home"].set_index("match_id")[feats].add_prefix("home_")
    a = long[long["venue"] == "away"].set_index("match_id")[feats].add_prefix("away_")
    out = out.join(h).join(a)

    # Differences are often more useful to a model than the raw pairs
    out["form_pts_diff"] = out["home_form_pts"] - out["away_form_pts"]
    out["form_gd_diff"] = (out["home_form_gf"] - out["home_form_ga"]) - (out["away_form_gf"] - out["away_form_ga"])
    out["venue_pts_diff"] = out["home_venue_pts"] - out["away_venue_pts"]
    return out


FEATURE_COLUMNS = [
    "elo_diff",
    "home_form_pts", "home_form_gf", "home_form_ga", "home_venue_pts",
    "away_form_pts", "away_form_gf", "away_form_ga", "away_venue_pts",
    "form_pts_diff", "form_gd_diff", "venue_pts_diff",
]
