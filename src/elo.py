"""Elo ratings for football, turned into home / draw / away probabilities.

Ratings are updated after every match, so they react quickly to form:
    expected = 1 / (1 + 10 ** (-(R_home + home_adv - R_away) / 400))
    R_home += k * margin * (actual - expected)        (actual: 1 win, 0.5 draw, 0 loss)
    R_away -= the same amount
`margin` gives bigger wins more weight (as on eloratings.net). Between seasons every
rating moves part of the way back to the average, because squads change in the summer.

Elo only gives a rating difference, not 1X2 probabilities. These come from an ordered
logistic model fitted on past matches: the bigger the difference, the more likely a home
win, with the draw in between.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

START = 1500.0


def margin_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


class EloModel:
    def __init__(self, k: float = 10.0, home_adv: float = 60.0, revert: float = 0.0, burn_in_days: int = 300):
        self.k = k                        # how fast ratings move
        self.home_adv = home_adv          # home advantage, in Elo points
        self.revert = revert              # share of the way back to the average between seasons
        self.burn_in_days = burn_in_days  # early matches (everyone still near 1500) are not used for calibration

    # ---------- ratings ----------
    def _new_team_rating(self) -> float:
        """A team without history (usually just promoted) starts at the average of the 3 lowest ratings."""
        if len(self.ratings_) < 3:
            return START
        return float(np.mean(sorted(self.ratings_.values())[:3]))

    def _run(self, matches: pd.DataFrame) -> np.ndarray:
        """Process matches in date order, updating ratings. Returns the pre-match rating differences."""
        self.ratings_ = {}
        diffs = np.empty(len(matches))
        season = None
        first_season = matches["season"].iloc[0] if len(matches) else None
        for i, m in enumerate(matches.itertuples(index=False)):
            if m.season != season:
                if season is not None:
                    mean = np.mean(list(self.ratings_.values()))
                    self.ratings_ = {t: r + self.revert * (mean - r) for t, r in self.ratings_.items()}
                season = m.season
            for team in (m.home, m.away):
                if team not in self.ratings_:
                    # In the first season of the data every team is new, so all start equal
                    self.ratings_[team] = START if m.season == first_season else self._new_team_rating()

            diff = self.ratings_[m.home] + self.home_adv - self.ratings_[m.away]
            diffs[i] = diff
            expected = 1 / (1 + 10 ** (-diff / 400))
            actual = 1.0 if m.home_goals > m.away_goals else 0.5 if m.home_goals == m.away_goals else 0.0
            change = self.k * margin_multiplier(m.home_goals - m.away_goals) * (actual - expected)
            self.ratings_[m.home] += change
            self.ratings_[m.away] -= change
        return diffs

    # ---------- fitting ----------
    def fit(self, matches: pd.DataFrame, as_of=None) -> "EloModel":
        matches = matches.sort_values("date", kind="stable")
        if as_of is not None:
            matches = matches[matches["date"] < pd.Timestamp(as_of)]
        diffs = self._run(matches)

        # Calibrate rating difference -> 1X2 with an ordered logit, skipping the burn-in period
        keep = (matches["date"] - matches["date"].min()).dt.days.to_numpy() >= self.burn_in_days
        if keep.sum() < 100:
            keep[:] = True
        x = diffs[keep] / 100
        res = matches["result"].to_numpy()[keep]
        is_a, is_d, is_h = (res == "A"), (res == "D"), (res == "H")

        def nll(p):
            c1, gap, beta = p
            c2 = c1 + np.exp(gap)            # second threshold always above the first
            p_a = expit(c1 - beta * x)
            p_ad = expit(c2 - beta * x)
            probs = np.where(is_a, p_a, np.where(is_d, p_ad - p_a, 1 - p_ad))
            return -np.sum(np.log(np.clip(probs, 1e-12, 1)))

        fit = minimize(nll, x0=[-0.5, np.log(1.0), 0.5], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        self.c1_, gap, self.beta_ = fit.x
        self.c2_ = self.c1_ + np.exp(gap)
        return self

    # ---------- prediction ----------
    def rating(self, team: str) -> float:
        return self.ratings_.get(team, self._new_team_rating())

    def predict_proba(self, home: str, away: str):
        """Return (P(home win), P(draw), P(away win))."""
        x = (self.rating(home) + self.home_adv - self.rating(away)) / 100
        p_a = expit(self.c1_ - self.beta_ * x)
        p_ad = expit(self.c2_ - self.beta_ * x)
        return 1 - p_ad, p_ad - p_a, p_a

    def ratings_table(self) -> pd.DataFrame:
        df = pd.DataFrame({"team": list(self.ratings_), "elo": list(self.ratings_.values())})
        return df.sort_values("elo", ascending=False).reset_index(drop=True)
