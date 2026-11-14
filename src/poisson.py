"""Poisson goals model (Maher / Dixon-Coles style, without the low-score correction).

Each team gets an attack and a defence rating. Expected goals:
    home: exp(mu + home_adv + attack[home] + defence[away])
    away: exp(mu + attack[away] + defence[home])
A higher `defence` value means the team concedes more.
Older matches get less weight (exponential time decay).
Teams with little history (e.g. just promoted) are shrunk towards the level of the
weakest established teams until they have played enough matches.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


class PoissonModel:
    def __init__(self, xi: float = 0.0019, ridge: float = 1e-3, max_goals: int = 10, prior_matches: float = 10.0):
        self.xi = xi              # time decay per day (0.0019 ~ half-life of one year)
        self.ridge = ridge        # small L2 penalty for stability
        self.max_goals = max_goals
        # Newly arrived teams have few matches, so their fitted ratings are noisy.
        # Blend them with a "promoted team" prior worth this many (weighted) matches.
        self.prior_matches = prior_matches

    # ---------- fitting ----------
    def fit(self, matches: pd.DataFrame, as_of=None) -> "PoissonModel":
        as_of = pd.Timestamp(as_of) if as_of is not None else matches["date"].max()
        teams = sorted(set(matches["home"]) | set(matches["away"]))
        self.teams_ = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h = matches["home"].map(self.teams_).to_numpy()
        a = matches["away"].map(self.teams_).to_numpy()
        yh = matches["home_goals"].to_numpy(float)
        ya = matches["away_goals"].to_numpy(float)
        days = (as_of - matches["date"]).dt.days.clip(lower=0).to_numpy(float)
        w = np.exp(-self.xi * days)

        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

        def objective(p):
            att, dfn, home, mu = unpack(p)
            log_lh = mu + home + att[h] + dfn[a]
            log_la = mu + att[a] + dfn[h]
            lh, la = np.exp(log_lh), np.exp(log_la)
            nll = -np.sum(w * (yh * log_lh - lh + ya * log_la - la))
            # identifiability (sum of attacks = 0) + ridge
            s = att.sum()
            nll += 10.0 * s ** 2 + self.ridge * (att @ att + dfn @ dfn)

            rh, ra = w * (lh - yh), w * (la - ya)  # d(nll)/d(log lambda)
            g_att = np.bincount(h, rh, n) + np.bincount(a, ra, n) + 20.0 * s + 2 * self.ridge * att
            g_def = np.bincount(a, rh, n) + np.bincount(h, ra, n) + 2 * self.ridge * dfn
            grad = np.concatenate([g_att, g_def, [rh.sum(), rh.sum() + ra.sum()]])
            return nll, grad

        p0 = np.zeros(2 * n + 2)
        p0[-1] = np.log(max(np.mean(np.r_[yh, ya]), 0.1))
        res = minimize(objective, p0, jac=True, method="L-BFGS-B")
        self.attack_, self.defence_, self.home_adv_, self.mu_ = unpack(res.x)
        self.converged_ = res.success

        # Effective number of matches per team, and which teams are new (first match < 1 year ago)
        self.n_eff_ = np.bincount(h, w, n) + np.bincount(a, w, n)
        first = pd.concat([matches[["home", "date"]].rename(columns={"home": "team"}),
                           matches[["away", "date"]].rename(columns={"away": "team"})]).groupby("team")["date"].min()
        self.newcomer_ = np.array([(as_of - first[t]).days < 365 for t in teams])
        est = ~self.newcomer_ if (~self.newcomer_).sum() >= 3 else np.ones(n, bool)
        strength = self.attack_[est] - self.defence_[est]
        weakest = np.argsort(strength)[:3]
        self.prior_ = (self.attack_[est][weakest].mean(), self.defence_[est][weakest].mean())
        return self

    # ---------- prediction ----------
    def _ratings(self, team):
        prior_att, prior_def = self.prior_
        if team not in self.teams_:
            # Unknown team (usually just promoted): use the average of the 3 weakest established teams
            return prior_att, prior_def
        i = self.teams_[team]
        att, dfn = self.attack_[i], self.defence_[i]
        if self.newcomer_[i] and self.prior_matches > 0:
            k = self.prior_matches / (self.prior_matches + self.n_eff_[i])
            att, dfn = (1 - k) * att + k * prior_att, (1 - k) * dfn + k * prior_def
        return att, dfn

    def expected_goals(self, home: str, away: str):
        ah, dh = self._ratings(home)
        aa, da = self._ratings(away)
        lh = np.exp(self.mu_ + self.home_adv_ + ah + da)
        la = np.exp(self.mu_ + aa + dh)
        return lh, la

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in 0..max_goals."""
        lh, la = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        return np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))

    def predict_proba(self, home: str, away: str):
        """Return (P(home win), P(draw), P(away win))."""
        m = self.score_matrix(home, away)
        m = m / m.sum()
        return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()

    def ratings_table(self) -> pd.DataFrame:
        df = pd.DataFrame({"team": list(self.teams_), "attack": self.attack_, "defence": self.defence_})
        df["strength"] = df["attack"] - df["defence"]
        return df.sort_values("strength", ascending=False).reset_index(drop=True)
