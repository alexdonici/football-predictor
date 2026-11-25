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
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson


def dc_tau(x, y, lh, la, rho):
    """Dixon-Coles adjustment factor for low scores (1 for every other score).

    A negative rho makes 0-0 and 1-1 more likely and 1-0 / 0-1 less likely.
    """
    x, y = np.asarray(x), np.asarray(y)
    tau = np.ones(np.broadcast(x, y, lh, la).shape)
    tau = np.where((x == 0) & (y == 0), 1 - lh * la * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lh * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + la * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


class PoissonModel:
    def __init__(self, xi: float = 0.0019, ridge: float = 1e-3, max_goals: int = 10, prior_matches: float = 10.0,
                 dixon_coles: bool = False, shots_weight: float = 0.25,
                 off_target_value: float = 0.0, corner_value: float = 0.0):
        self.xi = xi              # time decay per day (0.0019 ~ half-life of one year)
        self.ridge = ridge        # small L2 penalty for stability
        self.max_goals = max_goals
        # Newly arrived teams have few matches, so their fitted ratings are noisy.
        # Blend them with a "promoted team" prior worth this many (weighted) matches.
        self.prior_matches = prior_matches
        # Dixon-Coles: correct the probabilities of 0-0, 1-0, 0-1 and 1-1 with one extra parameter (rho)
        self.dixon_coles = dixon_coles
        # Shots on target: fit on a mix of real goals and "deserved" goals (shots on target x conversion rate).
        # 0 = goals only, 1 = shots only. Goals carry a lot of luck; shots on target are steadier.
        self.shots_weight = shots_weight
        # How much an off-target shot and a corner count, relative to a shot on target, in "deserved" goals
        self.off_target_value = off_target_value
        self.corner_value = corner_value

    # ---------- fitting ----------
    def fit(self, matches: pd.DataFrame, as_of=None) -> "PoissonModel":
        as_of = pd.Timestamp(as_of) if as_of is not None else matches["date"].max()
        teams = sorted(set(matches["home"]) | set(matches["away"]))
        self.teams_ = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h = matches["home"].map(self.teams_).to_numpy()
        a = matches["away"].map(self.teams_).to_numpy()
        goals_h = matches["home_goals"].to_numpy(float)
        goals_a = matches["away_goals"].to_numpy(float)
        days = (as_of - matches["date"]).dt.days.clip(lower=0).to_numpy(float)
        w = np.exp(-self.xi * days)
        yh, ya = self._target(matches, goals_h, goals_a, w)

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

        # Dixon-Coles rho, fitted after the ratings (a common two-step simplification)
        self.rho_ = 0.0
        if self.dixon_coles:
            lh = np.exp(self.mu_ + self.home_adv_ + self.attack_[h] + self.defence_[a])
            la = np.exp(self.mu_ + self.attack_[a] + self.defence_[h])
            low = (goals_h <= 1) & (goals_a <= 1)
            lh, la, xh, xa, wl = lh[low], la[low], goals_h[low], goals_a[low], w[low]

            def neg_ll(rho):
                tau = dc_tau(xh, xa, lh, la, rho)
                return np.inf if (tau <= 0).any() else -np.sum(wl * np.log(tau))

            self.rho_ = minimize_scalar(neg_ll, bounds=(-0.3, 0.3), method="bounded").x
        return self

    def _chances(self, matches, side: str) -> np.ndarray:
        """Chance index for one side: shots on target + weighted off-target shots + weighted corners."""
        sot = matches[f"{side}_sot"].to_numpy(float)
        out = sot.copy()
        if self.off_target_value:
            shots = matches[f"{side}_shots"].to_numpy(float) if f"{side}_shots" in matches else np.full_like(sot, np.nan)
            out = out + self.off_target_value * (shots - sot)
        if self.corner_value:
            corners = matches[f"{side}_corners"].to_numpy(float) if f"{side}_corners" in matches else np.full_like(sot, np.nan)
            out = out + self.corner_value * corners
        return out

    def _target(self, matches, goals_h, goals_a, w):
        """What the ratings are fitted on: real goals, optionally mixed with 'deserved' goals from match statistics."""
        if self.shots_weight <= 0 or "home_sot" not in matches:
            return goals_h, goals_a
        ch_h, ch_a = self._chances(matches, "home"), self._chances(matches, "away")
        ok = ~np.isnan(ch_h) & ~np.isnan(ch_a)
        if ok.sum() == 0:
            return goals_h, goals_a
        # Conversion rate: goals per unit of the chance index in this data (about 0.32 per shot on target)
        conv = (w[ok] * (goals_h[ok] + goals_a[ok])).sum() / (w[ok] * (ch_h[ok] + ch_a[ok])).sum()
        s = self.shots_weight
        yh = np.where(ok, (1 - s) * goals_h + s * conv * np.nan_to_num(ch_h), goals_h)
        ya = np.where(ok, (1 - s) * goals_a + s * conv * np.nan_to_num(ch_a), goals_a)
        return yh, ya

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
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        if self.rho_:
            m[:2, :2] *= dc_tau(np.array([[0, 0], [1, 1]]), np.array([[0, 1], [0, 1]]), lh, la, self.rho_)
        return m

    def predict_proba(self, home: str, away: str):
        """Return (P(home win), P(draw), P(away win))."""
        m = self.score_matrix(home, away)
        m = m / m.sum()
        return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()

    def ratings_table(self) -> pd.DataFrame:
        df = pd.DataFrame({"team": list(self.teams_), "attack": self.attack_, "defence": self.defence_})
        df["strength"] = df["attack"] - df["defence"]
        return df.sort_values("strength", ascending=False).reset_index(drop=True)
