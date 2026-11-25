"""Web dashboard for the football predictor.

    streamlit run streamlit_app.py

Streamlit reruns this whole script on every click, so slow work (downloading data,
fitting models, running the backtest) is wrapped in cached functions and done once.
"""
import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.data import current_season, load_seasons
from src.elo import EloModel
from src.evaluate import average, report, walk_forward
from src.poisson import PoissonModel

LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
HISTORY_SEASONS = 4        # finished seasons used as history
REFRESH_SECONDS = 6 * 3600  # how often the season in progress is re-downloaded

# Chart colours (validated for colour-blind readers on light and dark backgrounds)
COLORS = {"light": {"Poisson model": "#2a78d6", "Bookmakers": "#eb6834"},
          "dark": {"Poisson model": "#3987e5", "Bookmakers": "#d95926"}}
# Heatmap: unlikely scores fade into the page, likely ones stand out (single hue, light to strong)
HEAT = {"light": {"low": "#f1f5fb", "high": "#1f5fae", "gap": "#ffffff", "muted_text": "#52514e"},
        "dark": {"low": "#1b2230", "high": "#3987e5", "gap": "#0e1117", "muted_text": "#c3c2b7"}}


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(code: str) -> str:
    return f"20{code[:2]}/{code[2:]}"


def finished_seasons(n: int) -> list[str]:
    start = 2000 + int(current_season()[:2])
    return [season_code(y) for y in range(start - n, start)]


# ---------- cached data and models ----------
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner="Downloading match data...")
def load_data(league: str) -> tuple[pd.DataFrame, bool]:
    """History plus the season in progress. Returns (matches, includes_current_season)."""
    history = load_seasons(league, finished_seasons(HISTORY_SEASONS))
    try:
        current = load_seasons(league, [current_season()])
        return pd.concat([history, current], ignore_index=True), len(current) > 0
    except requests.RequestException:
        return history, False


@st.cache_resource(ttl=REFRESH_SECONDS, show_spinner="Fitting the models...")
def fit_models(league: str):
    data, _ = load_data(league)
    return PoissonModel().fit(data), EloModel().fit(data)


@st.cache_data(ttl=24 * 3600, show_spinner="Running the backtest (about 10 seconds)...")
def run_backtest(league: str):
    seasons = finished_seasons(HISTORY_SEASONS + 1)
    history, test = load_seasons(league, seasons[:-1]), load_seasons(league, seasons[-1:])
    preds = {"Poisson model": walk_forward(history, test, PoissonModel),
             "Elo model": walk_forward(history, test, EloModel)}
    preds["Average (Poisson + Elo)"] = average(*preds.values())
    return seasons[-1], report(preds, history), preds["Poisson model"]


def theme() -> str:
    try:
        return st.context.theme.type or "dark"
    except AttributeError:
        return "dark"


# ---------- page ----------
st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="centered")
st.title("Football match predictor")
st.caption("Home win / draw / away win probabilities from a Poisson goals model, "
           "tested week by week against bookmaker odds.")

league_name = st.sidebar.selectbox("League", list(LEAGUES.values()))
league = next(code for code, name in LEAGUES.items() if name == league_name)
st.sidebar.markdown(
    "[Source code on GitHub](https://github.com/alexdonici/football-predictor)  \n"
    "Data: [football-data.co.uk](https://www.football-data.co.uk/)")

try:
    data, has_current = load_data(league)
except requests.RequestException:
    st.error("Could not download the match data right now. Please try again later.")
    st.stop()

poisson, elo = fit_models(league)
last_date = data["date"].max()
if not has_current:
    st.warning(f"No data for the {season_label(current_season())} season yet; using finished seasons only.")
st.sidebar.caption(f"Data up to **{last_date:%d %b %Y}** ({len(data):,} matches)")

tab_predict, tab_ratings, tab_backtest, tab_about = st.tabs(
    ["Predict a match", "Team ratings", "How good is it?", "About"])

# ---------- 1. predict a match ----------
with tab_predict:
    latest = data[data["season"] == data["season"].max()]       # current season, or the last finished one
    teams = sorted(set(latest["home"]) | set(latest["away"]))
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams, index=0)
    away = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))

    if home == away:
        st.info("Pick two different teams.")
    else:
        ph, pd_, pa = poisson.predict_proba(home, away)
        lh, la = poisson.expected_goals(home, away)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{ph:.0%}")
        m2.metric("Draw", f"{pd_:.0%}")
        m3.metric(f"{away} win", f"{pa:.0%}")
        st.write(f"Expected goals: **{home} {lh:.2f} – {la:.2f} {away}**")

        # Most likely exact scores, as a heatmap (rows: home goals, columns: away goals)
        m = poisson.score_matrix(home, away)
        m = m / m.sum()
        n = 6
        cells = pd.DataFrame([(i, j, m[i, j]) for i in range(n) for j in range(n)],
                             columns=["home_goals", "away_goals", "p"])
        cells["score"] = cells["home_goals"].astype(str) + "-" + cells["away_goals"].astype(str)
        cells["label"] = np.where(cells["p"] >= 0.04, (cells["p"] * 100).round(0).astype(int).astype(str) + "%", "")
        base = alt.Chart(cells).encode(
            x=alt.X("away_goals:O", title=f"{away} goals", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("home_goals:O", title=f"{home} goals"),
        )
        hc = HEAT[theme()]
        heat = base.mark_rect(cornerRadius=3, stroke=hc["gap"], strokeWidth=2).encode(
            color=alt.Color("p:Q", scale=alt.Scale(range=[hc["low"], hc["high"]], domain=[0, float(m.max())]),
                            legend=None),
            tooltip=[alt.Tooltip("score:N", title="Score"), alt.Tooltip("p:Q", title="Probability", format=".1%")],
        )
        text = base.mark_text(fontSize=12).encode(
            text="label:N",
            color=alt.condition(alt.datum.p > m.max() * 0.5, alt.value("white"), alt.value(hc["muted_text"])),
        )
        st.markdown("**Most likely scores**")
        st.altair_chart((heat + text).properties(height=300), width="stretch")

        with st.expander("Second opinion: Elo model"):
            eh, ed, ea = elo.predict_proba(home, away)
            st.write(f"Elo ratings: {home} **{elo.rating(home):.0f}**, {away} **{elo.rating(away):.0f}**")
            st.write(f"Home {eh:.0%} · Draw {ed:.0%} · Away {ea:.0%}")
            st.caption("Elo is less accurate than the Poisson model in the backtests, so it is shown only for comparison.")

# ---------- 2. team ratings ----------
with tab_ratings:
    st.markdown("Team strength = attack rating − defence rating (Poisson model). "
                f"Higher is stronger. Teams of the {season_label(data['season'].max())} season.")
    table = poisson.ratings_table().merge(elo.ratings_table(), on="team")
    table = table[table["team"].isin(teams)].reset_index(drop=True)
    bars = alt.Chart(table).mark_bar(cornerRadiusEnd=4, color=COLORS[theme()]["Poisson model"]).encode(
        x=alt.X("strength:Q", title="Strength"),
        y=alt.Y("team:N", sort="-x", title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=200, labelPadding=6)),   # show every team name
        tooltip=["team", alt.Tooltip("attack:Q", format=".2f"), alt.Tooltip("defence:Q", format=".2f"),
                 alt.Tooltip("strength:Q", format=".2f"), alt.Tooltip("elo:Q", format=".0f", title="Elo")],
    ).properties(height=28 * len(table))
    st.altair_chart(bars, width="stretch")
    with st.expander("Table"):
        st.dataframe(table.round({"attack": 3, "defence": 3, "strength": 3, "elo": 0}),
                     hide_index=True, width="stretch")

# ---------- 3. backtest ----------
with tab_backtest:
    test_season, metrics, preds = run_backtest(league)
    st.markdown(
        f"Every match of **{season_label(test_season)}** was predicted using only the matches played before it, "
        "refitting the model every week. **Lower log loss and Brier = better probabilities.**")
    st.dataframe(metrics.style.format({"log loss": "{:.4f}", "Brier": "{:.4f}", "accuracy": "{:.1%}"}),
                 width="stretch")

    if {"odds_h", "odds_d", "odds_a"} <= set(preds.columns):
        from src.evaluate import implied_probs, one_hot

        ok = preds[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)
        p = preds[ok].reset_index(drop=True)
        y = one_hot(p["result"])
        ll_model = -np.log((p[["p_h", "p_d", "p_a"]].to_numpy() * y).sum(axis=1))
        ll_books = -np.log((implied_probs(p) * y).sum(axis=1))
        n = np.arange(1, len(p) + 1)
        curve = pd.concat([
            pd.DataFrame({"date": p["date"], "match": n, "log loss": np.cumsum(ll_model) / n, "series": "Poisson model"}),
            pd.DataFrame({"date": p["date"], "match": n, "log loss": np.cumsum(ll_books) / n, "series": "Bookmakers"}),
        ])
        # The first rounds average only a handful of matches and swing wildly, so start after 40 matches
        curve = curve[curve["match"] > 40]
        # Several matches share a date; keep the value at the end of each day so the line doesn't zigzag
        curve = curve.groupby(["series", "date"], as_index=False).last()
        colors = COLORS[theme()]
        lines = alt.Chart(curve).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("log loss:Q", title="Average log loss so far", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("date:T", title="Date"), "series:N",
                     alt.Tooltip("match:Q", title="Matches so far"), alt.Tooltip("log loss:Q", format=".4f")],
        )
        st.markdown("**Model vs bookmakers through the season**")
        st.altair_chart(lines.properties(height=300), width="stretch")
        st.caption("Running average of log loss since the start of the season (from the 41st match on): "
                   "where the blue line is above the orange one, the bookmakers' probabilities were better so far.")
    else:
        st.info("No bookmaker odds in the data for this league, so only the models are compared.")

# ---------- 4. about ----------
with tab_about:
    st.markdown("""
**How it works**

- **Poisson goals model.** Each team has an attack and a defence rating, fitted on past matches, with recent
  matches counting more (one-year half-life). Ratings are fitted on 75% real goals and 25% "deserved" goals
  from shots on target, because goals carry a lot of luck. Expected goals for both teams give the probability
  of every exact score, and adding those up gives home win / draw / away win.
- **Elo model.** One rating per team, updated after each match.
- **Testing.** Walk-forward backtest: each week is predicted only from earlier matches, then compared with the
  bookmakers' odds (margin removed).

**What did not help:** recent form, Elo on top of Poisson, logistic regression and XGBoost, the Dixon-Coles
correction, corners and off-target shots. Details and numbers are in the
[README](https://github.com/alexdonici/football-predictor#readme).

**Limits:** the model only knows results and match statistics, not injuries, line-ups, transfers or
manager changes, which the bookmakers do price in. It is not a betting tool.
""")
