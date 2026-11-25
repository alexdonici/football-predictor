# Football Match Outcome Prediction

[![tests](https://github.com/alexdonici/football-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/alexdonici/football-predictor/actions/workflows/tests.yml)

Predicts the probabilities of home win / draw / away win (1/X/2) for football matches and benchmarks them against bookmaker odds.

Two hand-built models, plus machine-learning models that combine them:
- **Poisson goals model**: every team gets an attack and a defence rating, fitted by maximum likelihood on historical results, with recent matches weighted more (exponential time decay). Match probabilities come from the distribution of possible scores.
- **Elo ratings**: every team has one rating, updated after each match (bigger wins move it more). The rating difference is turned into home/draw/away probabilities with an ordered logistic model fitted on past matches.
- **Machine learning** (logistic regression, XGBoost): learns how to combine the Poisson probabilities, Elo and recent form.

## Results (Premier League, walk-forward)

Each season is predicted week by week, using only matches played before each round.

**2025/26** (trained on 2021/22–2024/25, 380 matches)

| Model                         | Log loss ↓ | Brier ↓ | Accuracy ↑ |
|-------------------------------|-----------:|--------:|-----------:|
| Bookmakers (margin removed)   |     1.0153 |  0.6100 |      49.5% |
| **Poisson model**             | **1.0343** | **0.6218** | **47.6%** |
| Elo model                     |     1.0458 |  0.6301 |      47.1% |
| Average (Poisson + Elo)       |     1.0376 |  0.6247 |      47.6% |
| Baseline (past H/D/A rates)   |     1.0845 |  0.6562 |      42.6% |

**2024/25** (trained on 2021/22–2023/24, 380 matches)

| Model                         | Log loss ↓ | Brier ↓ | Accuracy ↑ |
|-------------------------------|-----------:|--------:|-----------:|
| Bookmakers (margin removed)   |     0.9706 |  0.5789 |      54.2% |
| **Poisson model**             | **0.9835** | **0.5876** | **52.9%** |
| Elo model                     |     1.0072 |  0.6032 |      52.1% |
| Average (Poisson + Elo)       |     0.9928 |  0.5941 |      52.6% |
| Baseline (past H/D/A rates)   |     1.0826 |  0.6569 |      40.8% |

Using match results only, the Poisson model closes ~88% of the 2024/25 log-loss gap between the naive baseline and the bookmakers. The remaining gap is expected: bookmakers also price in injuries, transfers and team news.

2025/26 was harder to predict for everyone: the bookmakers' log loss also rose from 0.971 to 1.015 (more draws, fewer goals). The Poisson model fell further behind (73% of the gap closed vs 88%), and its log loss rose from 0.99 in August–December to 1.08 in January–May. One hypothesis was that the ratings react too slowly to in-season changes; Elo, which updates after every match, was the first test of that idea (below).

**Improving the Poisson model.** Three ideas were tried, each tuned on 2023/24 and then checked once on the two test seasons:

| Idea | Result |
|------|--------|
| Tune the time decay (how fast old matches are forgotten) | No gain: the original one-year half-life was already optimal (0.9235 vs 0.9233 at best). Forgetting faster is clearly worse. |
| Dixon-Coles correction for low scores (0-0, 1-0, 0-1, 1-1) | Negligible (0.9230 vs 0.9235). The correction assumes more 0-0 draws than independent Poisson predicts, but in today's Premier League 0-0 is *rarer* than predicted (5.2% vs 6.2% of matches), so its parameter comes out near zero. Kept as an option, off by default. |
| **Shots on target**: fit the ratings on 75% real goals + 25% "deserved" goals (shots on target × the league's conversion rate, ≈0.32) | **Kept.** 2023/24: 0.9235 → 0.9224; 2024/25: unchanged (0.9833 → 0.9835); 2025/26: 1.0387 → **1.0343**, the share of the gap to the bookmakers closed rising from 66% to 73%. Goals carry a lot of luck; shots on target show more steadily who created chances. Using shots only (100%) is worse than goals only. |

| Also use off-target shots and corners in the "deserved" goals | No gain on 2023/24 (best: 0.9224 with shots on target only). Past off-target shots and corners do predict future goals on their own, but the Poisson model already captures that through shots on target. As extra ML features they gave a tiny gain on 2023/24 (0.9169 → 0.9153), too small to be worth it. Available as options (`off_target_value`, `corner_value`), off by default. |

The tables above use the final model (with shots on target).

**Elo vs Poisson.** Elo was expected to help because it reacts faster, but it is worse than Poisson in both test seasons, and averaging the two does not beat Poisson alone. Two lessons:
- Tuning on 2023/24 picked a *slow* Elo (K = 10, no reset between seasons): football results are noisy, so reacting strongly to each match hurts.
- Poisson uses the goals scored and conceded, while Elo mostly uses win/draw/loss, so Poisson simply sees more information.

Elo is still kept: it is a compact strength measure and will be one of the input features for the machine-learning models.

**Does recent form help?** Form features (points and goal difference per match over the last 5 matches, and home-only / away-only form) were tested with a logistic regression trained on past seasons (`python -m experiments.form_vs_elo`):

| Log loss                   | 2023/24 | 2024/25 | 2025/26 |
|----------------------------|--------:|--------:|--------:|
| Elo only                   |  0.9242 |  1.0086 |  1.0476 |
| Elo + form                 |  0.9231 |  1.0079 |  1.0430 |
| Elo + form + venue form    |  0.9241 |  1.0130 |  1.0425 |
| Form only (no Elo)         |  0.9825 |  1.0317 |  1.0525 |

Form adds at most ~0.005 on top of Elo, which is within noise, and home/away form adds nothing. Elo is already a weighted summary of past results, so the last five matches say little that is new.

**Can machine learning do better?** A logistic regression and an XGBoost model were trained on a table with one row per match: the Poisson probabilities, the Elo difference and the form features, all as they were *before* kick-off. Each test season is predicted by a model trained on all earlier seasons (`python -m experiments.ml_vs_poisson`):

| Log loss                        | 2023/24 (validation) | 2024/25 | 2025/26 |
|---------------------------------|---------------------:|--------:|--------:|
| Poisson model                   |               0.9217 |  **0.9878** |  1.0343 |
| LogReg: Poisson only            |               0.9219 |  0.9931 |  1.0347 |
| LogReg: Poisson + Elo + form    |           **0.9169** |  1.0001 |  **1.0335** |
| XGBoost: Poisson + Elo + form   |               0.9243 |  1.0093 |  1.0359 |

Settings were chosen on 2023/24 only, and 2024/25 and 2025/26 were run once. Machine learning does not beat the Poisson model consistently: the logistic regression wins on 2023/24 and (barely) on 2025/26 but loses clearly on 2024/25, and on average Poisson is still best. XGBoost is the weakest of the ML models: with ~1,500 training matches and features that carry little extra signal, a more flexible model mostly learns noise.

The 2020/21 season is excluded from training: it was played without fans, away teams won more often than home teams (40% vs 38%), and including it made every ML model worse on the validation season.

These numbers come from a longer history (from 2018/19) than the tables above, so the Poisson results differ slightly from them.

## Web dashboard

```bash
streamlit run streamlit_app.py
```

Pick two teams to see the home/draw/away probabilities, expected goals and the most likely exact scores; browse the team ratings; and see how the model did against the bookmakers in the last finished season, week by week. It covers the five big European leagues and includes the matches already played in the current season.

### With Docker

```bash
docker build -t football-predictor .
docker run -p 8501:8501 football-predictor
```

Then open http://localhost:8501. The image runs as a non-root user and has a health check; GitHub Actions builds it and checks that the dashboard starts on every push.

## Data

Historical results and odds from [football-data.co.uk](https://www.football-data.co.uk/), downloaded automatically and cached in `data/raw/`. Finished seasons are downloaded once; the season in progress is refreshed when the cache is older than 12 hours.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                                   # backtest Premier League 2025/26
python main.py --test 2425 --train 2122 2223 2324  # backtest 2024/25
python main.py --league SP1                      # La Liga
python main.py --predict Arsenal "Man United"    # next match; quote names with spaces
python main.py --export-features                 # pre-match features as CSV
```

Backtest predictions and metrics are saved to `results/`. `--predict` also uses the matches already played in the current season.

## How it is evaluated

Walk-forward backtest: the model is refitted every week using only matches played **before** the ones it predicts, so there is no look-ahead (this is checked by a unit test).

Metrics:
- **Log loss** – penalises confident wrong predictions (lower is better)
- **Brier score** – squared error of the probabilities (lower is better)
- **Accuracy** – how often the most likely outcome happened

Benchmarks: bookmaker implied probabilities (margin removed by normalisation) and a naive baseline using historical home/draw/away rates.

## Design notes

- **Time decay** `xi = 0.0019`/day ≈ a one-year half-life. Tuning it on 2023/24 confirmed this value; faster forgetting makes predictions worse, even though slow ratings can look odd (at the end of 2024/25 the model still ranked Man City above Liverpool, the champions).
- **Promoted teams.** A team with no history starts at the level of the three weakest established teams. Without extra care, a newly promoted team's rating swings to extremes after one or two results (e.g. Ipswich got a 0.03% away-win chance at Man City after losing their opener). Ratings of new teams are therefore blended with that prior, weighted like 10 matches. The weight was chosen on 2023/24 (validation) and then checked once on 2024/25: metrics unchanged, extreme probabilities gone (minimum outcome probability 0.03% → 0.7%).
- **Date parsing** handles both `dd/mm/yy` (football-data.co.uk) and ISO `yyyy-mm-dd` (some mirrors). Parsing ISO dates with `dayfirst=True` silently swaps day and month and breaks the chronological order of the backtest.
- The optimiser uses an analytic gradient (verified against finite differences).
- **Elo settings** were tuned on 2023/24 only (grid over K and between-season reversion) and then applied unchanged to 2024/25 and 2025/26. New teams start at the average of the three lowest ratings. The first 300 days of data are not used to calibrate the rating difference → 1X2 mapping, because all teams still sit near the starting rating.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Tests run offline on simulated leagues: parameter recovery, valid probabilities, metric sanity checks, date parsing, current-season detection, Elo behaviour, form calculations, ML probabilities and no-look-ahead checks of the backtest, Elo, the features and the ML training table. They run on every push via GitHub Actions.

## Roadmap

- [x] Data loading and cleaning
- [x] Poisson model with time decay
- [x] Walk-forward backtest vs bookmakers
- [x] Elo ratings
- [x] Recent form features (little value beyond Elo)
- [x] Machine learning models (logistic regression / XGBoost): no consistent gain over Poisson
- [x] Poisson improvements: shots on target (kept), Dixon-Coles and time-decay tuning (no gain)
- [x] Web dashboard (Streamlit)
- [x] Docker image (built and smoke-tested in CI)
- [ ] Online deployment

## Project structure

```
src/data.py       download and clean match data
src/poisson.py    Poisson goals model
src/elo.py        Elo ratings + ordered logit for 1X2
src/features.py   pre-match features (Elo, form) for ML models
src/ml.py         ML training table, logistic regression and XGBoost
experiments/      reproducible one-off analyses
src/evaluate.py   walk-forward backtest and metrics
main.py           command-line entry point
streamlit_app.py  web dashboard
Dockerfile        container image for the dashboard
tests/            offline unit tests
results/          backtest outputs (predictions, metrics)
```

## License

MIT
