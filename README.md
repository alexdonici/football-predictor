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
| **Poisson model**             | **1.0387** | **0.6250** | **47.4%** |
| Elo model                     |     1.0458 |  0.6301 |      47.1% |
| Average (Poisson + Elo)       |     1.0399 |  0.6263 |      46.6% |
| Baseline (past H/D/A rates)   |     1.0845 |  0.6562 |      42.6% |

**2024/25** (trained on 2021/22–2023/24, 380 matches)

| Model                         | Log loss ↓ | Brier ↓ | Accuracy ↑ |
|-------------------------------|-----------:|--------:|-----------:|
| Bookmakers (margin removed)   |     0.9706 |  0.5789 |      54.2% |
| **Poisson model**             | **0.9833** | **0.5879** | **53.4%** |
| Elo model                     |     1.0072 |  0.6032 |      52.1% |
| Average (Poisson + Elo)       |     0.9931 |  0.5944 |      52.4% |
| Baseline (past H/D/A rates)   |     1.0826 |  0.6569 |      40.8% |

Using match results only, the Poisson model closes ~89% of the 2024/25 log-loss gap between the naive baseline and the bookmakers. The remaining gap is expected: bookmakers also price in injuries, transfers and team news.

2025/26 was harder to predict for everyone: the bookmakers' log loss also rose from 0.971 to 1.015 (more draws, fewer goals). The Poisson model fell further behind (66% of the gap closed vs 89%), and its log loss rose from 0.99 in August–December to 1.09 in January–May. One hypothesis was that the ratings react too slowly to in-season changes; Elo, which updates after every match, was the first test of that idea (below).

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
| Poisson model                   |               0.9226 |  **0.9875** |  1.0388 |
| LogReg: Poisson only            |               0.9259 |  0.9919 |  1.0375 |
| LogReg: Poisson + Elo + form    |           **0.9178** |  1.0009 |  **1.0357** |
| XGBoost: Poisson + Elo + form   |               0.9301 |  1.0139 |  1.0371 |

Settings were chosen on 2023/24 only, and 2024/25 and 2025/26 were run once. Machine learning does not beat the Poisson model consistently: the logistic regression wins on 2023/24 and 2025/26 but loses clearly on 2024/25, and on average Poisson is still best. XGBoost is the weakest of the ML models: with ~1,500 training matches and features that carry little extra signal, a more flexible model mostly learns noise.

The 2020/21 season is excluded from training: it was played without fans, away teams won more often than home teams (40% vs 38%), and including it made every ML model worse on the validation season.

These numbers come from a longer history (from 2018/19) than the tables above, so the Poisson results differ slightly from them.

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

- **Time decay** `xi = 0.0019`/day ≈ a one-year half-life. This is a trade-off: it makes the ratings slow to react. At the end of 2024/25 the model still ranks Man City above Liverpool, the champions. Tuning `xi` and adding form features is the next step.
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
- [ ] Web dashboard (Django or Streamlit)
- [ ] Docker image and online deployment

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
tests/            offline unit tests
results/          backtest outputs (predictions, metrics)
```

## License

MIT
