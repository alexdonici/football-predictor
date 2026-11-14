# Football Match Outcome Prediction

[![tests](https://github.com/alexdonici/football-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/alexdonici/football-predictor/actions/workflows/tests.yml)

Predicts the probabilities of home win / draw / away win (1/X/2) for football matches and benchmarks them against bookmaker odds.

The first model is a **Poisson goals model**: every team gets an attack and a defence rating, fitted by maximum likelihood on historical results, with recent matches weighted more (exponential time decay). Match probabilities come from the distribution of possible scores.

## Results (Premier League, walk-forward)

Each season is predicted week by week, using only matches played before each round.

**2025/26** (trained on 2021/22–2024/25, 380 matches)

| Model                         | Log loss ↓ | Brier ↓ | Accuracy ↑ |
|-------------------------------|-----------:|--------:|-----------:|
| Bookmakers (margin removed)   |     1.0153 |  0.6100 |      49.5% |
| **Poisson model**             | **1.0387** | **0.6250** |**47.4%**|
| Baseline (past H/D/A rates)   |     1.0845 |  0.6562 |      42.6% |

**2024/25** (trained on 2021/22–2023/24, 380 matches)

| Model                         | Log loss ↓ | Brier ↓ | Accuracy ↑ |
|-------------------------------|-----------:|--------:|-----------:|
| Bookmakers (margin removed)   |     0.9706 |  0.5789 |      54.2% |
| **Poisson model**             | **0.9833** | **0.5879** | **53.4%** |
| Baseline (past H/D/A rates)   |     1.0826 |  0.6569 |      40.8% |

Using match results only, the Poisson model closes ~89% of the 2024/25 log-loss gap between the naive baseline and the bookmakers. The remaining gap is expected: bookmakers also price in injuries, transfers and team news.

2025/26 was harder to predict for everyone: the bookmakers' log loss also rose from 0.971 to 1.015 (more draws, fewer goals). The Poisson model fell further behind (66% of the gap closed vs 89%), and its log loss rose from 0.99 in August–December to 1.09 in January–May. The ratings react too slowly to in-season changes, which is what Elo and recent-form features (next step) should address.

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

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Tests run offline on simulated leagues: parameter recovery, valid probabilities, metric sanity checks, date parsing, current-season detection and a no-look-ahead check of the backtest. They run on every push via GitHub Actions.

## Roadmap

- [x] Data loading and cleaning
- [x] Poisson model with time decay
- [x] Walk-forward backtest vs bookmakers
- [ ] Elo ratings and recent form features
- [ ] Machine learning model (logistic regression / XGBoost)
- [ ] Web dashboard (Django or Streamlit)
- [ ] Docker image and online deployment

## Project structure

```
src/data.py       download and clean match data
src/poisson.py    Poisson goals model
src/evaluate.py   walk-forward backtest and metrics
main.py           command-line entry point
tests/            offline unit tests
results/          backtest outputs (predictions, metrics)
```

## License

MIT
