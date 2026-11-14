"""Download and clean historical match data from football-data.co.uk."""
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Pre-match odds columns, in order of preference: market average, then Bet365.
# (Closing odds - AvgCH/AvgCD/AvgCA - are a harder benchmark; see README.)
ODDS_SETS = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]

# The file of the season in progress gets new matches every few days, so its cache expires.
CURRENT_SEASON_MAX_AGE_HOURS = 12


def current_season(today: date | None = None) -> str:
    """Season code of the season in progress, e.g. '2627' for any date from July 2026 to June 2027."""
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def download_season(league: str, season: str, force: bool = False) -> Path:
    """Download one season (e.g. league='E0', season='2425') and cache it locally.

    Finished seasons are downloaded once; the current season is re-downloaded when the cache is stale.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{league}_{season}.csv"
    if path.exists() and not force:
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if season != current_season() or age_hours < CURRENT_SEASON_MAX_AGE_HOURS:
            return path
    url = BASE_URL.format(season=season, league=league)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def parse_dates(s: pd.Series) -> pd.Series:
    """football-data.co.uk uses dd/mm/yy or dd/mm/yyyy; some mirrors use ISO yyyy-mm-dd.

    Passing dayfirst=True to ISO dates silently swaps day and month, so detect the format first.
    """
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[iso] = pd.to_datetime(s[iso], format="%Y-%m-%d")
    out[~iso] = pd.to_datetime(s[~iso], dayfirst=True, format="mixed")
    return out


def load_season(league: str, season: str) -> pd.DataFrame:
    """Load one season and keep only the columns the models need."""
    path = download_season(league, season)
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    odds_cols = next((cols for cols in ODDS_SETS if all(c in df.columns for c in cols)), None)
    out = pd.DataFrame({
        "date": parse_dates(df["Date"]),
        "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
        "home_goals": df["FTHG"].astype(int),
        "away_goals": df["FTAG"].astype(int),
        "result": df["FTR"],  # H / D / A
    })
    if odds_cols:
        out["odds_h"], out["odds_d"], out["odds_a"] = (df[c] for c in odds_cols)
    out["season"] = season
    return out


def load_seasons(league: str, seasons: list[str]) -> pd.DataFrame:
    """Load several seasons into one DataFrame, sorted by date."""
    frames = [load_season(league, s) for s in seasons]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
