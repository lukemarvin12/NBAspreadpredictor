# src/enrich_with_nba_api_advanced.py

"""
Enrich raw odds data with per-game team stats from nba_api.

Input:
  data/raw/nba_2018-2025.xlsx
    (must contain at least: season, date, home, away)

Output:
  data/processed/nba_2018-2025_nbaapi_stats.xlsx
    (sidecar file: season/date/home/away + home/away box-score stats)
"""

import os
import time
import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

# ---- PATHS -------------------------------------------------------------------

INPUT_PATH = "data/raw/nba_2018-2025.xlsx"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nba_2018-2025_nbaapi_stats.xlsx")

# ---- TEAM ABBREV MAP ---------------------------------------------------------

TEAM_MAP = {
    # West
    "sa": "SAS",
    "sas": "SAS",
    "por": "POR",
    "gs": "GSW",
    "gsw": "GSW",
    "lal": "LAL",
    "lac": "LAC",
    "hou": "HOU",
    "dal": "DAL",
    "den": "DEN",
    "utah": "UTA",
    "uta": "UTA",
    "mem": "MEM",
    "min": "MIN",
    "okc": "OKC",
    "nop": "NOP",
    "no": "NOP",
    "sac": "SAC",
    "phx": "PHX",

    # East
    "tor": "TOR",
    "phi": "PHI",
    "ind": "IND",
    "wsh": "WAS",
    "was": "WAS",
    "orl": "ORL",
    "mil": "MIL",
    "bkn": "BKN",
    "nj": "NJN",  # if old Nets code ever appears
    "chi": "CHI",
    "cle": "CLE",
    "bos": "BOS",
    "mia": "MIA",
    "atl": "ATL",
    "ny": "NYK",
    "nyk": "NYK",
    "cha": "CHA",
    "det": "DET",
}


def season_year_to_nba_season_str(season_year: int) -> str:
    """Convert 2018 -> '2018-19'."""
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def load_league_game_logs_for_season(season_year: int) -> pd.DataFrame:
    """
    Call nba_api once per season to get team game logs (team-level box scores).
    """
    season_str = season_year_to_nba_season_str(season_year)
    print(f"Fetching LeagueGameLog for season {season_str}...")

    # Your version of nba_api doesn't accept player_or_team -> use defaults
    logs = LeagueGameLog(
        season=season_str,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["TEAM_ABBREVIATION"] = logs["TEAM_ABBREVIATION"].astype(str)
    logs["MATCHUP"] = logs["MATCHUP"].astype(str)

    # "LAL vs BOS" => home, "LAL @ BOS" => away
    logs["is_home"] = logs["MATCHUP"].str.contains(" vs ").astype(int)

    return logs


def map_team_tag(tag: str) -> str:
    """
    Map your odds file team tag (like 'sa', 'gs', 'utah') to NBA official code.
    """
    if tag is None or (isinstance(tag, float) and pd.isna(tag)):
        return None
    t = str(tag).strip().lower()
    if t not in TEAM_MAP:
        raise KeyError(
            f"Unknown team tag '{tag}'. Add it to TEAM_MAP in enrich_with_nba_api_advanced.py."
        )
    return TEAM_MAP[t]


def enrich_with_logs(df: pd.DataFrame, logs_by_season: dict) -> pd.DataFrame:
    """
    For each odds row (one game), attach home/away team stats from LeagueGameLog.

    Expects df to have columns: season, date, home, away.
    Returns a NEW dataframe with ONLY:
      season, date, home, away, + home_* and away_* stats.
    """
    # Normalize
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_nba"] = df["home"].apply(map_team_tag)
    df["away_nba"] = df["away"].apply(map_team_tag)

    # Stats to pull from nba_api team logs
    stat_cols = ["PTS", "FGA", "FTA", "OREB", "TOV", "FG3M", "FG3A"]

    # Create a new dataframe with just keys + stats
    out_cols = ["season", "date", "home", "away"]
    out = df[out_cols].copy()

    for prefix in ["home_", "away_"]:
        for stat in stat_cols:
            out[prefix + stat.lower()] = pd.NA

    logs_cache = {season: logs.copy() for season, logs in logs_by_season.items()}

    for idx, row in df.iterrows():
        season_year = int(row["season"])
        game_date = row["date"]
        home_code = row["home_nba"]
        away_code = row["away_nba"]

        if season_year not in logs_cache:
            continue

        logs = logs_cache[season_year]

        home_mask = (
            (logs["GAME_DATE"] == game_date)
            & (logs["TEAM_ABBREVIATION"] == home_code)
            & (logs["is_home"] == 1)
        )
        away_mask = (
            (logs["GAME_DATE"] == game_date)
            & (logs["TEAM_ABBREVIATION"] == away_code)
            & (logs["is_home"] == 0)
        )

        home_logs = logs.loc[home_mask]
        away_logs = logs.loc[away_mask]

        if len(home_logs) != 1 or len(away_logs) != 1:
            # Could be preseason/in-season/bubble quirks; skip quietly.
            continue

        h = home_logs.iloc[0]
        a = away_logs.iloc[0]

        for stat in stat_cols:
            s = stat.lower()
            out.at[idx, "home_" + s] = h.get(stat)
            out.at[idx, "away_" + s] = a.get(stat)

    return out


def main():
    print(f"Loading raw odds data from: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH)

    required_cols = {"season", "date", "home", "away"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    seasons = sorted(int(s) for s in df["season"].unique())
    print("Seasons found in data:", seasons)

    logs_by_season = {}
    for yr in seasons:
        logs_by_season[yr] = load_league_game_logs_for_season(yr)
        time.sleep(0.6)  # be nice to the API

    print("Enriching odds data with nba_api team logs...")
    enriched = enrich_with_logs(df, logs_by_season)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving nba_api sidecar stats to: {OUTPUT_PATH}")
    enriched.to_excel(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()