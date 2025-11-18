"""
enrich_with_nba_api.py

Reads your slim odds file (nba_2018-2025.xlsx), pulls team-level game stats
from the official NBA stats API (via nba_api), merges them in, and writes
an enriched Excel file with home/away stats per game.

Output: data/processed/nba_2018-2025_with_stats.xlsx
"""

import os
import time
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.library.parameters import SeasonTypeAllStar


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

INPUT_PATH  = "data/raw/nba_2018-2025.xlsx"   # adjust if your file is elsewhere
OUTPUT_DIR  = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nba_2018-2025_with_stats.xlsx")

# Map your short codes -> official NBA team abbreviations used by nba_api
TEAM_MAP = {
        # Eastern Conference
    "atl": "ATL",  # Atlanta Hawks
    "bos": "BOS",  # Boston Celtics
    "bkn": "BKN",  # Brooklyn Nets
    "nj":  "BKN",  # legacy New Jersey -> treat as Nets
    "njn": "BKN",
    "cha": "CHA",  # Charlotte Hornets (Bobcats/Hornets collapsed)
    "chi": "CHI",  # Chicago Bulls
    "cle": "CLE",  # Cleveland Cavaliers
    "det": "DET",  # Detroit Pistons
    "ind": "IND",  # Indiana Pacers
    "mia": "MIA",  # Miami Heat
    "mil": "MIL",  # Milwaukee Bucks
    "ny":  "NYK",  # New York Knicks
    "nyk": "NYK",
    "orl": "ORL",  # Orlando Magic
    "phi": "PHI",  # Philadelphia 76ers
    "tor": "TOR",  # Toronto Raptors
    "wsh": "WAS",  # Washington Wizards (dataset tag)
    "was": "WAS",

    # Western Conference
    "dal": "DAL",  # Dallas Mavericks
    "den": "DEN",  # Denver Nuggets
    "gs":  "GSW",  # Golden State Warriors (short tag in your sheet)
    "gsw": "GSW",
    "hou": "HOU",  # Houston Rockets
    "lac": "LAC",  # LA Clippers
    "lal": "LAL",  # LA Lakers
    "mem": "MEM",  # Memphis Grizzlies
    "min": "MIN",  # Minnesota Timberwolves
    "nop": "NOP",  # New Orleans Pelicans
    "no":  "NOP",  # legacy New Orleans -> treat as Pels
    "noh": "NOP",
    "nok": "NOP",
    "okc": "OKC",  # Oklahoma City Thunder
    "sea": "OKC",  # legacy Sonics -> map into OKC for current-league view
    "phx": "PHX",  # Phoenix Suns
    "por": "POR",  # Portland Trail Blazers
    "sa":  "SAS",  # San Antonio Spurs
    "sas": "SAS",
    "sac": "SAC",  # Sacramento Kings
    "uta": "UTA",  # Utah Jazz
    "utah": "UTA",  # Utah Jazz
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def season_label_to_str(season_label: int) -> str:
    """
    Convert numeric season label (e.g., 2008) to NBA API format '2007-08'.
    2008 means the 2007-2008 season, etc.
    """
    start_year = season_label - 1
    end_year_short = str(season_label)[-2:]
    return f"{start_year}-{end_year_short}"


def fetch_league_gamelog_for_season(season_label: int) -> pd.DataFrame:
    """
    Fetch BOTH regular season and playoff game logs for a given season label
    using nba_api's LeagueGameLog endpoint.
    """
    season_str = season_label_to_str(season_label)
    print(f"  Fetching NBA gamelog for season {season_label} ({season_str})")

    frames = []

    for stype, stype_name in [
        ("Regular Season", SeasonTypeAllStar.regular),
        ("Playoffs", SeasonTypeAllStar.playoffs),
    ]:
        print(f"    - {stype} ...", end="", flush=True)
        gl = leaguegamelog.LeagueGameLog(
            season=season_str,
            season_type_all_star=stype_name
        )
        df = gl.get_data_frames()[0]
        df["SEASON_LABEL"] = season_label
        df["SEASON_TYPE"] = stype
        frames.append(df)
        print(f" got {len(df)} rows")
        time.sleep(0.6)  # avoid hammering the API

    df_all = pd.concat(frames, ignore_index=True)

    # Normalize date
    df_all["GAME_DATE"] = pd.to_datetime(df_all["GAME_DATE"]).dt.date.astype(str)

    return df_all


def build_home_away_games_df(gamelog_all: pd.DataFrame) -> pd.DataFrame:
    """
    Transform leaguegamelog rows (one row per team per game) into
    one row per game, with home_*/away_* stats.
    """

    # HOME: rows with 'vs.' in the MATCHUP string
    home = gamelog_all[gamelog_all["MATCHUP"].str.contains(" vs. ", na=False)].copy()
    # AWAY: rows with '@' in the MATCHUP string
    away = gamelog_all[gamelog_all["MATCHUP"].str.contains(" @ ", na=False)].copy()

    # Select and rename stat columns for home/away
    stat_cols = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON_LABEL",
        "SEASON_TYPE",
        "TEAM_ABBREVIATION",
        "PTS",
        "REB",
        "AST",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "TOV",
        "STL",
        "BLK",
        "PLUS_MINUS",
    ]

    home = home[stat_cols].rename(columns={
        "TEAM_ABBREVIATION": "home_team_abbr",
        "PTS": "home_pts",
        "REB": "home_reb",
        "AST": "home_ast",
        "FGM": "home_fgm",
        "FGA": "home_fga",
        "FG3M": "home_fg3m",
        "FG3A": "home_fg3a",
        "FTM": "home_ftm",
        "FTA": "home_fta",
        "TOV": "home_tov",
        "STL": "home_stl",
        "BLK": "home_blk",
        "PLUS_MINUS": "home_plus_minus",
    })

    away = away[stat_cols].rename(columns={
        "TEAM_ABBREVIATION": "away_team_abbr",
        "PTS": "away_pts",
        "REB": "away_reb",
        "AST": "away_ast",
        "FGM": "away_fgm",
        "FGA": "away_fga",
        "FG3M": "away_fg3m",
        "FG3A": "away_fg3a",
        "FTM": "away_ftm",
        "FTA": "away_fta",
        "TOV": "away_tov",
        "STL": "away_stl",
        "BLK": "away_blk",
        "PLUS_MINUS": "away_plus_minus",
    })

    games = pd.merge(
        home,
        away,
        on=["GAME_ID", "GAME_DATE", "SEASON_LABEL", "SEASON_TYPE"],
        how="inner",
    )

    # For convenience, also expose final score margin, etc.
    games["home_margin"] = games["home_pts"] - games["away_pts"]

    return games


def map_team_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home_abbr and away_abbr columns using TEAM_MAP.
    """
    df = df.copy()
    df["home_code_clean"] = df["home"].astype(str).str.lower().str.strip()
    df["away_code_clean"] = df["away"].astype(str).str.lower().str.strip()

    df["home_abbr"] = df["home_code_clean"].map(TEAM_MAP)
    df["away_abbr"] = df["away_code_clean"].map(TEAM_MAP)

    missing_home = df["home_abbr"].isna().sum()
    missing_away = df["away_abbr"].isna().sum()
    if missing_home or missing_away:
        print(f"WARNING: {missing_home} home team codes and {missing_away} away team codes "
              f"could not be mapped. Check TEAM_MAP for missing entries.")

    return df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("Loading odds file:", INPUT_PATH)
    odds_df = pd.read_excel(INPUT_PATH)

    # Normalize date
    odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.date.astype(str)

    # Map team codes
    odds_df = map_team_codes(odds_df)

    # Fetch gamelogs for all seasons present in odds_df
    seasons = sorted(odds_df["season"].dropna().unique())
    print("Seasons found in odds file:", seasons)

    gamelog_frames = []
    for s in seasons:
        try:
            gl = fetch_league_gamelog_for_season(int(s))
            gamelog_frames.append(gl)
        except Exception as e:
            print(f"  ERROR fetching gamelog for season {s}: {e}")

    if not gamelog_frames:
        raise RuntimeError("No gamelog data fetched. Aborting.")

    gamelog_all = pd.concat(gamelog_frames, ignore_index=True)

    # Build home/away game-level stats
    print("Building home/away game stats from gamelog...")
    games_stats = build_home_away_games_df(gamelog_all)

    # Merge odds with stats
    # odds_df: season, date, home_abbr, away_abbr, regular/playoffs, spread, whos_favored...
    # games_stats: SEASON_LABEL, GAME_DATE, home_team_abbr, away_team_abbr, stats...

    print("Merging odds with NBA stats...")
    merged = odds_df.merge(
        games_stats,
        left_on=["season", "date", "home_abbr", "away_abbr"],
        right_on=["SEASON_LABEL", "GAME_DATE", "home_team_abbr", "away_team_abbr"],
        how="left",
    )

    print("Rows in merged dataframe:", len(merged))
    unmatched = merged["GAME_ID"].isna().sum()
    if unmatched:
        print(f"WARNING: {unmatched} rows have no matching GAME_ID/stats. Likely team code or date mismatches.")

    # Create output directory and save to Excel
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Saving enriched dataset to:", OUTPUT_PATH)
    merged.to_excel(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()