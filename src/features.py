# src/features.py

import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic, leakage-safe features to the ATS dataset.

    Expects df to have at least:
      - date (string or datetime)
      - home, away
      - spread_home  (signed from home perspective)
      - home_win     (1 if home wins outright, else 0)
      - home_pts, away_pts
      - home_margin  (home_pts - away_pts)

    Adds:
      - team, opponent, is_home
      - abs_spread
      - team_win_pct_before_game, opp_win_pct_before_game
      - team_recent_win_pct_3, opp_recent_win_pct_3
      - team_rest, opp_rest
      - box-score rolling stats (5-game) for pts & margin (team & opp)
      - delta features (team - opp) for win%, recent win%, rest, pts, margin
    """

    df = df.copy()

    # Ensure date is datetime for sorting and rest calculations
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 0. Define team / opponent from home perspective
    # ------------------------------------------------------------------
    df["team"] = df["home"]
    df["opponent"] = df["away"]
    df["is_home"] = 1  # always 1 in this dataset, but explicit

    # ------------------------------------------------------------------
    # 1. Basic spread features
    # ------------------------------------------------------------------
    if "spread_home" not in df.columns:
        raise ValueError("Expected 'spread_home' column in df.")
    df["abs_spread"] = df["spread_home"].abs()

    # ------------------------------------------------------------------
    # 2. Win / loss flags from home perspective
    # ------------------------------------------------------------------
    if "home_win" not in df.columns:
        raise ValueError("Expected 'home_win' column in df.")

    df["team_win"] = df["home_win"]         # team = home
    df["opp_win"] = 1 - df["home_win"]      # opponent = away

    # ------------------------------------------------------------------
    # 3. Season-to-date win% BEFORE the game
    # ------------------------------------------------------------------
    # For team (home)
    g_team = df.groupby("team")["team_win"]
    team_cum_wins = g_team.cumsum() - df["team_win"]         # wins BEFORE this game
    team_cum_games = g_team.cumcount()                       # number of prior games
    df["team_win_pct_before_game"] = team_cum_wins / team_cum_games.replace(0, pd.NA)

    # For opponent (away)
    g_opp = df.groupby("opponent")["opp_win"]
    opp_cum_wins = g_opp.cumsum() - df["opp_win"]
    opp_cum_games = g_opp.cumcount()
    df["opp_win_pct_before_game"] = opp_cum_wins / opp_cum_games.replace(0, pd.NA)

    # Early season games: neutral prior 0.5
    df["team_win_pct_before_game"].fillna(0.5, inplace=True)
    df["opp_win_pct_before_game"].fillna(0.5, inplace=True)

    # ------------------------------------------------------------------
    # 4. Recent form (last 3 games, excluding this one)
    # ------------------------------------------------------------------
    df["team_recent_win_pct_3"] = (
        df.groupby("team")["team_win"]
        .transform(lambda s: s.shift().rolling(3).mean())
    )
    df["opp_recent_win_pct_3"] = (
        df.groupby("opponent")["opp_win"]
        .transform(lambda s: s.shift().rolling(3).mean())
    )

    df["team_recent_win_pct_3"].fillna(0.5, inplace=True)
    df["opp_recent_win_pct_3"].fillna(0.5, inplace=True)

    # ------------------------------------------------------------------
    # 5. Rest days (for home team & opponent)
    # ------------------------------------------------------------------
    df["team_rest"] = (
        df.groupby("team")["date"]
        .diff()
        .dt.days
        .clip(upper=10)
    )
    df["opp_rest"] = (
        df.groupby("opponent")["date"]
        .diff()
        .dt.days
        .clip(upper=10)
    )

    # Assume 1 day rest for first game we see for each team
    df["team_rest"].fillna(1, inplace=True)
    df["opp_rest"].fillna(1, inplace=True)

    # ------------------------------------------------------------------
    # 6. Box-score based rolling stats (5-game) for points & margin
    # ------------------------------------------------------------------
    if not {"home_pts", "away_pts", "home_margin"}.issubset(df.columns):
        raise ValueError("Expected 'home_pts', 'away_pts', 'home_margin' columns in df.")

    # From the home team's perspective
    df["team_pts_for"] = df["home_pts"]
    df["team_pts_against"] = df["away_pts"]
    df["team_margin"] = df["home_margin"]

    # From the opponent's perspective (they are the away team here)
    df["opp_pts_for"] = df["away_pts"]
    df["opp_pts_against"] = df["home_pts"]
    df["opp_margin"] = -df["home_margin"]

    # Rolling 5-game averages BEFORE this game (shift by 1)
    def rolling_mean_5(group_series):
        return group_series.shift().rolling(5).mean()

    # Team 5-game stats
    df["team_pts_for_5"] = df.groupby("team")["team_pts_for"].transform(rolling_mean_5)
    df["team_pts_against_5"] = df.groupby("team")["team_pts_against"].transform(rolling_mean_5)
    df["team_margin_5"] = df.groupby("team")["team_margin"].transform(rolling_mean_5)

    # Opponent 5-game stats
    df["opp_pts_for_5"] = df.groupby("opponent")["opp_pts_for"].transform(rolling_mean_5)
    df["opp_pts_against_5"] = df.groupby("opponent")["opp_pts_against"].transform(rolling_mean_5)
    df["opp_margin_5"] = df.groupby("opponent")["opp_margin"].transform(rolling_mean_5)

    # Fill early-season NaNs with league averages
    league_pf = df["home_pts"].mean()
    league_pa = df["away_pts"].mean()
    league_margin = df["home_margin"].mean()

    for col in [
        "team_pts_for_5",
        "team_pts_against_5",
        "team_margin_5",
        "opp_pts_for_5",
        "opp_pts_against_5",
        "opp_margin_5",
    ]:
        if "margin" in col:
            df[col].fillna(league_margin, inplace=True)
        elif "against" in col:
            df[col].fillna(league_pa, inplace=True)
        else:  # pts_for
            df[col].fillna(league_pf, inplace=True)

    # ------------------------------------------------------------------
    # 7. Delta features (team - opponent)
    # ------------------------------------------------------------------
    df["delta_win_pct"] = df["team_win_pct_before_game"] - df["opp_win_pct_before_game"]
    df["delta_recent_win_pct_3"] = df["team_recent_win_pct_3"] - df["opp_recent_win_pct_3"]
    df["delta_rest"] = df["team_rest"] - df["opp_rest"]

    df["delta_pts_for_5"] = df["team_pts_for_5"] - df["opp_pts_for_5"]
    df["delta_margin_5"] = df["team_margin_5"] - df["opp_margin_5"]

    # Convert date back to string for consistency with rest of pipeline
    df["date"] = df["date"].dt.date.astype(str)

    return df