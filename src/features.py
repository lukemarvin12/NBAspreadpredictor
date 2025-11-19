# src/features.py

import pandas as pd
import numpy as np


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic + advanced, leakage-safe features to the ATS dataset.

    Expects df to have at least:
      - date (string or datetime)
      - home, away
      - spread_home  (signed from home perspective)
      - home_win     (1 if home wins outright, else 0)
      - home_pts, away_pts
      - home_margin  (home_pts - away_pts)

    Optionally (for advanced metrics), if present:
      - home_fga, home_fta, home_tov, home_oreb
      - away_fga, away_fta, away_tov, away_oreb

    Adds:
      - team, opponent, is_home
      - abs_spread
      - team_win_pct_before_game, opp_win_pct_before_game
      - team_recent_win_pct_3, opp_recent_win_pct_3
      - team_rest, opp_rest
      - team_pts_for_5, team_pts_against_5, team_margin_5
      - opp_pts_for_5, opp_pts_against_5, opp_margin_5
      - delta_win_pct, delta_recent_win_pct_3, delta_rest
      - delta_pts_for_5, delta_margin_5
      - (if boxscore cols exist) OffRtg/DefRtg/Pace rolling 5-game + deltas
      - rest/travel flags: B2B, 3-in-4 (team & opp)
      - altitude flags: is_altitude_home
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
    g_team = df.groupby("team")["team_win"]
    team_cum_wins = g_team.cumsum() - df["team_win"]         # wins BEFORE this game
    team_cum_games = g_team.cumcount()                       # number of prior games
    df["team_win_pct_before_game"] = team_cum_wins / team_cum_games.replace(0, pd.NA)

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
    required_basic = {"home_pts", "away_pts", "home_margin"}
    if not required_basic.issubset(df.columns):
        raise ValueError(f"Expected columns missing in df: {required_basic - set(df.columns)}")

    df["team_pts_for"] = df["home_pts"]
    df["team_pts_against"] = df["away_pts"]
    df["team_margin"] = df["home_margin"]

    df["opp_pts_for"] = df["away_pts"]
    df["opp_pts_against"] = df["home_pts"]
    df["opp_margin"] = -df["home_margin"]

    def rolling_mean_5(group_series):
        return group_series.shift().rolling(5).mean()

    df["team_pts_for_5"] = df.groupby("team")["team_pts_for"].transform(rolling_mean_5)
    df["team_pts_against_5"] = df.groupby("team")["team_pts_against"].transform(rolling_mean_5)
    df["team_margin_5"] = df.groupby("team")["team_margin"].transform(rolling_mean_5)

    df["opp_pts_for_5"] = df.groupby("opponent")["opp_pts_for"].transform(rolling_mean_5)
    df["opp_pts_against_5"] = df.groupby("opponent")["opp_pts_against"].transform(rolling_mean_5)
    df["opp_margin_5"] = df.groupby("opponent")["opp_margin"].transform(rolling_mean_5)

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
        else:
            df[col].fillna(league_pf, inplace=True)

    # ------------------------------------------------------------------
    # 7. Delta features (team - opponent) for basic rolling stats
    # ------------------------------------------------------------------
    df["delta_win_pct"] = df["team_win_pct_before_game"] - df["opp_win_pct_before_game"]
    df["delta_recent_win_pct_3"] = df["team_recent_win_pct_3"] - df["opp_recent_win_pct_3"]
    df["delta_rest"] = df["team_rest"] - df["opp_rest"]

    df["delta_pts_for_5"] = df["team_pts_for_5"] - df["opp_pts_for_5"]
    df["delta_margin_5"] = df["team_margin_5"] - df["opp_margin_5"]

    # ------------------------------------------------------------------
    # 8. Advanced metrics from boxscore (possessions, OffRtg, DefRtg, Pace)
    #    Only computed if required columns exist.
    # ------------------------------------------------------------------
    box_cols = {
        "home_fga", "home_fta", "home_tov", "home_oreb",
        "away_fga", "away_fta", "away_tov", "away_oreb",
    }
    has_box = box_cols.issubset(df.columns)

    if has_box:
        # Estimate possessions per Dean Oliver formula
        # pos = FGA + 0.44*FTA - OREB + TOV
        df["home_possessions"] = (
            df["home_fga"]
            + 0.44 * df["home_fta"]
            - df["home_oreb"]
            + df["home_tov"]
        )
        df["away_possessions"] = (
            df["away_fga"]
            + 0.44 * df["away_fta"]
            - df["away_oreb"]
            + df["away_tov"]
        )

        # Avoid divide-by-zero
        df["home_possessions"].replace(0, np.nan, inplace=True)
        df["away_possessions"].replace(0, np.nan, inplace=True)

        # Offensive/Defensive Ratings (per 100 possessions)
        df["home_offrtg"] = 100.0 * df["home_pts"] / df["home_possessions"]
        df["home_defrtg"] = 100.0 * df["away_pts"] / df["home_possessions"]
        df["away_offrtg"] = 100.0 * df["away_pts"] / df["away_possessions"]
        df["away_defrtg"] = 100.0 * df["home_pts"] / df["away_possessions"]

        # Approx game pace: average possessions for the game
        df["game_pace"] = (df["home_possessions"] + df["away_possessions"]) / 2.0

        # Rolling 5-game OffRtg/DefRtg/Pace for team (home) and opponent (away)
        def rolling_mean_5_shift(group_series):
            return group_series.shift().rolling(5).mean()

        df["team_offrtg_5"] = df.groupby("team")["home_offrtg"].transform(rolling_mean_5_shift)
        df["team_defrtg_5"] = df.groupby("team")["home_defrtg"].transform(rolling_mean_5_shift)
        df["team_pace_5"] = df.groupby("team")["game_pace"].transform(rolling_mean_5_shift)

        df["opp_offrtg_5"] = df.groupby("opponent")["away_offrtg"].transform(rolling_mean_5_shift)
        df["opp_defrtg_5"] = df.groupby("opponent")["away_defrtg"].transform(rolling_mean_5_shift)
        df["opp_pace_5"] = df.groupby("opponent")["game_pace"].transform(rolling_mean_5_shift)

        # Fill NaNs with league averages
        avg_off = pd.concat([df["home_offrtg"], df["away_offrtg"]]).mean()
        avg_def = pd.concat([df["home_defrtg"], df["away_defrtg"]]).mean()
        avg_pace = df["game_pace"].mean()

        for col in [
            "team_offrtg_5", "team_defrtg_5", "team_pace_5",
            "opp_offrtg_5", "opp_defrtg_5", "opp_pace_5",
        ]:
            if "offrtg" in col:
                df[col].fillna(avg_off, inplace=True)
            elif "defrtg" in col:
                df[col].fillna(avg_def, inplace=True)
            else:  # pace
                df[col].fillna(avg_pace, inplace=True)

        # Delta advanced metrics
        df["delta_offrtg_5"] = df["team_offrtg_5"] - df["opp_offrtg_5"]
        df["delta_defrtg_5"] = df["team_defrtg_5"] - df["opp_defrtg_5"]
        df["delta_pace_5"] = df["team_pace_5"] - df["opp_pace_5"]

    else:
        # If no boxscore detail, we just skip these; training code will only use cols that exist
        pass

    # ------------------------------------------------------------------
    # 9. Rest / travel flags: B2B, 3-in-4
    # ------------------------------------------------------------------
    # B2B: rest == 1
    df["team_is_b2b"] = (df["team_rest"] == 1).astype(int)
    df["opp_is_b2b"] = (df["opp_rest"] == 1).astype(int)

    # 3-in-4: 3rd game in <=4 days (approx)
    def three_in_four(dates: pd.Series) -> pd.Series:
        # dates is already sorted per team
        return (dates - dates.shift(2)).dt.days <= 4

    team_dates = df.groupby("team")["date"].transform(lambda s: s)
    opp_dates = df.groupby("opponent")["date"].transform(lambda s: s)

    df["team_is_3in4"] = df.groupby("team")["date"].transform(three_in_four).fillna(False).astype(int)
    df["opp_is_3in4"] = df.groupby("opponent")["date"].transform(three_in_four).fillna(False).astype(int)

    # ------------------------------------------------------------------
    # 10. Altitude flag (DEN, UTA as examples)
    # ------------------------------------------------------------------
    # Your 'home'/'away' codes are lower-case like 'den', 'uta'
    ALTITUDE_TEAMS = {"den", "uta"}
    df["is_altitude_home"] = df["home"].astype(str).str.lower().isin(ALTITUDE_TEAMS).astype(int)
    df["is_altitude_away"] = df["away"].astype(str).str.lower().isin(ALTITUDE_TEAMS).astype(int)

    # Convert date back to string for consistency with rest of pipeline
    df["date"] = df["date"].dt.date.astype(str)

    return df