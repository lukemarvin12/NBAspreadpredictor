import pandas as pd
import numpy as np


# ------------------------------------------------------------
# Helper: normalize score columns
# ------------------------------------------------------------
def _normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the frame has home_pts and away_pts columns, regardless of how
    they are named after merges (score_home, home_pts_x, home_pts_y, etc.).
    """
    df = df.copy()

    # Home points
    home_candidates = ["home_pts", "score_home", "home_pts_x", "home_pts_y"]
    for c in home_candidates:
        if c in df.columns:
            df["home_pts"] = df[c]
            break

    # Away points
    away_candidates = ["away_pts", "score_away", "away_pts_x", "away_pts_y"]
    for c in away_candidates:
        if c in df.columns:
            df["away_pts"] = df[c]
            break

    if "home_pts" not in df.columns or "away_pts" not in df.columns:
        raise ValueError(
            "Could not find usable home/away score columns. "
            f"Available columns: {list(df.columns)}"
        )

    return df


# ------------------------------------------------------------
# Main feature function
# ------------------------------------------------------------
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team + opponent rolling features and matchup deltas.

    Expects df (from build_dataset.py) to have at least:
      - date
      - home, away
      - spread (home-signed)
      - some version of scores (home_pts_x / score_home, etc.)

    Returns the same rows with many new feature columns added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Normalize scores into home_pts / away_pts
    df = _normalize_scores(df)

    # ---------------------------------------------------------------------
    # 1. Build a long-format frame with one row per team per game
    # ---------------------------------------------------------------------
    # Keep a stable index to join features back later
    df = df.reset_index().rename(columns={"index": "game_idx"})

    # Home perspective row
    home_df = pd.DataFrame({
        "game_idx": df["game_idx"],
        "date": df["date"],
        "team": df["home"],
        "opponent": df["away"],
        "pts": df["home_pts"],
        "opp_pts": df["away_pts"],
        "is_home": 1,
    })

    # Away perspective row
    away_df = pd.DataFrame({
        "game_idx": df["game_idx"],
        "date": df["date"],
        "team": df["away"],
        "opponent": df["home"],
        "pts": df["away_pts"],
        "opp_pts": df["home_pts"],
        "is_home": 0,
    })

    long_df = pd.concat([home_df, away_df], ignore_index=True)

    # Sort for time series operations
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 2. Basic outcomes: win flags
    # ---------------------------------------------------------------------
    long_df["win"] = (long_df["pts"] > long_df["opp_pts"]).astype(int)

    # ---------------------------------------------------------------------
    # 3. Win% before game (no leakage)
    #    Use cumulative wins & games BEFORE the current game.
    # ---------------------------------------------------------------------
    g = long_df.groupby("team")

    long_df["games_before"] = g.cumcount()
    long_df["cum_wins"] = g["win"].cumsum() - long_df["win"]
    long_df["win_pct_before_game"] = (
        long_df["cum_wins"] / long_df["games_before"].replace(0, np.nan)
    )

    # Neutral prior for early season
    long_df["win_pct_before_game"].fillna(0.5, inplace=True)

    # ---------------------------------------------------------------------
    # 4. Recent form: last 3 games win%
    #    Use groupby.transform with rolling to avoid MultiIndex issues.
    # ---------------------------------------------------------------------
    long_df["recent_win_pct_3"] = g["win"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    long_df["recent_win_pct_3"].fillna(0.5, inplace=True)

    # ---------------------------------------------------------------------
    # 5. Rest days, B2B, 3-in-4
    # ---------------------------------------------------------------------
    long_df["prev_date"] = g["date"].shift(1)
    long_df["rest"] = (long_df["date"] - long_df["prev_date"]).dt.days
    long_df["rest"] = long_df["rest"].clip(lower=0, upper=10)
    long_df["rest"].fillna(3, inplace=True)  # neutral rest for first game

    long_df["is_b2b"] = (long_df["rest"] == 1).astype(int)
    long_df["is_3in4"] = (long_df["rest"] <= 2).astype(int)

    # ---------------------------------------------------------------------
    # 6. Rolling box-score stats (5- & 10-game)
    # ---------------------------------------------------------------------
    long_df["margin"] = long_df["pts"] - long_df["opp_pts"]

    for window in [5, 10]:
        long_df[f"pf_{window}"] = g["pts"].transform(
            lambda s: s.shift(1).rolling(window).mean()
        )
        long_df[f"pa_{window}"] = g["opp_pts"].transform(
            lambda s: s.shift(1).rolling(window).mean()
        )
        long_df[f"margin_{window}"] = g["margin"].transform(
            lambda s: s.shift(1).rolling(window).mean()
        )

    # ---------------------------------------------------------------------
    # 7. Split back into home vs opponent features
    # ---------------------------------------------------------------------
    home_rows = long_df[long_df["is_home"] == 1].set_index("game_idx")
    opp_rows = long_df[long_df["is_home"] == 0].set_index("game_idx")

    base = df.set_index("game_idx")

    # Team (home) features
    team_feats = home_rows[[
        "win_pct_before_game",
        "recent_win_pct_3",
        "rest",
        "is_b2b",
        "is_3in4",
        "pf_5", "pa_5", "margin_5",
        "pf_10", "pa_10", "margin_10",
    ]].rename(columns={
        "win_pct_before_game": "team_win_pct_before_game",
        "recent_win_pct_3": "team_recent_win_pct_3",
        "rest": "team_rest",
        "is_b2b": "team_is_b2b",
        "is_3in4": "team_is_3in4",
        "pf_5": "team_pf_5",
        "pa_5": "team_pa_5",
        "margin_5": "team_margin_5",
        "pf_10": "team_pf_10",
        "pa_10": "team_pa_10",
        "margin_10": "team_margin_10",
    })

    # Opponent (away) features
    opp_feats = opp_rows[[
        "win_pct_before_game",
        "recent_win_pct_3",
        "rest",
        "is_b2b",
        "is_3in4",
        "pf_5", "pa_5", "margin_5",
        "pf_10", "pa_10", "margin_10",
    ]].rename(columns={
        "win_pct_before_game": "opp_win_pct_before_game",
        "recent_win_pct_3": "opp_recent_win_pct_3",
        "rest": "opp_rest",
        "is_b2b": "opp_is_b2b",
        "is_3in4": "opp_is_3in4",
        "pf_5": "opp_pf_5",
        "pa_5": "opp_pa_5",
        "margin_5": "opp_margin_5",
        "pf_10": "opp_pf_10",
        "pa_10": "opp_pa_10",
        "margin_10": "opp_margin_10",
    })

    # Join onto base game-level frame
    base = base.join(team_feats, how="left")
    base = base.join(opp_feats, how="left")

    # ---------------------------------------------------------------------
    # 8. Delta features (home − away)
    # ---------------------------------------------------------------------
    base["delta_win_pct"] = (
        base["team_win_pct_before_game"] - base["opp_win_pct_before_game"]
    )
    base["delta_recent_win_pct_3"] = (
        base["team_recent_win_pct_3"] - base["opp_recent_win_pct_3"]
    )
    base["delta_rest"] = base["team_rest"] - base["opp_rest"]
    base["delta_margin_5"] = base["team_margin_5"] - base["opp_margin_5"]
    base["delta_pf_5"] = base["team_pf_5"] - base["opp_pf_5"]

    # ---------------------------------------------------------------------
    # 9. Altitude flags
    # ---------------------------------------------------------------------
    altitude_teams = {"DEN", "UTA"}

    # Prefer abbr columns if present
    if "home_abbr" in base.columns and "away_abbr" in base.columns:
        home_code = base["home_abbr"].astype(str).str.upper()
        away_code = base["away_abbr"].astype(str).str.upper()
    else:
        home_code = base["home"].astype(str).str.upper()
        away_code = base["away"].astype(str).str.upper()

    base["is_altitude_home"] = home_code.isin(altitude_teams).astype(int)
    base["is_altitude_away"] = away_code.isin(altitude_teams).astype(int)

    # ---------------------------------------------------------------------
    # 10. Spread features
    # ---------------------------------------------------------------------
    if "spread" in base.columns:
        base["abs_spread"] = base["spread"].abs()

    # ---------------------------------------------------------------------
    # 11. Fill some NaNs with neutral priors
    # ---------------------------------------------------------------------
    base["team_win_pct_before_game"].fillna(0.5, inplace=True)
    base["opp_win_pct_before_game"].fillna(0.5, inplace=True)
    base["team_recent_win_pct_3"].fillna(0.5, inplace=True)
    base["opp_recent_win_pct_3"].fillna(0.5, inplace=True)

    # Reset index to normal
    base = base.reset_index(drop=True)

    return base