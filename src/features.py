# src/features.py

import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple, home-team–centric features to the dataset.

    Assumes df has (at minimum):
      - date
      - home, away
      - score_home, score_away
      - spread  (from the HOME team's perspective: + = home dog, - = home favorite)
    and anything else from build_dataset.py.
    """

    # Make sure we're in chronological order and reset index for clean groupby ops
    df = df.sort_values("date").reset_index(drop=True).copy()

    # -----------------------------------
    # 0. Define team / opponent columns
    #    (from HOME perspective)
    # -----------------------------------
    # "team" = home team, "opponent" = away team
    if "team" not in df.columns:
        df["team"] = df["home"]

    if "opponent" not in df.columns:
        df["opponent"] = df["away"]

    # From this dataset's perspective, every row is a home game
    df["is_home"] = 1

    # -----------------------------------
    # 1. Win / loss flags (home perspective)
    # -----------------------------------
    if "team_win" not in df.columns or "opp_win" not in df.columns:
        home_win = (df["score_home"] > df["score_away"]).astype(int)
        df["team_win"] = home_win            # team = home
        df["opp_win"] = 1 - home_win         # opponent = away

    # -----------------------------------
    # 2. Spread features (home-signed)
    # -----------------------------------
    # spread already assumed to be from HOME perspective:
    #   spread < 0 -> home favored
    #   spread > 0 -> home underdog
    df["abs_spread"] = df["spread"].abs()

    # -----------------------------------
    # 3. Season-to-date win% BEFORE the game
    #    (no MultiIndex tricks, just cumsum / cumcount)
    # -----------------------------------
    # For the home team ("team")
    g_team = df.groupby("team")["team_win"]
    team_cum_wins = g_team.cumsum() - df["team_win"]          # wins BEFORE this game
    team_cum_games = g_team.cumcount()                        # number of prior games
    df["team_win_pct_before_game"] = team_cum_wins / team_cum_games.replace(0, pd.NA)

    # For the opponent ("opponent")
    g_opp = df.groupby("opponent")["opp_win"]
    opp_cum_wins = g_opp.cumsum() - df["opp_win"]
    opp_cum_games = g_opp.cumcount()
    df["opp_win_pct_before_game"] = opp_cum_wins / opp_cum_games.replace(0, pd.NA)

    # Fill early-season NaNs with 0.5 as a neutral prior
    df["team_win_pct_before_game"].fillna(0.5, inplace=True)
    df["opp_win_pct_before_game"].fillna(0.5, inplace=True)

    # -----------------------------------
    # 4. Recent form (last 3 games)
    # -----------------------------------
    df["team_recent_win_pct"] = (
        df.groupby("team")["team_win"]
        .transform(lambda s: s.shift().rolling(3).mean())
    )

    df["opp_recent_win_pct"] = (
        df.groupby("opponent")["opp_win"]
        .transform(lambda s: s.shift().rolling(3).mean())
    )

    df["team_recent_win_pct"].fillna(0.5, inplace=True)
    df["opp_recent_win_pct"].fillna(0.5, inplace=True)

    # -----------------------------------
    # 5. Rest days (for home team & opponent)
    # -----------------------------------
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

    # Assume 1 day rest if it's the first game we see for that team
    df["team_rest"].fillna(1, inplace=True)
    df["opp_rest"].fillna(1, inplace=True)

    return df