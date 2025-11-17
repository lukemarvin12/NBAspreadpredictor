# src/features.py

import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic, game-level features for modeling.

    Assumes df has (at minimum):
      - date           (datetime)
      - home, away     (team identifiers)
      - score_home, score_away
      - spread         (closing spread)
      - optionally: whos_favored, moneyline_* etc.

    High-level steps:
      1. Sort games chronologically and add a game_id.
      2. Build a "long" team-level table (one row per TEAM per GAME).
      3. On that long table, compute:
           - team_win (1/0)
           - team_win_pct_before_game
           - team_recent_win_pct (last 3 games)
           - team_rest (days since last game)
      4. Split the long table back into home vs away rows and
         merge the features onto the original game-level df.
      5. Add some simple game-level features like abs_spread, home_win, etc.

    The key idea: we want each team’s stats to reflect ALL prior games
    (home + away), not just their home games.
    """

    # ------------------------------------------------------------------
    # 0. Basic setup: sort by date, ensure we don't mutate the original df
    # ------------------------------------------------------------------
    df = df.sort_values("date").reset_index(drop=True).copy()

    # Unique ID per game (used to merge features back later)
    df["game_id"] = df.index

    # ------------------------------------------------------------------
    # 1. Simple game-level flags from the HOME perspective
    # ------------------------------------------------------------------

    # Did the home team win?
    df["home_win"] = (df["score_home"] > df["score_away"]).astype(int)
    # Away win is just the opposite
    df["away_win"] = 1 - df["home_win"]

    # Absolute value of the spread (how big the line is)
    df["abs_spread"] = df["spread"].abs()

    # Who is favored? If we have a 'whos_favored' column, use it.
    # Otherwise you could infer from spread sign if you KNOW the convention.
    if "whos_favored" in df.columns:
        df["is_home_favorite"] = (df["whos_favored"].str.lower() == "home").astype(int)
    else:
        # Fallback heuristic (only use if you're sure about sign convention)
        df["is_home_favorite"] = (df["spread"] < 0).astype(int)

    # ------------------------------------------------------------------
    # 2. Build a long-format team-level table (one row per team per game)
    #    This is the crucial step to make team stats use ALL games.
    # ------------------------------------------------------------------

    # Home side: team = home, opponent = away
    home = df[["game_id", "date", "home", "away", "score_home", "score_away"]].copy()
    home.rename(
        columns={
            "home": "team",
            "away": "opponent",
            "score_home": "team_score",
            "score_away": "opp_score",
        },
        inplace=True,
    )
    home["is_home"] = 1
    home["side"] = "home"

    # Away side: team = away, opponent = home
    away = df[["game_id", "date", "home", "away", "score_home", "score_away"]].copy()
    away.rename(
        columns={
            "away": "team",
            "home": "opponent",
            "score_away": "team_score",
            "score_home": "opp_score",
        },
        inplace=True,
    )
    away["is_home"] = 0
    away["side"] = "away"

    # Stack them together → now we have one row for each TEAM in each GAME
    long = pd.concat([home, away], ignore_index=True)

    # Sort within each team by date (and game_id as tie-breaker)
    long = long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Compute team-level stats on the long table
    # ------------------------------------------------------------------

    # 3.1. Win flag from this team's perspective
    long["team_win"] = (long["team_score"] > long["opp_score"]).astype(int)

    # Group by team for cumulative stats
    g_team = long.groupby("team")["team_win"]

    # 3.2. Season-to-date win% BEFORE this game:
    #      - cumulative wins up to previous game / number of previous games
    team_cum_wins = g_team.cumsum() - long["team_win"]          # wins before this game
    team_cum_games = g_team.cumcount()                          # number of prior games

    long["team_win_pct_before_game"] = team_cum_wins / team_cum_games.replace(0, pd.NA)
    # Neutral prior (0.5) for first games / early NaNs
    long["team_win_pct_before_game"].fillna(0.5, inplace=True)

    # 3.3. Recent form: win% over last 3 games (excluding this one)
    long["team_recent_win_pct"] = (
        long.groupby("team")["team_win"]
            .transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    )
    # Again, fill very early games with 0.5 as a neutral baseline
    long["team_recent_win_pct"].fillna(0.5, inplace=True)

    # 3.4. Rest days for this team: days since last time THIS team played
    long["team_rest"] = (
        long.groupby("team")["date"]
            .diff()
            .dt.days
            .clip(upper=10)  # cap super-long breaks at 10 days
    )
    # Assume 1 day of rest if it's the first appearance we have for that team
    long["team_rest"].fillna(1, inplace=True)

    # ------------------------------------------------------------------
    # 4. Split long back into home vs away and merge onto the original df
    # ------------------------------------------------------------------

    # Features for the home team in each game
    home_feats = (
        long[long["side"] == "home"]
        .loc[:, ["game_id", "team_win_pct_before_game", "team_recent_win_pct", "team_rest"]]
        .rename(
            columns={
                "team_win_pct_before_game": "home_win_pct_before_game",
                "team_recent_win_pct": "home_recent_win_pct",
                "team_rest": "home_rest",
            }
        )
    )

    # Features for the away team in each game
    away_feats = (
        long[long["side"] == "away"]
        .loc[:, ["game_id", "team_win_pct_before_game", "team_recent_win_pct", "team_rest"]]
        .rename(
            columns={
                "team_win_pct_before_game": "away_win_pct_before_game",
                "team_recent_win_pct": "away_recent_win_pct",
                "team_rest": "away_rest",
            }
        )
    )

    # Merge them onto the game-level df by game_id
    df = df.merge(home_feats, on="game_id", how="left")
    df = df.merge(away_feats, on="game_id", how="left")

    # ------------------------------------------------------------------
    # 5. Convenience "difference" features (home minus away)
    # ------------------------------------------------------------------
    df["win_pct_diff_before_game"] = (
        df["home_win_pct_before_game"] - df["away_win_pct_before_game"]
    )
    df["recent_win_pct_diff"] = (
        df["home_recent_win_pct"] - df["away_recent_win_pct"]
    )
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    # You now have a game-level dataset with:
    #   - home_* features for the home team
    #   - away_* features for the away team
    #   - *_diff features capturing relative advantage

    return df