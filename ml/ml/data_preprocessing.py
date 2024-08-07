"""
This module includes functions for preprocessing data.
"""
import pandas as pd

from ml.ml.logger import logger


def filter_clubs_by_min_matches(
        df: pd.DataFrame,
        id_col: str,
        home_col: str,
        away_col: str,
        threshold: int
) -> pd.DataFrame:
    """
    Filters the DataFrame to include only clubs with a minimum number of played
        matches.

    Args:
        df (pd.DataFrame): The input DataFrame containing match data.
        id_col (str): The column name for the game id.
        home_col (str): The column name for the home club.
        away_col (str): The column name for the away club.
        threshold (int): The minimum number of matches a club must have played
            to be included.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    home_match_counts = (
        df[[home_col, id_col]]
        .groupby(home_col)
        .count()
        .rename({id_col: "home_match_count"}, axis=1)
    )

    away_match_counts = (
        df[[away_col, id_col]]
        .groupby(away_col)
        .count()
        .rename({id_col: "away_match_count"}, axis=1)
    )

    df = (
        df
        .merge(home_match_counts, on=home_col, how="left")
        .merge(away_match_counts, on=away_col, how="left")
    )

    filtered_df = df.loc[
        (df["home_match_count"] >= threshold) &
        (df["away_match_count"] >= threshold)
    ]

    filtered_df = filtered_df.drop(
        columns=["home_match_count", "away_match_count"]
    )

    logger.info(
        "The DataFrame filtered to include only clubs with a minimum number "
        "of matches: threshold = %d, original size = %d, filtered size = %d",
        threshold,
        df.shape[0],
        filtered_df.shape[0]
    )

    return filtered_df.reset_index(drop=True)

