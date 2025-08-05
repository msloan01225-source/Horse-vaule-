import pandas as pd

def calculate_scores(df):
    """
    Adds Win %, Place %, and Value Score columns.
    """
    # Win % from model probability
    df["Win %"] = (df["Model Prob"] * 100).round(1)

    # Place % approximation (top 3 finish for most races)
    df["Place %"] = (df["Win %"] * 1.8).clip(upper=100).round(1)

    # Value Score = Model Prob - Market Prob
    df["Value Score"] = (df["Model Prob"] - df["Market Prob"]).round(3)

    return df
