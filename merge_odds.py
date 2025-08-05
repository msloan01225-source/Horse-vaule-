import pandas as pd
from racingpost_scraper import get_all_todays_data as get_rp_data
from timeform_scraper import get_all_todays_data as get_tf_data

def merge_best_prices():
    rp_df = get_rp_data()
    tf_df = get_tf_data()
    if rp_df.empty and tf_df.empty:
        return pd.DataFrame()
    rp_df["Horse"] = rp_df["Horse"].str.lower().str.strip()
    tf_df["Horse"] = tf_df["Horse"].str.lower().str.strip()
    merged = pd.merge(
        rp_df, tf_df,
        on=["Horse", "Race URL"],
        how="outer"
    )
    merged["Best Price"] = merged[["Odds_RP", "Odds_TF"]].max(axis=1)
    merged["Market Prob"] = 1 / merged["Best Price"]
    merged["Model Prob"] = 1 / merged.groupby("Race URL")["Horse"].transform("count")
    merged["Value Score"] = merged["Model Prob"] - merged["Market Prob"]
    return merged
