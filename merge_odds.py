import pandas as pd
from racingpost_scraper import get_all_todays_data as get_rp_data
from timeform_scraper import get_all_todays_data as get_tf_data

def merge_best_prices(day="Today", debug=False):
    rp_df = get_rp_data(day=day, debug=debug)
    tf_df = get_tf_data(day=day, debug=debug)

    if rp_df.empty or tf_df.empty:
        print(f"[Merge] ⚠ Missing data – RP rows: {len(rp_df)}, TF rows: {len(tf_df)}")
        return pd.DataFrame()

    merged = pd.merge(
        rp_df,
        tf_df,
        on=["Race", "Time", "Course"],
        how="inner",
        suffixes=("_RP", "_TF")
    )

    print(f"[Merge] ✅ Merged {len(merged)} races")
    return merged
