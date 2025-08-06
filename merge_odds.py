import pandas as pd
from racingpost_scraper import get_racingpost_data
from timeform_scraper import get_timeform_data

def merge_best_prices(day="Today", debug=False):
    rp = get_racingpost_data(day, debug=debug)
    tf = get_timeform_data(day, debug=debug)

    if debug:
        print(f"[Merge] RP rows: {len(rp)}, TF rows: {len(tf)}")

    if rp.empty or tf.empty:
        return pd.DataFrame()

    merged = pd.merge(
        rp,
        tf,
        on=["Race", "Time", "Course", "RaceName"],
        how="inner",
        suffixes=("_RP", "_TF")
    )

    if debug:
        print(f"[Merge] Merged {len(merged)} races")

    # Placeholder columns for best odds, win/place probability
    # Replace with your actual logic or scraping later
    merged["Best Odds"] = 1.0
    merged["WinProb"] = 0.0
    merged["PlaceProb"] = 0.0

    return merged
