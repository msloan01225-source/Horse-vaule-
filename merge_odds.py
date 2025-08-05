import pandas as pd
from datetime import datetime
import pytz
import traceback

from racingpost_scraper import get_all_todays_data as get_rp_data
from timeform_scraper import get_all_todays_data as get_tf_data
from australia_scraper import get_all_todays_data as get_aus_data
from usa_scraper import get_all_todays_data as get_usa_data


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def safe_scraper(name, func):
    try:
        log(f"Starting {name} scraper...")
        df = func()
        log(f"{name} scraper finished – {len(df)} rows found.")
        return df
    except Exception as e:
        log(f"❌ {name} scraper FAILED: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def calculate_countdown(row):
    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)

    # Assign timezone based on country
    if row["Country"] == "UK":
        tz = pytz.timezone("Europe/London")
    elif row["Country"] == "AUS":
        tz = pytz.timezone("Australia/Sydney")
    elif row["Country"] == "USA":
        tz = pytz.timezone("US/Eastern")
    else:
        tz = pytz.UTC

    try:
        race_time_local = datetime.strptime(row["Race Time"], "%H:%M")
        race_time_local = tz.localize(datetime.combine(datetime.now(tz).date(), race_time_local.time()))
        countdown_minutes = int((race_time_local - now_utc.astimezone(tz)).total_seconds() / 60)
    except Exception:
        countdown_minutes = None

    if countdown_minutes is not None:
        if countdown_minutes > 0:
            return f"{countdown_minutes} mins"
        elif -10 <= countdown_minutes <= 0:
            return "Starting Now"
        else:
            return "Finished"
    return "Unknown"


def merge_best_prices():
    rp_df = safe_scraper("Racing Post", get_rp_data)
    tf_df = safe_scraper("Timeform", get_tf_data)
    aus_df = safe_scraper("Australia", get_aus_data)
    usa_df = safe_scraper("USA", get_usa_data)

    # UK Best Price merge
    if not rp_df.empty:
        rp_df["Horse_norm"] = rp_df["Horse"].str.lower().str.strip()
    if not tf_df.empty:
        tf_df["Horse_norm"] = tf_df["Horse"].str.lower().str.strip()

    if not rp_df.empty and not tf_df.empty:
        log("Merging UK Racing Post + Timeform odds...")
        merged_uk = pd.merge(
            rp_df, tf_df,
            on=["Horse_norm", "Race URL", "Race Time"],
            how="outer",
            suffixes=("_RP", "_TF")
        )
        merged_uk["Best Price"] = merged_uk[["Odds_RP", "Odds_TF"]].max(axis=1)
        merged_uk["Market Prob"] = 1 / merged_uk["Best Price"]
        merged_uk["Model Prob"] = 1 / merged_uk.groupby("Race URL")["Horse_norm"].transform("count")
        merged_uk["Value Score"] = merged_uk["Model Prob"] - merged_uk["Market Prob"]
        merged_uk["Country"] = "UK"
        merged_uk["Horse"] = merged_uk["Horse_norm"]
        uk_final = merged_uk[["Country", "Race Time", "Horse", "Best Price",
                              "Market Prob", "Model Prob", "Value Score", "Race URL",
                              "Odds_RP", "Odds_TF"]]
    else:
        log("⚠ Skipping UK merge – one or both UK scrapers returned no data.")
        uk_final = pd.DataFrame()

    # Rename AUS & USA best price columns
    if not aus_df.empty:
        aus_df = aus_df.rename(columns={"Odds_AUS": "Best Price"})
    if not usa_df.empty:
        usa_df = usa_df.rename(columns={"Odds_USA": "Best Price"})

    # Combine all countries
    combined = pd.concat([uk_final, aus_df, usa_df], ignore_index=True)

    if not combined.empty:
        combined["Countdown"] = combined.apply(calculate_countdown, axis=1)
        # Remove finished races
        combined = combined[~combined["Countdown"].isin(["Finished"])]
        combined = combined.sort_values(by=["Race Time", "Value Score"], ascending=[True, False]).reset_index(drop=True)
        log(f"Final combined dataset: {len(combined)} rows.")
    else:
        log("❌ No race data returned from any scraper.")

    return combined
