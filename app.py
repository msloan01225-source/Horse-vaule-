import streamlit as st
import pandas as pd
from racingpost_scraper import get_racingpost_data
from timeform_scraper import get_timeform_data
from merge_odds import merge_best_prices

st.set_page_config(page_title="BetEdge", layout="wide")

st.title("🏇 BetEdge – Value Finder")

@st.cache_data
def load_data(day):
    rp_df = get_racingpost_data(day=day, debug=False)
    tf_df = get_timeform_data(day=day, debug=False)

    if rp_df.empty or tf_df.empty:
        return pd.DataFrame()

    merged = pd.merge(rp_df, tf_df, on=["Horse", "Race"], how="inner", suffixes=("_RP", "_TF"))
    merged = merge_best_prices(merged)

    # Calculate value % and confidence
    merged["Value %"] = ((merged["Best Odds"] * merged["Win Probability"]) - 1) * 100
    merged["Confidence"] = (merged["Win Probability"] * 100).round(1)

    merged = merged.sort_values(by="Value %", ascending=False).reset_index(drop=True)
    return merged

day_option = st.radio("Select race day", ["Today", "Tomorrow"])
df = load_data(day_option)

if df.empty:
    st.warning("⚠ No race data available. Check scrapers.")
else:
    # Top 2 separately
    st.subheader("🔥 Top 2 Value Picks")
    st.dataframe(df.head(2)[["Race", "Horse", "Best Odds", "Win Probability", "Value %", "Confidence"]])

    # Full table
    st.subheader("📋 All Races – Sorted by Value")
    st.dataframe(df[["Race", "Horse", "Best Odds", "Win Probability", "Value %", "Confidence"]])
