import streamlit as st
import pandas as pd
from datetime import datetime
from racingpost_scraper import get_racingpost_data
from timeform_scraper import get_timeform_data

st.set_page_config(page_title="BetEdge - Horse Value Tracker", layout="wide")

st.title("🐎 BetEdge - Horse Value Finder")
st.markdown("_Find the best value horses based on market vs model probability._")

# Sidebar filters
st.sidebar.header("Filters")
race_day = st.sidebar.selectbox("📅 Race Day", ["Today", "Tomorrow"])
high_value_only = st.sidebar.checkbox("🔍 Show only high-value horses (Value Score > 0.05)", value=True)
min_odds = st.sidebar.slider("🎯 Minimum Odds", 1.0, 50.0, 4.0, 0.5)

# Debug mode
debug_mode = st.sidebar.checkbox("🐞 Debug Mode", value=False)

@st.cache_data(ttl=3600)
def load_uk_data(day):
    if debug_mode:
        st.write(f"🔄 Fetching Racing Post data for {day}...")
    rp_df = get_racingpost_data(day=day, debug=debug_mode)

    if debug_mode:
        st.write(f"🔄 Fetching Timeform data for {day}...")
    tf_df = get_timeform_data(day=day, debug=debug_mode)

    if rp_df.empty or tf_df.empty:
        if debug_mode:
            st.error("❌ One or both scrapers returned no data.")
        return pd.DataFrame()

    # Merge on Horse + Race
    df = pd.merge(rp_df, tf_df, on=["Horse", "Race"], how="inner", suffixes=("_RP", "_TF"))

    # Calculate Value Score
    df["Market Prob"] = 1 / df["Odds_RP"]
    df["Model Prob"] = df["WinProb_TF"] / 100  # TF win probability in %
    df["Value Score"] = df["Model Prob"] - df["Market Prob"]

    # Place probability
    if "PlaceProb_TF" in df.columns:
        df["Place Prob"] = df["PlaceProb_TF"] / 100

    return df

# Refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Load merged data
df = load_uk_data(race_day)

if df.empty:
    st.error("No race data found. Try enabling Debug Mode to see scraper output.")
else:
    # Filter
    if high_value_only:
        df = df[df["Value Score"] > 0.05]
    df = df[df["Odds_RP"] >= min_odds]

    if df.empty:
        st.warning("No horses matched your filters.")
    else:
        df = df.sort_values(by="Value Score", ascending=False)

        st.markdown(f"### 📈 Top Value Horses ({len(df)} found)")
        display_cols = ["Horse", "Race", "Odds_RP", "Market Prob", "Model Prob", "Value Score"]
        if "Place Prob" in df.columns:
            display_cols.append("Place Prob")

        st.dataframe(df[display_cols].style.format({
            "Odds_RP": "{:.2f}",
            "Market Prob": "{:.2%}",
            "Model Prob": "{:.2%}",
            "Value Score": "{:.3f}",
            "Place Prob": "{:.2%}"
        }))
