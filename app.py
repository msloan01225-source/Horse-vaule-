import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from scraper import get_today_race_urls, get_tomorrow_race_urls, get_race_data

st.set_page_config(page_title="Horse Value App", layout="wide")

st.markdown("## 🐎 Horse Value Finder")
st.markdown("_Scan today's and tomorrow's UK races for the best value horses based on exchange odds vs statistical probability._")

# --- USER CONTROLS ---
col1, col2 = st.columns(2)
with col1:
    race_day = st.selectbox("📅 Select Race Day", ["Today", "Tomorrow"])
with col2:
    high_value_only = st.checkbox("🔍 Show only high-value horses (Value Score > 0.05)", value=True)

min_odds = st.slider("🎯 Minimum Odds Filter", min_value=1.0, max_value=50.0, value=4.0, step=0.5)

# --- CACHE + SCRAPE FUNCTIONS ---
@st.cache_data(ttl=86400)  # cache for 24 hours
def load_all_race_data(day="Today"):
    if day == "Today":
        urls = get_today_race_urls()
    else:
        urls = get_tomorrow_race_urls()

    all_data = []
    for url in urls:
        df = get_race_data(url)
        if not df.empty:
            num_runners = len(df)
            df["Market Prob"] = 1 / df["Exchange Odds"]
            df["Model Prob"] = 1 / num_runners
            df["Value Score"] = df["Model Prob"] - df["Market Prob"]
            df["Race URL"] = url
            all_data.append(df)
    if all_data:
        return pd.concat(all_data)
    else:
        return pd.DataFrame()

# --- RUN SCAN ---
if st.button("🚀 Scan Races"):
    with st.spinner("Loading race data..."):
        df = load_all_race_data(day=race_day)
        if df.empty:
            st.error("No race data found.")
        else:
            # --- FILTERS ---
            if high_value_only:
                df = df[df["Value Score"] > 0.05]
            df = df[df["Exchange Odds"] >= min_odds]

            if df.empty:
                st.warning("No horses matched your filters.")
            else:
                df = df.sort_values(by="Value Score", ascending=False)
                st.markdown(f"### 📈 Top Value Horses ({len(df)} found)")
                st.dataframe(df[["Horse", "Exchange Odds", "Value Score", "Race URL"]].reset_index(drop=True), use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("Made with ❤️ by [Your Name]. Powered by Racing Post + Exchange Odds.")
