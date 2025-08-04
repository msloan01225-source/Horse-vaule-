
import streamlit as st
import pandas as pd
from scraper import get_today_race_urls, get_race_data

st.set_page_config(page_title="Horse Value Finder", layout="wide")
st.title("🐎 Top Horse Racing Value Finder")

st.markdown("""
Automatically scans all of today's UK races and ranks horses by **value score**.
Use the sidebar filters to narrow down top-value opportunities for the day.
""")

# Sidebar filters
st.sidebar.header("🔍 Filters")
min_score = st.sidebar.slider("Minimum Value Score", 0.00, 0.20, 0.05, 0.01)
odds_range = st.sidebar.slider("Exchange Odds Range", 1.0, 50.0, (2.0, 15.0))

if st.button("📊 Scan Today's Races"):
    with st.spinner("Scanning Racing Post..."):
        urls = get_today_race_urls()
        all_data = []

        for url in urls:
            try:
                df = get_race_data(url)
                if not df.empty:
                    num_runners = len(df)
                    df["Market Prob"] = 1 / df["Exchange Odds"]
                    df["Model Prob"] = 1 / num_runners
                    df["Value Score"] = df["Model Prob"] - df["Market Prob"]
                    all_data.append(df)
            except Exception:
                continue

        if all_data:
            full_df = pd.concat(all_data)
            # Apply filters
            filtered = full_df[
                (full_df["Exchange Odds"].between(odds_range[0], odds_range[1])) &
                (full_df["Value Score"] >= min_score)
            ].copy()

            filtered.sort_values("Value Score", ascending=False, inplace=True)

            # Format display
            filtered["Exchange Odds"] = filtered["Exchange Odds"].round(2)
            filtered["Market Prob"] = filtered["Market Prob"].round(3)
            filtered["Model Prob"] = filtered["Model Prob"].round(3)
            filtered["Value Score"] = filtered["Value Score"].round(3)

            st.success(f"✅ Found {len(filtered)} horses with value score ≥ {min_score}")
            st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
        else:
            st.error("No valid races found.")
else:
    st.info("Click the button to scan today's races.")
