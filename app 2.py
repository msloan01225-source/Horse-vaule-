
import streamlit as st
import pandas as pd
from scraper import get_today_race_urls, get_race_data

st.set_page_config(page_title="Horse Value Finder", layout="wide")
st.title("🐎 Top Horse Racing Value Finder")

st.markdown("""
This app automatically scans all of today's UK races from Racing Post,
analyzes the odds, and ranks horses based on value.

**Value Score** is calculated as: `Model Prob - Market Prob`, where:
- Market Prob = 1 / Odds
- Model Prob = placeholder (assumes equal chance per runner)
""")

if st.button("📊 Scan All Today's Races"):
    with st.spinner("Fetching race data... please wait ⏳"):
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
            except Exception as e:
                continue

        if all_data:
            full_df = pd.concat(all_data)
            ranked = full_df.sort_values("Value Score", ascending=False)
            st.success(f"✅ Found {len(ranked)} horses across {len(urls)} races.")
            st.dataframe(ranked.reset_index(drop=True))
        else:
            st.error("No races or odds found today.")
else:
    st.info("Click the button above to scan today's races.")
