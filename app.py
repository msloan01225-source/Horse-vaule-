import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from scraper import get_today_race_urls, get_tomorrow_race_urls, get_race_data
from merge_odds import merge_best_prices

st.set_page_config(page_title="BetEdge Horse Value Finder", layout="wide")

st.markdown("# 🐎 BetEdge Horse Value Finder")
st.markdown("_Scan today's and tomorrow's UK races for the best value horses based on exchange odds vs statistical probability._")

# --- Sidebar filters ---
race_day = st.sidebar.selectbox("📅 Select Race Day", ["Today", "Tomorrow"])
high_value_only = st.sidebar.checkbox("🔍 Show only high-value horses (Value Score > 0.05)", value=True)
min_odds = st.sidebar.slider("🎯 Minimum Odds Filter", min_value=1.0, max_value=50.0, value=4.0, step=0.5)

# --- Cache race data for 1 hour ---
@st.cache_data(ttl=3600)
def load_race_data(day="Today"):
    urls = get_today_race_urls() if day == "Today" else get_tomorrow_race_urls()
    all_dfs = []
    for url in urls:
        df = get_race_data(url)
        if not df.empty:
            # Calculate Market Probability and Model Probability
            num_runners = len(df)
            df["Market Prob"] = 1 / df["Exchange Odds"]
            df["Model Prob"] = 1 / num_runners
            df["Value Score"] = df["Model Prob"] - df["Market Prob"]
            df["Race URL"] = url
            all_dfs.append(df)
    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

# --- Cache merged odds ---
@st.cache_data(ttl=3600)
def load_merged_odds():
    return merge_best_prices()

# --- Manual refresh button ---
if st.sidebar.button("🔄 Refresh Data Now"):
    st.cache_data.clear()
    st.experimental_rerun()

# --- Load data ---
race_data = load_race_data(day=race_day)
odds_data = load_merged_odds()

if race_data.empty or odds_data.empty:
    st.error("No race data or odds data available. Please try refreshing or check back later.")
    st.stop()

# --- Merge race data with best odds ---
df = pd.merge(race_data, odds_data, how="left", left_on=["Horse", "Race"], right_on=["Horse", "Race"])

# --- Filter by minimum odds ---
df = df[df["Exchange Odds"] >= min_odds]

# --- Filter high value horses only ---
if high_value_only:
    df = df[df["Value Score"] > 0.05]

if df.empty:
    st.warning("No horses matched your filters.")
    st.stop()

# --- Highlight Bet365 odds if best ---
def highlight_best_bet365(row):
    return ['background-color: #d4edda' if row["Exchange Odds"] == row["Best Odds"] else '' for _ in row]

# --- Prepare columns for display ---
df["Race Link"] = df["Race URL"].apply(lambda url: f"[Link]({url})")

display_cols = ["Horse", "Race", "Race Link", "Exchange Odds", "Best Odds", "Best Bookmaker", "Win %", "Place %", "Value Score"]

# Sort by Value Score descending
df = df.sort_values(by="Value Score", ascending=False)

# Reorder columns, hide 'Race URL' because Race Link replaces it
df_display = df[["Horse", "Race Link", "Exchange Odds", "Best Odds", "Best Bookmaker", "Win %", "Place %", "Value Score"]]
df_display = df_display.rename(columns={"Race Link": "Race"})

# --- Display DataFrame with styling ---
st.markdown(f"### 📈 Top Value Horses ({len(df_display)})")
st.dataframe(
    df_display.style.apply(highlight_best_bet365, axis=1)
    .format({
        "Exchange Odds": "{:.2f}",
        "Best Odds": "{:.2f}",
        "Win %": "{:.1f}%",
        "Place %": "{:.1f}%",
        "Value Score": "{:.3f}"
    })
    .set_properties(**{'text-align': 'left'})
    .set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'left')]
    }]),
    use_container_width=True
)
