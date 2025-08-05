import streamlit as st
import pandas as pd
from datetime import datetime
import os
from racingpost_scraper import get_racingpost_data
from timeform_scraper import get_timeform_data

st.set_page_config(page_title="BetEdge – UK Value Tracker", layout="wide")

# Logo
st.image("logo.png", width=200)
st.title("🐎 BetEdge – UK Racing Value Finder")

# Persistent state for proxy toggle
if "use_proxies" not in st.session_state:
    st.session_state.use_proxies = False
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Sidebar controls
race_day = st.sidebar.selectbox("📅 Select Race Day", ["Today", "Tomorrow"])
st.sidebar.checkbox("Use Proxies", value=st.session_state.use_proxies, key="use_proxies")
st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode, key="debug_mode")

value_threshold = st.sidebar.slider("📊 Minimum Value Score", 0.00, 0.30, 0.05, 0.01)
high_value_only = st.sidebar.checkbox("🔍 Show Only High-Value Horses", True)
bookmaker_filter = st.sidebar.selectbox("🏦 Bookmaker Filter", ["All", "Bet365", "Betfair", "PaddyPower"])

min_odds = st.sidebar.slider("🎯 Minimum Odds Filter", 1.0, 50.0, 4.0, 0.5)

# Refresh button
if st.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()  # Updated for new Streamlit versions

@st.cache_data(ttl=3600)
def load_uk_data(day="Today"):
    rp_df = get_racingpost_data(day, use_proxies=st.session_state.use_proxies, debug=st.session_state.debug_mode)
    tf_df = get_timeform_data(day, use_proxies=st.session_state.use_proxies, debug=st.session_state.debug_mode)

    if rp_df.empty or tf_df.empty:
        return pd.DataFrame()

    # Merge RP & TF data
    df = pd.merge(rp_df, tf_df, on=["Horse", "Race"], how="inner", suffixes=("_RP", "_TF"))
    df["Market Prob"] = 1 / df["Odds"]
    df["Model Prob"] = 1 / df.groupby("Race")["Horse"].transform("count")
    df["Value Score"] = df["Model Prob"] - df["Market Prob"]

    return df

with st.spinner("⏳ Fetching latest UK race data…"):
    df = load_uk_data(race_day)

# Debug HTML download helper – only show if Debug Mode is ON
if st.session_state.debug_mode:
    import glob
    debug_files = glob.glob("*.html")
    if debug_files:
        st.subheader("📂 Debug HTML Files")
        for file in debug_files:
            with open(file, "rb") as f:
                st.download_button(
                    label=f"Download {file}",
                    data=f,
                    file_name=file,
                    mime="text/html"
                )

if df.empty:
    st.error("No race data found.")
else:
    if high_value_only:
        df = df[df["Value Score"] >= value_threshold]
    df = df[df["Odds"] >= min_odds]

    if bookmaker_filter != "All":
        df = df[df["Best Bookmaker"] == bookmaker_filter]

    df = df.sort_values(by="Value Score", ascending=False)
    total_runners = len(df)
    high_value_count = len(df[df["Value Score"] >= value_threshold])

    st.markdown(f"✅ Found **{total_runners}** runners (**{high_value_count}** high value) – Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    csv = df.to_csv(index=False)
    st.download_button(
        label="💾 Download CSV",
        data=csv,
        file_name=f"betedge_uk_races_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M')}.csv",
        mime="text/csv",
    )

    st.dataframe(df.style.format({"Value Score": "{:.3f}", "Odds": "{:.2f}"}))
