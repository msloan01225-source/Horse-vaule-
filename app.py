import streamlit as st
import pandas as pd
from datetime import datetime
from racingpost_scraper import get_racingpost_data
from timeform_scraper import get_timeform_data

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="BetEdge – UK Horse Value Tracker", layout="wide")

# --- LOGO ---
st.image("logo.png", width=250)  # Make sure 'logo.png' is in your repo root

st.markdown("## 🐎 BetEdge – UK Horse Value Tracker")
st.markdown("_Live UK racecards from Racing Post & Timeform, combined into one feed._")

# --- FILTERS ---
col1, col2 = st.columns(2)
with col1:
    race_day = st.selectbox("📅 Select Race Day", ["Today", "Tomorrow"])
with col2:
    show_duplicates = st.checkbox("Show Duplicate Races from Both Sources", value=False)

# --- UK DATA LOADER ---
@st.cache_data(ttl=3600)
def load_uk_data(day="Today"):
    print(f"[UK Loader] 🔄 Starting UK data load for {day}")

    # Racing Post
    rp_df = get_racingpost_data(
        day,
        use_proxies=st.session_state.get("use_proxies", False),
        debug=st.session_state.get("debug_mode", False)
    )
    print(f"[UK Loader] 📊 Racing Post returned {len(rp_df)} rows")

    # Timeform
    tf_df = get_timeform_data(
        day,
        use_proxies=st.session_state.get("use_proxies", False),
        debug=st.session_state.get("debug_mode", False)
    )
    print(f"[UK Loader] 📊 Timeform returned {len(tf_df)} rows")

    # Merge
    if not rp_df.empty and not tf_df.empty:
        merged_df = pd.concat([rp_df, tf_df], ignore_index=True)
        if not show_duplicates:
            merged_df = merged_df.drop_duplicates(subset=["Race"])
        print(f"[UK Loader] ✅ Merged total: {len(merged_df)} rows")
        return merged_df
    elif not rp_df.empty:
        print("[UK Loader] ⚠ Timeform empty, using Racing Post only")
        return rp_df
    elif not tf_df.empty:
        print("[UK Loader] ⚠ Racing Post empty, using Timeform only")
        return tf_df
    else:
        print("[UK Loader] ❌ No UK race data")
        return pd.DataFrame()

# --- REFRESH BUTTON ---
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()
    st.experimental_rerun()

# --- LOAD & DISPLAY ---
df = load_uk_data(race_day)

if df.empty:
    st.error("No UK race data found for the selected day.")
else:
    st.dataframe(df.reset_index(drop=True))
