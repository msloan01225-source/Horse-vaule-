import streamlit as st
import pandas as pd
from datetime import datetime

# ===== Placeholders for Scrapers =====
try:
    from racingpost_scraper import get_racingpost_data
except ImportError:
    def get_racingpost_data(day="Today"):
        return pd.DataFrame(columns=["Race", "Time", "Course", "Horse", "Win_Value", "Place_Value"])

try:
    from timeform_scraper import get_timeform_data
except ImportError:
    def get_timeform_data(day="Today"):
        return pd.DataFrame(columns=["Race", "Time", "Course", "Horse", "Win_Value", "Place_Value"])

# ===== Config =====
st.set_page_config(page_title="BetEdge Value Tracker", layout="wide")

# ===== Functions =====
def load_uk_data(day):
    rp_df = get_racingpost_data(day)
    tf_df = get_timeform_data(day)

    if rp_df.empty and tf_df.empty:
        return pd.DataFrame()

    # Merge on Race + Horse
    try:
        df = pd.merge(rp_df, tf_df, on=["Race", "Horse"], suffixes=("_RP", "_TF"))
    except KeyError:
        # If merge fails, just concat
        df = pd.concat([rp_df, tf_df], ignore_index=True)

    # Calculate average value scores
    if "Win_Value_RP" in df.columns and "Win_Value_TF" in df.columns:
        df["Win_Value"] = df[["Win_Value_RP", "Win_Value_TF"]].mean(axis=1)
    elif "Win_Value" not in df.columns:
        df["Win_Value"] = 0

    if "Place_Value_RP" in df.columns and "Place_Value_TF" in df.columns:
        df["Place_Value"] = df[["Place_Value_RP", "Place_Value_TF"]].mean(axis=1)
    elif "Place_Value" not in df.columns:
        df["Place_Value"] = 0

    return df


def color_confidence(val):
    """Colour code confidence based on value."""
    if val >= 70:
        color = "green"
    elif val >= 40:
        color = "yellow"
    else:
        color = "red"
    return f"background-color: {color}; color: black;"


# ===== UI =====
st.title("🏇 BetEdge – Premium Value Tracker")
st.write("Algorithm-driven **Win** & **Place** value finder for all UK races.")

race_day = st.selectbox("Select race day:", ["Today", "Tomorrow"])
df = load_uk_data(race_day)

if df.empty:
    st.warning("⚠ No race data available at the moment.")
else:
    # Separate Win & Place tables
    win_df = df.sort_values(by="Win_Value", ascending=False)
    place_df = df.sort_values(by="Place_Value", ascending=False)

    st.subheader("🏆 Win Value Rankings")
    st.dataframe(
        win_df[["Race", "Horse", "Win_Value", "Time", "Course"]]
        .style.applymap(color_confidence, subset=["Win_Value"])
    )

    st.subheader("🎯 Place Value Rankings")
    st.dataframe(
        place_df[["Race", "Horse", "Place_Value", "Time", "Course"]]
        .style.applymap(color_confidence, subset=["Place_Value"])
    )

# ===== Footer =====
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
