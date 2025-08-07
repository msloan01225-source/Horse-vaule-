
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(page_title="BetEdge - Horse Racing", layout="wide")
st.title("🏇 BetEdge – Horse Racing Value Tracker")

# Mock data generation
@st.cache_data(ttl=600)
def load_mock_data():
    np.random.seed(42)
    horses = [f"Horse {i}" for i in range(1, 21)]
    courses = np.random.choice(["Ascot", "Newmarket", "Cheltenham", "York"], size=20)
    bookies = np.random.choice(["Bet365", "SkyBet", "PaddyPower", "WilliamHill"], size=20)
    odds = np.round(np.random.uniform(2.0, 10.0, size=20), 2)
    win_val = np.round(np.random.uniform(5, 25, size=20), 1)

    df = pd.DataFrame({
        "Horse": horses,
        "Course": courses,
        "Bookie": bookies,
        "Best Odds": odds,
        "Win_Value": win_val,
    })

    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

df = load_mock_data()

# Filter sidebar
with st.sidebar:
    st.header("🔎 Filters")
    selected_course = st.multiselect("Select Course(s)", options=df["Course"].unique(), default=list(df["Course"].unique()))
    selected_bookie = st.multiselect("Select Bookie(s)", options=df["Bookie"].unique(), default=list(df["Bookie"].unique()))
    min_value = st.slider("Minimum BetEdge Win %", 0, 100, 0)

# Filtered dataframe
filtered_df = df[
    (df["Course"].isin(selected_course)) &
    (df["Bookie"].isin(selected_bookie)) &
    (df["BetEdge Win %"] >= min_value)
]

# Sort and display
df_win = filtered_df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
df_place = filtered_df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

st.write(f"📅 Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
st.subheader("🏆 Top Win Rankings")
st.dataframe(df_win[["Horse", "Course", "Bookie", "Best Odds", "BetEdge Win %"]])

st.subheader("🥈 Top Place Rankings")
st.dataframe(df_place[["Horse", "Course", "Bookie", "Best Odds", "BetEdge Place %"]])

# Charts (optional - can be moved to tabs)
st.subheader("📊 BetEdge Win % (Top 10)")
st.bar_chart(df_win.head(10).set_index("Horse")["BetEdge Win %"])

st.subheader("📊 BetEdge Place % (Top 10)")
st.bar_chart(df_place.head(10).set_index("Horse")["BetEdge Place %"])
