import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="EdgeBet – Smart Value Tracker", layout="wide")
st.title("🎯 EdgeBet – Smart Value Tracker")
st.markdown("Your daily dashboard of high-value opportunities across sports betting.")

# ---------------------------
# Sidebar Menu
# ---------------------------
sport = st.sidebar.selectbox("Select Sport", ["Horse Racing (UK)", "Horse Racing (USA)", "Football (Coming Soon)"])
view_mode = st.sidebar.radio("View Mode", ["Charts", "Tables", "EdgeBrain Lab"])
selected_course = st.sidebar.multiselect("Select Course(s)", ["All", "Ascot", "York", "Cheltenham", "Aintree", "Churchill Downs", "Belmont"], default=["All"])
selected_bookies = st.sidebar.multiselect("Select Bookie(s)", ["All", "Bet365", "SkyBet", "William Hill", "Coral"], default=["All"])

# ---------------------------
# Mock Data Generator
# ---------------------------
def generate_mock_data(region="UK"):
    np.random.seed(42)
    courses = {
        "UK": ["Ascot", "York", "Cheltenham", "Aintree"],
        "USA": ["Churchill Downs", "Belmont", "Gulfstream", "Santa Anita"]
    }
    bookies = ["Bet365", "SkyBet", "William Hill", "Coral"]
    rows = []
    for course in courses.get(region, []):
        for i in range(1, 6):
            horse = f"{course} Runner {i}"
            odds = np.round(np.random.uniform(2, 8), 2)
            model_odds = np.round(np.random.uniform(2, 6), 2)
            win_val = np.round(np.random.uniform(5, 25), 1)
            bookie = np.random.choice(bookies)
            rows.append({
                "Time": f"{12+i}:00",
                "Course": course,
                "Horse": horse,
                "Bookie": bookie,
                "Best Odds": odds,
                "Model Odds": model_odds,
                "Win_Value": win_val,
                "Place_Value": round(win_val * 0.6, 1)
            })
    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Edge %"] = ((1 / df["Model Odds"] - 1 / df["Best Odds"]) * df["Best Odds"] * 100).round(1)
    df["Suggested Stake (£)"] = (df["Edge %"] / 2).clip(1, 50).round(2)
    return df

# ---------------------------
# Data Filtering
# ---------------------------
region = "UK" if "UK" in sport else "USA"
df = generate_mock_data(region)

if "All" not in selected_course:
    df = df[df["Course"].isin(selected_course)]

if "All" not in selected_bookies:
    df = df[df["Bookie"].isin(selected_bookies)]

df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# ---------------------------
# View Modes
# ---------------------------
def color_val(v):
    if v > 20: return 'background-color:#58D68D;color:black'
    if v > 10: return 'background-color:#F9E79F;color:black'
    return 'background-color:#F5B7B1;color:black'

if view_mode == "Charts":
    st.subheader("📊 Top 20 – BetEdge Win %")
    st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
    st.subheader("📊 Top 20 – BetEdge Place %")
    st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])

elif view_mode == "Tables":
    st.subheader("🏆 Win Rankings")
    st.dataframe(df_win[["Time", "Course", "Horse", "Bookie", "Best Odds", "Win_Value", "BetEdge Win %"]]
                 .style.applymap(color_val, subset=["BetEdge Win %"]))

    st.subheader("🥈 Place Rankings")
    st.dataframe(df_place[["Time", "Course", "Horse", "Bookie", "Best Odds", "Place_Value", "BetEdge Place %"]]
                 .style.applymap(color_val, subset=["BetEdge Place %"]))

elif view_mode == "EdgeBrain Lab":
    st.subheader("🧠 EdgeBrain Lab – Value Detection Engine")
    st.info("This is a preview of value-based betting opportunities derived from mock model calculations.")
    st.dataframe(df.sort_values("Edge %", ascending=False)
                 [["Time", "Course", "Horse", "Bookie", "Best Odds", "Model Odds", "Edge %", "Suggested Stake (£)"]]
                 .style.applymap(color_val, subset=["Edge %"]))
