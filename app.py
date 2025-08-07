
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------- Page Config --------------------
st.set_page_config(page_title="BetEdge Pro", layout="wide")
st.markdown("## 🏇⚽ BetEdge – Smart Betting Insights")
st.markdown("Use the sidebar to navigate between sports, regions, and filters. All data is mock for demonstration.")

# -------------------- Sidebar Navigation --------------------
sport = st.sidebar.selectbox("🎯 Select Sport", ["Horse Racing", "Football"])

if sport == "Horse Racing":
    market = st.sidebar.radio("🏇 Market", ["UK", "USA"], horizontal=True)
    bookmaker_filter = st.sidebar.multiselect("Bookmaker", ["Bet365", "SkyBet", "PaddyPower"], default=["Bet365"])
    course_filter = st.sidebar.multiselect("Course", ["Ascot", "Aintree", "Cheltenham", "York", "Churchill Downs", "Belmont"], default=[])
    min_value = st.sidebar.slider("Minimum BetEdge Win %", 0, 100, 10)
    odds_range = st.sidebar.slider("Odds Range", 1.0, 20.0, (2.0, 10.0))

# -------------------- Mock Data Generator --------------------
def generate_mock_data(market):
    np.random.seed(42 if market == "UK" else 24)
    courses = ["Ascot", "Aintree", "Cheltenham", "York"] if market == "UK" else ["Churchill Downs", "Belmont", "Saratoga"]
    horses = [f"Horse {i}" for i in range(1, 101)]
    data = []
    for horse in horses:
        course = np.random.choice(courses)
        odds = np.round(np.random.uniform(1.5, 12), 2)
        win_val = np.round(np.random.uniform(5, 30), 1)
        place_val = np.round(win_val * 0.6, 1)
        predicted_win = np.round(100 / odds, 1)
        predicted_place = np.round(predicted_win * 0.6, 1)
        betedge_win = np.round((predicted_win * 0.6 + win_val * 0.4), 1)
        betedge_place = np.round((predicted_place * 0.6 + place_val * 0.4), 1)
        bookie = np.random.choice(["Bet365", "SkyBet", "PaddyPower"])
        data.append([course, horse, odds, win_val, place_val, predicted_win, predicted_place, betedge_win, betedge_place, bookie])
    df = pd.DataFrame(data, columns=["Course", "Horse", "Odds", "Win_Value", "Place_Value", "Predicted Win %", "Predicted Place %", "BetEdge Win %", "BetEdge Place %", "Bookmaker"])
    return df

# -------------------- Display Horse Racing --------------------
if sport == "Horse Racing":
    df = generate_mock_data(market)
    df = df[(df["Odds"] >= odds_range[0]) & (df["Odds"] <= odds_range[1])]
    df = df[df["BetEdge Win %"] >= min_value]
    if course_filter:
        df = df[df["Course"].isin(course_filter)]
    if bookmaker_filter:
        df = df[df["Bookmaker"].isin(bookmaker_filter)]
    df = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🏆 Top 20 Win % – {market} Market")
        st.bar_chart(df.head(20).set_index("Horse")["BetEdge Win %"])
    with col2:
        st.subheader(f"🥈 Top 20 Place % – {market} Market")
        st.bar_chart(df.head(20).set_index("Horse")["BetEdge Place %"])

    st.subheader("📋 Full Rankings Table")
    st.dataframe(df[["Course", "Horse", "Odds", "Bookmaker", "BetEdge Win %", "BetEdge Place %"]])

# -------------------- Display Football (Mock Placeholder) --------------------
elif sport == "Football":
    st.subheader("⚽ Football – Coming Soon")
    matches = [("Chelsea vs Arsenal", "Premier League"), ("Real Madrid vs Barca", "La Liga"), ("Inter vs Milan", "Serie A")]
    df_foot = pd.DataFrame(matches, columns=["Match", "League"])
    df_foot["Odds"] = np.random.uniform(1.8, 4.0, len(df_foot)).round(2)
    df_foot["Value %"] = np.random.uniform(5, 25, len(df_foot)).round(1)
    df_foot["Predicted %"] = (100 / df_foot["Odds"]).round(1)
    df_foot["BetEdge %"] = (0.6 * df_foot["Predicted %"] + 0.4 * df_foot["Value %"]).round(1)
    st.dataframe(df_foot)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("🔒 Powered by **BetEdge Analytics** | Mock data only | © 2025")
