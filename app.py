import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random

# ----------------- CONFIG -----------------
st.set_page_config(page_title="BetEdge", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏇⚽ BetEdge – Smart Betting Tracker</h1>", unsafe_allow_html=True)

# ----------------- MOCK GENERATORS -----------------
def generate_mock_horse_data(region="UK"):
    courses = {
        "UK": ["Ascot", "Newmarket", "Cheltenham", "York"],
        "USA": ["Belmont Park", "Gulfstream", "Churchill Downs", "Santa Anita"]
    }
    bookies = ["Bet365", "SkyBet", "PaddyPower", "WilliamHill"]

    rows = []
    for _ in range(100):
        course = random.choice(courses[region])
        horse = f"Horse {random.randint(1, 99)}"
        time = f"{random.randint(12, 17)}:{random.choice(['00', '15', '30', '45'])}"
        bookie = random.choice(bookies)
        odds = round(np.random.uniform(2, 8), 2)
        win_val = round(np.random.uniform(5, 25), 1)

        rows.append({
            "Time": time,
            "Course": course,
            "Horse": horse,
            "Bookie": bookie,
            "Best Odds": odds,
            "Win_Value": win_val,
            "Place_Value": round(win_val * 0.6, 1)
        })
    
    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

def generate_mock_football_data():
    teams = ["Man City", "Liverpool", "Chelsea", "Arsenal", "Barcelona", "Real Madrid"]
    bookies = ["Bet365", "SkyBet", "Betfair", "Coral"]
    markets = ["1X2", "Over/Under", "Both Teams to Score", "Correct Score"]
    
    rows = []
    for _ in range(50):
        team1, team2 = random.sample(teams, 2)
        market = random.choice(markets)
        bookie = random.choice(bookies)
        odds = round(np.random.uniform(1.5, 5.0), 2)
        value = round(np.random.uniform(5, 20), 1)
        rows.append({
            "Fixture": f"{team1} vs {team2}",
            "Market": market,
            "Bookie": bookie,
            "Odds": odds,
            "Value": value,
            "Predicted %": round((1 / odds) * 100, 1),
            "BetEdge %": round(((1 / odds) * 100 * 0.6) + (value * 0.4), 1)
        })
    return pd.DataFrame(rows)

# ----------------- SIDEBAR NAVIGATION -----------------
section = st.sidebar.selectbox("Select Sport", ["UK Horse Racing", "USA Horse Racing", "Football"])

# ----------------- HORSE RACING LOGIC -----------------
def show_horse_racing(region):
    st.markdown(f"### 🎯 {region} Racing Value Tracker")
    df = generate_mock_horse_data(region)

    courses = sorted(df["Course"].unique())
    bookies = sorted(df["Bookie"].unique())

    course_filter = st.multiselect("Filter by Course", options=["All"] + courses, default=["All"])
    bookie_filter = st.multiselect("Filter by Bookie", options=["All"] + bookies, default=["All"])
    view_mode = st.radio("View Mode", ["Charts", "Tables"])

    if "All" not in course_filter:
        df = df[df["Course"].isin(course_filter)]
    if "All" not in bookie_filter:
        df = df[df["Bookie"].isin(bookie_filter)]

    df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if df.empty:
        st.warning("No races match the selected filters.")
        return

    st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    def color_val(v):
        if v > 20: return 'background-color:#58D68D;color:black'
        if v > 10: return 'background-color:#F9E79F;color:black'
        return 'background-color:#F5B7B1;color:black'

    if view_mode == "Tables":
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win[["Time", "Course", "Horse", "Bookie", "Best Odds", "Win_Value", "Predicted Win %", "BetEdge Win %"]]
                     .style.applymap(color_val, subset=["BetEdge Win %"]))

        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place[["Time", "Course", "Horse", "Bookie", "Best Odds", "Place_Value", "Predicted Place %", "BetEdge Place %"]]
                     .style.applymap(color_val, subset=["BetEdge Place %"]))
    else:
        st.subheader("📊 Top 20 Win %")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("📊 Top 20 Place %")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])

# ----------------- FOOTBALL LOGIC -----------------
def show_football():
    st.markdown("### ⚽ Football Value Picks")
    df = generate_mock_football_data()

    market_filter = st.multiselect("Market Type", options=["All"] + df["Market"].unique().tolist(), default=["All"])
    bookie_filter = st.multiselect("Bookie", options=["All"] + df["Bookie"].unique().tolist(), default=["All"])

    if "All" not in market_filter:
        df = df[df["Market"].isin(market_filter)]
    if "All" not in bookie_filter:
        df = df[df["Bookie"].isin(bookie_filter)]

    df = df.sort_values("BetEdge %", ascending=False)

    if df.empty:
        st.warning("No football matches match the selected filters.")
        return

    st.dataframe(df.style.highlight_max(axis=0, subset=["BetEdge %"], color="lightgreen"))
    st.subheader("📊 Top 20 Football Picks")
    st.bar_chart(df.head(20).set_index("Fixture")["BetEdge %"])

# ----------------- ROUTER -----------------
if section == "UK Horse Racing":
    show_horse_racing("UK")
elif section == "USA Horse Racing":
    show_horse_racing("USA")
elif section == "Football":
    show_football()
