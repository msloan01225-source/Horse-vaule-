
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---- CONFIG ----
st.set_page_config(page_title="BetEdge – Value Tracker", layout="wide")

# ---- Sidebar Navigation ----
st.sidebar.title("📂 BetEdge Sports")
sport = st.sidebar.radio("Select Sport", ["🏇 UK Racing", "🏇 USA Racing", "⚽ Football"], index=0)

# ---- Shared Mock Functions ----
def generate_mock_racing_data(region="UK"):
    np.random.seed(42 if region == "UK" else 24)
    courses = ["Ascot", "Newbury", "Cheltenham", "Aintree", "York"] if region == "UK" else ["Belmont", "Gulfstream", "Santa Anita", "Churchill Downs"]
    bookies = ["Bet365", "SkyBet", "William Hill", "Coral"]
    horses = [f"Horse {i+1}" for i in range(50)]

    rows = []
    for horse in horses:
        course = np.random.choice(courses)
        bookie = np.random.choice(bookies)
        odds = np.random.uniform(2, 10)
        win_val = np.random.uniform(5, 25)
        rows.append({
            "Horse": horse,
            "Course": course,
            "Bookie": bookie,
            "Best Odds": round(odds, 2),
            "Win_Value": round(win_val, 1),
            "Place_Value": round(win_val * 0.6, 1)
        })
    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

def generate_mock_football_data():
    teams = [f"Team {i}" for i in range(1, 21)]
    bookies = ["Bet365", "SkyBet", "William Hill", "Paddy Power"]
    matches = []
    for i in range(10):
        team_a, team_b = np.random.choice(teams, 2, replace=False)
        bookie = np.random.choice(bookies)
        odds_a = round(np.random.uniform(1.5, 3.5), 2)
        odds_b = round(np.random.uniform(1.5, 3.5), 2)
        value = round(np.random.uniform(5, 25), 1)
        matches.append({
            "Match": f"{team_a} vs {team_b}",
            "Bookie": bookie,
            "Odds Team A": odds_a,
            "Odds Team B": odds_b,
            "Value %": value
        })
    return pd.DataFrame(matches)

def style_value(val):
    if val > 20: return "background-color: #58D68D; color: black"
    if val > 10: return "background-color: #F9E79F; color: black"
    return "background-color: #F5B7B1; color: black"

# ---- UK RACING ----
if sport == "🏇 UK Racing":
    st.title("🇬🇧 UK Horse Racing – BetEdge")
    df = generate_mock_racing_data("UK")

    courses = ["All"] + sorted(df["Course"].unique())
    bookies = ["All"] + sorted(df["Bookie"].unique())

    selected_course = st.sidebar.selectbox("Select Course", courses)
    selected_bookie = st.sidebar.selectbox("Select Bookie", bookies)

    if selected_course != "All":
        df = df[df["Course"] == selected_course]
    if selected_bookie != "All":
        df = df[df["Bookie"] == selected_bookie]

    st.subheader("🏆 Top 20 BetEdge Win %")
    st.dataframe(df.sort_values("BetEdge Win %", ascending=False).head(20).style.applymap(style_value, subset=["BetEdge Win %"]))

    st.subheader("🥈 Top 20 BetEdge Place %")
    st.dataframe(df.sort_values("BetEdge Place %", ascending=False).head(20).style.applymap(style_value, subset=["BetEdge Place %"]))

# ---- USA RACING ----
elif sport == "🏇 USA Racing":
    st.title("🇺🇸 USA Horse Racing – BetEdge")
    df = generate_mock_racing_data("USA")

    courses = ["All"] + sorted(df["Course"].unique())
    bookies = ["All"] + sorted(df["Bookie"].unique())

    selected_course = st.sidebar.selectbox("Select USA Course", courses)
    selected_bookie = st.sidebar.selectbox("Select Bookie", bookies)

    if selected_course != "All":
        df = df[df["Course"] == selected_course]
    if selected_bookie != "All":
        df = df[df["Bookie"] == selected_bookie]

    st.subheader("🏆 Top 20 BetEdge Win % (USA)")
    st.dataframe(df.sort_values("BetEdge Win %", ascending=False).head(20).style.applymap(style_value, subset=["BetEdge Win %"]))

    st.subheader("🥈 Top 20 BetEdge Place % (USA)")
    st.dataframe(df.sort_values("BetEdge Place %", ascending=False).head(20).style.applymap(style_value, subset=["BetEdge Place %"]))

# ---- FOOTBALL ----
elif sport == "⚽ Football":
    st.title("⚽ Football – Value Match Betting")
    df = generate_mock_football_data()
    st.dataframe(df.style.applymap(style_value, subset=["Value %"]))
