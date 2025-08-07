import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="BetEdge Sports Tracker", layout="wide")
st.title("🎯 BetEdge – Sports Value Tracker")

# --- Sidebar ---
st.sidebar.image("https://i.imgur.com/jXtdxZm.png", width=200)
sport = st.sidebar.selectbox("Select Sport", ["Horse Racing", "Football"])
st.sidebar.markdown("---")
st.sidebar.text_input("🔐 Username")
st.sidebar.text_input("🔑 Password", type="password")
st.sidebar.button("Login (placeholder)")

# --- Mock Data ---
def mock_horse_data():
    horses = [f"Horse {i}" for i in range(1, 31)]
    courses = ["Ascot", "Newmarket", "Cheltenham", "York"]
    bookies = ["Bet365", "SkyBet", "PaddyPower", "Ladbrokes"]
    data = []
    for horse in horses:
        course = np.random.choice(courses)
        bookie = np.random.choice(bookies)
        odds = round(np.random.uniform(2.0, 12.0), 2)
        win_val = round(np.random.uniform(5, 25), 1)
        place_val = round(win_val * 0.6, 1)
        data.append({
            "Course": course,
            "Horse": horse,
            "Bookmaker": bookie,
            "Best Odds": odds,
            "Win_Value": win_val,
            "Place_Value": place_val
        })
    df = pd.DataFrame(data)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

def mock_football_data():
    matches = [f"Team A{i} vs Team B{i}" for i in range(1, 21)]
    bookies = ["Bet365", "SkyBet", "PaddyPower", "Ladbrokes"]
    data = []
    for match in matches:
        bookie = np.random.choice(bookies)
        odds = round(np.random.uniform(1.5, 3.5), 2)
        value = round(np.random.uniform(10, 30), 1)
        data.append({
            "Match": match,
            "Bookmaker": bookie,
            "Odds": odds,
            "Value Score": value,
            "Predicted Win %": round((1 / odds) * 100, 1),
            "BetEdge %": round((1 / odds * 100 * 0.6 + value * 0.4), 1)
        })
    return pd.DataFrame(data)

# --- Horse Racing ---
if sport == "Horse Racing":
    st.header("🐎 Horse Racing Value Picks")
    df = mock_horse_data()

    with st.expander("🔍 Filter Options"):
        course_filter = st.multiselect("Course", df["Course"].unique())
        bookie_filter = st.multiselect("Bookmaker", df["Bookmaker"].unique())

    if course_filter:
        df = df[df["Course"].isin(course_filter)]
    if bookie_filter:
        df = df[df["Bookmaker"].isin(bookie_filter)]

    st.subheader("🥇 Top 3 BetEdge Win Picks")
    top3 = df.sort_values("BetEdge Win %", ascending=False).head(3)
    for i, row in top3.iterrows():
        st.markdown(f"**{i+1}. {row['Horse']} ({row['Course']})** – {row['Bookmaker']}")
        st.progress(row["BetEdge Win %"] / 100)
        st.caption(f"Win %: {row['BetEdge Win %']}% | Odds: {row['Best Odds']}")

    st.subheader("📊 Full Horse Data")
    st.dataframe(df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True))

    st.download_button("⬇ Download CSV", df.to_csv(index=False), file_name="horse_value_data.csv")

# --- Football ---
elif sport == "Football":
    st.header("⚽ Football Value Fixtures")
    df = mock_football_data()

    st.subheader("🥇 Top 3 Fixtures")
    top3 = df.sort_values("BetEdge %", ascending=False).head(3)
    for i, row in top3.iterrows():
        st.markdown(f"**{i+1}. {row['Match']} – {row['Bookmaker']}**")
        st.progress(row["BetEdge %"] / 100)
        st.caption(f"Edge: {row['BetEdge %']}% | Odds: {row['Odds']}")

    st.subheader("📊 Full Football Data")
    st.dataframe(df.sort_values("BetEdge %", ascending=False).reset_index(drop=True))

    st.download_button("⬇ Download CSV", df.to_csv(index=False), file_name="football_value_data.csv")
