import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="BetEdge Sports Value Tracker", layout="wide")
st.title("📈 BetEdge – Sports Value Tracker")

# -------- Sidebar Navigation --------
st.sidebar.title("🏠 Menu")
sport = st.sidebar.radio("Select Sport", ["Horse Racing", "Football"])

# -------- Utility Functions --------
def generate_mock_horse_data():
    np.random.seed(42)
    horses = [f"Horse {i+1}" for i in range(30)]
    data = {
        "Horse": horses,
        "Race": [f"Course {np.random.randint(1,5)} {np.random.randint(12,18)}:{np.random.randint(0,60):02d}" for _ in horses],
        "Best Odds": np.round(np.random.uniform(2.0, 10.0, size=30), 2),
    }
    df = pd.DataFrame(data)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Win_Value"] = np.round(np.random.uniform(5, 25, size=30), 1)
    df["Place_Value"] = np.round(df["Win_Value"] * 0.6, 1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6 + df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Win %"] * 0.6 + df["Place_Value"] * 0.4)).round(1)
    return df

def generate_mock_football_data():
    np.random.seed(7)
    matches = [f"Team {i+1} vs Team {i+2}" for i in range(1, 31, 2)]
    odds = np.round(np.random.uniform(1.8, 4.5, size=len(matches)), 2)
    value = np.round(np.random.uniform(5, 20, size=len(matches)), 1)
    df = pd.DataFrame({
        "Match": matches,
        "Best Odds": odds,
        "Implied %": (1 / odds * 100).round(1),
        "Model %": value,
    })
    df["BetEdge %"] = ((df["Implied %"] * 0.5 + df["Model %"] * 0.5)).round(1)
    return df

# -------- Main Interface --------
st.markdown(f"**Last updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

if sport == "Horse Racing":
    df = generate_mock_horse_data()
    view = st.radio("View Mode", ["Tables", "Charts"], horizontal=True)
    
    if view == "Tables":
        st.subheader("🏇 Top Horse Racing Value Bets")
        st.dataframe(df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True))
    else:
        st.subheader("📊 BetEdge Win % (Top 20 Horses)")
        st.bar_chart(df.sort_values("BetEdge Win %", ascending=False).head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("📊 BetEdge Place % (Top 20 Horses)")
        st.bar_chart(df.sort_values("BetEdge Place %", ascending=False).head(20).set_index("Horse")["BetEdge Place %"])

elif sport == "Football":
    df = generate_mock_football_data()
    view = st.radio("View Mode", ["Tables", "Charts"], horizontal=True)

    if view == "Tables":
        st.subheader("⚽ Top Football Value Bets")
        st.dataframe(df.sort_values("BetEdge %", ascending=False).reset_index(drop=True))
    else:
        st.subheader("📊 BetEdge % (Top Football Matches)")
        st.bar_chart(df.sort_values("BetEdge %", ascending=False).set_index("Match")["BetEdge %"])
