import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="EdgeBet Phase 2", layout="wide")

# --- SIDEBAR MENU ---
st.sidebar.title("📂 Navigation")
main_section = st.sidebar.radio("Go to", ["🏠 Home", "🐎 Horse Racing", "⚽ Football", "🧠 EdgeBrain™"])

# --- MOCK DATA GENERATION ---
def generate_horse_data(region="UK"):
    courses = ["Ascot", "York", "Newbury", "Goodwood"] if region == "UK" else ["Belmont", "Churchill Downs", "Santa Anita"]
    bookies = ["SkyBet", "Bet365", "Ladbrokes", "Coral", "PaddyPower"]
    rows = []
    for i in range(50):
        course = random.choice(courses)
        horse = f"Horse_{i+1}"
        odds = round(random.uniform(2.0, 10.0), 2)
        win_val = round(random.uniform(5, 25), 1)
        place_val = round(win_val * 0.6, 1)
        bookie = random.choice(bookies)
        rows.append({
            "Course": course,
            "Horse": horse,
            "Bookmaker": bookie,
            "Best Odds": odds,
            "Win_Value": win_val,
            "Place_Value": place_val
        })
    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

def generate_football_data():
    matches = []
    leagues = ["Premier League", "Championship", "La Liga", "Serie A"]
    for i in range(30):
        league = random.choice(leagues)
        match = f"Team A{i} vs Team B{i}"
        edge = round(random.uniform(5, 30), 1)
        market = random.choice(["1X2", "Over/Under", "Both Teams to Score"])
        matches.append({
            "League": league,
            "Match": match,
            "Market": market,
            "Edge %": edge
        })
    return pd.DataFrame(matches)

# --- COLOR HIGHLIGHT ---
def color_val(v):
    if v > 20: return 'background-color:#58D68D;color:black'
    if v > 10: return 'background-color:#F9E79F;color:black'
    return 'background-color:#F5B7B1;color:black'

# --- HOME PAGE ---
if main_section == "🏠 Home":
    st.title("🏠 Welcome to EdgeBet")
    st.markdown("""
    EdgeBet is the future of smart betting.

    Navigate through the sports using the menu to explore:

    - 📊 Value opportunities in UK & USA Horse Racing  
    - ⚽ Football value bets with Edge %  
    - 🧠 The EdgeBrain™ – our proprietary betting intelligence engine
    """)

# --- HORSE RACING ---
elif main_section == "🐎 Horse Racing":
    st.title("🐎 Horse Racing Value Finder")
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Select Region", ["UK", "USA"])
    with col2:
        market = st.selectbox("Market Type", ["Win", "Place"])

    df_horse = generate_horse_data(region)

    # Filters
    courses = sorted(df_horse["Course"].unique())
    bookies = sorted(df_horse["Bookmaker"].unique())
    course_filter = st.multiselect("Filter by Course", ["All"] + courses, default=["All"])
    bookie_filter = st.multiselect("Filter by Bookmaker", ["All"] + bookies, default=["All"])

    if "All" not in course_filter:
        df_horse = df_horse[df_horse["Course"].isin(course_filter)]
    if "All" not in bookie_filter:
        df_horse = df_horse[df_horse["Bookmaker"].isin(bookie_filter)]

    view_mode = st.radio("View Mode", ["📈 Charts", "📋 Tables"])

    if view_mode == "📋 Tables":
        st.subheader(f"🏆 {market} Rankings – BetEdge %")
        sort_col = "BetEdge Win %" if market == "Win" else "BetEdge Place %"
        st.dataframe(df_horse.sort_values(sort_col, ascending=False).style.applymap(color_val, subset=[sort_col]))
    else:
        st.subheader(f"📊 {market} – BetEdge % Chart (Top 20)")
        sort_col = "BetEdge Win %" if market == "Win" else "BetEdge Place %"
        st.bar_chart(df_horse.sort_values(sort_col, ascending=False).head(20).set_index("Horse")[sort_col])

# --- FOOTBALL ---
elif main_section == "⚽ Football":
    st.title("⚽ Football Value Insights")
    df_foot = generate_football_data()
    league_filter = st.multiselect("Filter by League", ["All"] + sorted(df_foot["League"].unique()), default=["All"])
    if "All" not in league_filter:
        df_foot = df_foot[df_foot["League"].isin(league_filter)]
    st.dataframe(df_foot.sort_values("Edge %", ascending=False).style.applymap(color_val, subset=["Edge %"]))

# --- EDGEBRAIN ---
elif main_section == "🧠 EdgeBrain™":
    st.title("🧠 EdgeBrain – Predictive Engine")
    st.markdown("""
    EdgeBrain™ is our core algorithmic brain powering value detection and probability adjustments.

    **Coming soon:**
    - Smart Picks
    - Interactive Value Zones
    - AI-driven Bet Recommendations
    """)
    st.info("Prototype visuals and simulations will appear here soon.")
