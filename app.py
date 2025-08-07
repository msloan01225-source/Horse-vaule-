import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ----------- CONFIG ----------- #
st.set_page_config(page_title="BetEdge", layout="wide")

# ----------- MOCK DATA LOADER ----------- #
@st.cache_data(ttl=600)
def load_mock_data(sport, region="UK"):
    np.random.seed(42)
    courses = {
        "UK": ["Ascot", "Cheltenham", "Newmarket", "York"],
        "USA": ["Belmont", "Saratoga", "Churchill", "Keeneland"]
    }
    horses = [f"Horse {i}" for i in range(1, 21)]
    bookies = ["SkyBet", "Bet365", "Coral", "PaddyPower", "William Hill"]
    data = []
    for course in courses.get(region, []):
        for horse in horses:
            odds = np.random.uniform(2, 8)
            win_val = np.random.uniform(5, 25)
            place_val = win_val * 0.6
            bookie = np.random.choice(bookies)
            data.append({
                "Course": course,
                "Horse": horse,
                "Bookie": bookie,
                "Best Odds": round(odds, 2),
                "Win_Value": round(win_val, 1),
                "Place_Value": round(place_val, 1),
                "Predicted Win %": round((1 / odds) * 100, 1),
                "Predicted Place %": round((1 / odds) * 60, 1)
            })
    df = pd.DataFrame(data)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

# ----------- SIDEBAR NAVIGATION ----------- #
st.sidebar.image("https://i.imgur.com/dYcYQ7E.png", width=200)
main_tab = st.sidebar.radio("📂 Select Sport", ["Horse Racing", "Football", "EdgeBrain™"])

# ----------- HORSE RACING SECTION ----------- #
if main_tab == "Horse Racing":
    st.title("🏇 Horse Racing – BetEdge Rankings")
    region = st.selectbox("Region", ["UK", "USA"])
    df = load_mock_data("racing", region=region)

    # Filters
    courses = sorted(df["Course"].unique())
    bookies = sorted(df["Bookie"].unique())

    selected_courses = st.multiselect("Filter by Course", ["All"] + courses, default=["All"])
    selected_bookies = st.multiselect("Filter by Bookie", ["All"] + bookies, default=["All"])

    if "All" not in selected_courses:
        df = df[df["Course"].isin(selected_courses)]
    if "All" not in selected_bookies:
        df = df[df["Bookie"].isin(selected_bookies)]

    view = st.radio("View Mode", ["Charts", "Tables"])
    df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    def color_val(v):
        if v > 20: return 'background-color:#58D68D;color:black'
        elif v > 10: return 'background-color:#F9E79F;color:black'
        else: return 'background-color:#F5B7B1;color:black'

    if view == "Tables":
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win.style.applymap(color_val, subset=["BetEdge Win %"]))
        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place.style.applymap(color_val, subset=["BetEdge Place %"]))
    else:
        st.subheader("📊 BetEdge Win % Chart (Top 20)")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("📊 BetEdge Place % Chart (Top 20)")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])

# ----------- FOOTBALL SECTION ----------- #
elif main_tab == "Football":
    st.title("⚽ Football – Mock Insights")
    leagues = ["Premier League", "Championship", "Serie A", "La Liga"]
    league = st.selectbox("Select League", leagues)
    teams = [f"Team {i}" for i in range(1, 21)]
    odds = np.random.uniform(1.5, 5, 20)
    edges = np.random.uniform(5, 25, 20)
    df_foot = pd.DataFrame({
        "Team": teams,
        "Best Odds": np.round(odds, 2),
        "Predicted Win %": np.round((1 / odds) * 100, 1),
        "Edge Score": np.round(edges, 1)
    })
    df_foot["Edge Rank"] = ((df_foot["Predicted Win %"] * 0.6) + (df_foot["Edge Score"] * 0.4)).round(1)
    df_foot = df_foot.sort_values("Edge Rank", ascending=False)

    def color_val(v):
        if v > 20: return 'background-color:#58D68D;color:black'
        elif v > 10: return 'background-color:#F9E79F;color:black'
        else: return 'background-color:#F5B7B1;color:black'

    st.subheader("📈 Top Edge Teams")
    st.dataframe(df_foot.style.applymap(color_val, subset=["Edge Rank"]))

# ----------- EDGEBRAIN AI MODULE ----------- #
elif main_tab == "EdgeBrain™":
    st.title("🧠 EdgeBrain – Bet Intelligence")
    st.markdown("""
    **EdgeBrain™** is our proprietary algorithm designed to surface the most valuable betting insights.
    
    - Combines real-time pricing, predicted probabilities, and market inefficiencies.
    - Currently operating in **mock mode** for development.
    - Will evolve to incorporate live data, historical results, and betting signals.

    🚀 *Coming soon: advanced predictive models powered by machine learning.*
    """)
    mock_examples = pd.DataFrame({
        "Selection": [f"Pick {i}" for i in range(1, 6)],
        "Sport": ["Horse Racing", "Football", "Horse Racing", "Football", "Horse Racing"],
        "Confidence %": np.round(np.random.uniform(70, 95, 5), 1),
        "Expected Value": np.round(np.random.uniform(5, 18, 5), 1)
    })
    st.subheader("🔍 Current Top Picks")
    st.dataframe(mock_examples.style.applymap(color_val, subset=["Confidence %"]))
