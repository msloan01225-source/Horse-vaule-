import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="BetEdge Value App", layout="wide")

# ---------- MOCK DATA ----------
def generate_mock_horse_data():
    np.random.seed(42)
    horses = [f"Horse {i}" for i in range(1, 51)]
    courses = ["Ascot", "Newbury", "Cheltenham", "York", "Epsom"]
    bookies = ["Bet365", "SkyBet", "PaddyPower", "Betfair", "William Hill"]

    data = []
    for horse in horses:
        data.append({
            "Horse": horse,
            "Course": np.random.choice(courses),
            "Bookie": np.random.choice(bookies),
            "Best Odds": round(np.random.uniform(2.0, 8.0), 2),
            "Win_Value": round(np.random.uniform(5, 25), 1)
        })
    df = pd.DataFrame(data)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["BetEdge %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    return df

def generate_mock_football_data():
    teams = ["Arsenal", "Chelsea", "Man City", "Liverpool", "Spurs", "Villa", "Newcastle", "Brighton"]
    fixtures = []
    for _ in range(20):
        home, away = np.random.choice(teams, 2, replace=False)
        value = round(np.random.uniform(5, 20), 1)
        fixtures.append({
            "Home": home,
            "Away": away,
            "Bookie": np.random.choice(["Bet365", "SkyBet", "PaddyPower"]),
            "Value Score": value,
            "Kickoff": datetime.now().strftime("%H:%M")
        })
    return pd.DataFrame(fixtures)

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.image("https://i.imgur.com/lYXZ5jJ.png", width=180)
page = st.sidebar.radio("Navigate", ["🏇 Horse Racing", "⚽ Football", "📈 Stats", "ℹ️ About"])

# ---------- HORSE RACING ----------
if page == "🏇 Horse Racing":
    st.title("🏇 BetEdge – Horse Racing")
    df = generate_mock_horse_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_course = st.selectbox("Course", ["All"] + sorted(df["Course"].unique().tolist()))
    with col2:
        selected_bookie = st.selectbox("Bookie", ["All"] + sorted(df["Bookie"].unique().tolist()))
    with col3:
        min_value = st.slider("Minimum BetEdge %", 0, 50, 10)

    # Apply filters
    filtered_df = df[df["BetEdge %"] >= min_value]
    if selected_course != "All":
        filtered_df = filtered_df[filtered_df["Course"] == selected_course]
    if selected_bookie != "All":
        filtered_df = filtered_df[filtered_df["Bookie"] == selected_bookie]

    st.subheader("📊 Top Horses by BetEdge %")
    st.dataframe(
        filtered_df.sort_values("BetEdge %", ascending=False).reset_index(drop=True),
        use_container_width=True
    )

    st.subheader("📈 Chart: BetEdge % (Top 20)")
    st.bar_chart(
        filtered_df.sort_values("BetEdge %", ascending=False).head(20).set_index("Horse")["BetEdge %"]
    )

# ---------- FOOTBALL ----------
elif page == "⚽ Football":
    st.title("⚽ BetEdge – Football")
    df = generate_mock_football_data()

    st.subheader("🔥 Top Value Fixtures")
    st.dataframe(
        df.sort_values("Value Score", ascending=False).reset_index(drop=True),
        use_container_width=True
    )

    st.subheader("📊 Value Score Chart")
    st.bar_chart(df.sort_values("Value Score", ascending=False).head(20).set_index("Home")["Value Score"])

# ---------- STATS ----------
elif page == "📈 Stats":
    st.title("📈 Performance Overview")
    col1, col2 = st.columns(2)
    col1.metric("✅ Win Accuracy", "72%", "↑ 4%")
    col2.metric("💰 Avg ROI", "14.2%", "↑ 1.7%")

    st.subheader("🧮 Summary")
    st.markdown("""
    - Total Races Tracked: **210**
    - Bets Placed: **153**
    - Profitable Days: **18 of last 30**
    - Hit Rate on Top 3 Picks: **78%**
    """)

# ---------- ABOUT ----------
elif page == "ℹ️ About":
    st.title("ℹ️ About BetEdge")
    st.markdown("""
    **BetEdge** is a performance-driven racing and sports value tracker that helps identify edge opportunities.

    This platform is built to simulate bookmaker inefficiencies using predicted win probabilities and mock value scores.

    **Key Features:**
    - Horse Racing & Football Value Picks
    - Custom Filters & Analytics
    - Easy-to-read Tables and Charts

    _Built with ❤️ using Streamlit_
    """)
