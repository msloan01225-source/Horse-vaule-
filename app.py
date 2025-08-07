import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ----------- Config ----------- #
st.set_page_config(page_title="EdgeBet – Bet Smarter", layout="wide", initial_sidebar_state="expanded")

# ----------- Styles / Branding ----------- #
st.markdown("""
    <style>
        body { color: white; background-color: #0E1117; }
        .stApp { background-color: #0E1117; }
        .title { font-size: 36px; font-weight: 700; color: #FAFAFA; }
        .subtitle { font-size: 24px; font-weight: 400; color: #AAAAAA; }
        .section-header { font-size: 22px; margin-top: 20px; color: #F3F3F3; }
        .metric-box { background-color: #1E1E1E; padding: 12px; border-radius: 8px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ----------- Sidebar Navigation ----------- #
menu = st.sidebar.selectbox("🔎 Navigate", [
    "🏠 Home", "🏇 Horse Racing", "⚽ Football", "🧠 EdgeBrain", "📊 Results & ROI", "ℹ️ About EdgeBet"
])

# ----------- Mock Data Generator ----------- #
def generate_mock_racing_data(region):
    courses = ["Ascot", "Newmarket", "Aintree", "York", "Churchill Downs", "Santa Anita"] if region == "USA" else \
              ["Ascot", "Newbury", "Doncaster", "Cheltenham"]
    bookies = ["Bet365", "SkyBet", "William Hill", "Paddy Power"]
    data = []
    for _ in range(60):
        course = np.random.choice(courses)
        bookie = np.random.choice(bookies)
        horse = f"Horse {np.random.randint(1, 100)}"
        odds = np.round(np.random.uniform(2, 12), 2)
        win_val = np.round(np.random.uniform(5, 25), 1)
        place_val = np.round(win_val * 0.6, 1)
        predicted_win = round(100 / odds, 1)
        data.append({
            "Horse": horse,
            "Course": course,
            "Bookie": bookie,
            "Odds": odds,
            "Win Value %": win_val,
            "Place Value %": place_val,
            "Predicted Win %": predicted_win,
            "BetEdge Win %": round(predicted_win * 0.6 + win_val * 0.4, 1),
            "BetEdge Place %": round((predicted_win * 0.6 * 0.6) + (place_val * 0.4), 1)
        })
    return pd.DataFrame(data)

# ----------- Home Page ----------- #
if menu == "🏠 Home":
    st.markdown("<div class='title'>Welcome to EdgeBet</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Your all-in-one smart betting dashboard</div>", unsafe_allow_html=True)
    st.video("https://www.w3schools.com/html/mov_bbb.mp4")
    st.markdown("#### 🔎 Navigate through the sidebar to get started with Horse Racing, Football, EdgeBrain, and Results.")

# ----------- Horse Racing ----------- #
elif menu == "🏇 Horse Racing":
    st.markdown("## 🏇 Horse Racing Value Tracker")

    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region", ["UK", "USA"])
    with col2:
        min_val = st.slider("Minimum BetEdge Win %", 0, 100, 10)

    df = generate_mock_racing_data(region)
    selected_courses = st.multiselect("Filter by Course", options=sorted(df["Course"].unique()), default=sorted(df["Course"].unique()))
    selected_bookies = st.multiselect("Filter by Bookie", options=sorted(df["Bookie"].unique()), default=sorted(df["Bookie"].unique()))

    df_filtered = df[
        (df["Course"].isin(selected_courses)) &
        (df["Bookie"].isin(selected_bookies)) &
        (df["BetEdge Win %"] >= min_val)
    ]

    tab1, tab2 = st.tabs(["📈 Chart View", "📋 Table View"])

    with tab1:
        st.subheader("Top 20 by BetEdge Win %")
        st.bar_chart(df_filtered.sort_values("BetEdge Win %", ascending=False).head(20).set_index("Horse")["BetEdge Win %"])

        st.subheader("Top 20 by BetEdge Place %")
        st.bar_chart(df_filtered.sort_values("BetEdge Place %", ascending=False).head(20).set_index("Horse")["BetEdge Place %"])

    with tab2:
        st.dataframe(df_filtered.sort_values("BetEdge Win %", ascending=False), use_container_width=True)

# ----------- Football ----------- #
elif menu == "⚽ Football":
    st.markdown("## ⚽ Football Value Tracker (Mock Data)")
    st.markdown("Coming Soon: League filters, bookie odds comparisons, match models.")

    df = pd.DataFrame({
        "Match": ["Team A vs Team B", "Team C vs Team D", "Team E vs Team F"],
        "Market": ["Over 2.5 Goals", "Match Winner", "BTTS"],
        "Bookie": ["Bet365", "SkyBet", "William Hill"],
        "Odds": [1.85, 2.2, 1.95],
        "Model %": [58, 52, 61],
        "Edge %": [8, 4, 11]
    })
    st.dataframe(df)

# ----------- EdgeBrain Intelligence ----------- #
elif menu == "🧠 EdgeBrain":
    st.markdown("## 🧠 EdgeBrain AI Insights")
    st.markdown("EdgeBrain is your machine-learning powered assistant for spotting value.")
    st.markdown("⚙️ This will evolve into real predictive models soon.")
    mock_picks = df.sort_values("BetEdge Win %", ascending=False).head(5)
    for i, row in mock_picks.iterrows():
        st.metric(label=f"{row['Horse']} @ {row['Course']} ({row['Bookie']})", value=f"{row['BetEdge Win %']}% Edge")

# ----------- Results & ROI ----------- #
elif menu == "📊 Results & ROI":
    st.markdown("## 📊 Historical Picks & ROI (Mocked)")
    st.line_chart(np.cumsum(np.random.normal(1.5, 2, 60)))  # Mock ROI chart
    st.dataframe(df.sample(10), use_container_width=True)

# ----------- About ----------- #
elif menu == "ℹ️ About EdgeBet":
    st.markdown("## ℹ️ About EdgeBet")
    st.markdown("""
        **EdgeBet** combines data science, value betting, and modern UX to create a powerful betting tool.
        - 🎯 Smart odds & model overlay
        - 🤖 AI-powered 'EdgeBrain'
        - 🇬🇧 UK + 🇺🇸 US racing (football expanding)
        - 📊 ROI tracking + historical picks
    """)
