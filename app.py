import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ---- PAGE & THEME SETUP ----
st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- MOCK vs LIVE FLAG ----
# (for now we only have mock data, so flag=False)
LIVE_DATA = False

# ---- MOCK EDGE BRAIN + RACE TIMES ----
def generate_mock_data(n=40):
    now = datetime.now()
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot", "York", "Cheltenham", "Churchill Downs", "Santa Anita"], n
    )
    countries = ["UK" if c in ["Ascot", "York", "Cheltenham"] else "USA" for c in courses]
    bookies = np.random.choice(["SkyBet", "Bet365", "Betfair"], n)
    odds = np.random.uniform(2, 10, n).round(2)
    win_val = np.random.uniform(5, 30, n).round(1)
    # Random race times in next 12h
    times = [
        (now + timedelta(minutes=int(x))).strftime("%H:%M")
        for x in np.random.uniform(0, 12*60, n)
    ]

    df = pd.DataFrame({
        "Horse": horses,
        "Time": times,
        "Course": courses,
        "Country": countries,
        "Bookie": bookies,
        "Odds": odds,
        "Win_Value": win_val,
        "Place_Value": (win_val * 0.6).round(1)
    })

    df["Predicted Win %"] = (1 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = (
        df["Predicted Win %"] * 0.6 + df["Win_Value"] * 0.4
    ).round(1)
    df["BetEdge Place %"] = (
        df["Predicted Place %"] * 0.6 + df["Place_Value"] * 0.4
    ).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"] > 25, "✅",
        np.where(df["BetEdge Win %"] > 15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ---- MAIN SIDEBAR MENU ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview", "Horse Racing", "Football", "EdgeBrain", "How It Works"],
        icons=['house', 'activity', 'soccer', 'robot', 'book'],
        menu_icon="cast", default_index=0
    )

# ---- SHARED BADGE ----
badge = "🟢 Live data" if LIVE_DATA else "🔴 Mock data"
st.sidebar.markdown(f"**Data Source:** {badge}")

# ---- PAGES ----
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive value models.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners", len(df))
    st.metric("Top Edge Value", f"{df['BetEdge Win %'].max()}%")

elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    st.markdown(f"**Data Source:** {badge}")

    # --- Filters Row ---
    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.selectbox("Select Country", ["All", "UK", "USA"])
    with c2:
        bookie = st.selectbox("Select Bookmaker", ["All", "SkyBet", "Bet365", "Betfair"])
    with c3:
        courses = df["Course"].unique().tolist()
        course_filter = st.multiselect("Select Courses", courses, default=courses)

    # --- Value Slider ---
    min_val = int(df["BetEdge Win %"].min())
    max_val = int(df["BetEdge Win %"].max())
    edge_range = st.slider(
        "🎯 Filter by BetEdge Win %",
        min_val, max_val, (min_val, max_val)
    )

    # --- View Toggle ---
    view_mode = st.radio(
        "View Mode", ["📊 Charts", "📋 Tables"], horizontal=True
    )

    # --- Apply All Filters ---
    mask = (
        ((df["Country"] == country) | (country == "All")) &
        ((df["Bookie"] == bookie) | (bookie == "All")) &
        (df["Course"].isin(course_filter)) &
        (df["BetEdge Win %"].between(*edge_range))
    )
    filtered = df[mask]
    win_df = filtered.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    place_df = filtered.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if filtered.empty:
        st.warning("No horses match your filters.")
    else:
        if view_mode == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(win_df.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(place_df.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win_df, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(place_df, use_container_width=True)

elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.markdown(f"**Data Source:** {badge}")
    st.info("Mock data – coming soon with EdgeBrain integration.")

elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.markdown(f"**Data Source:** {badge}")
    st.write("AI-enhanced simulated predictions using EdgeBrain logic.")
    st.dataframe(df[["Horse", "Course", "Odds", "BetEdge Win %", "Risk"]], use_container_width=True)

elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet uses a smart hybrid model combining:

    - ✅ Implied Probability from Odds  
    - 📊 Win & Place Value Metrics  
    - 🧠 Proprietary EdgeBrain Simulation  
    - 🔎 Risk Indicators and Smart Filtering

    This generates a **BetEdge score** – your edge % over the market.
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # replace when your own intro is hosted

# ---- FOOTER ----
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
