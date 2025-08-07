import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu
import altair as alt

st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")

# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- MOCK EDGE BRAIN ----
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot", "York", "Cheltenham", "Churchill Downs", "Santa Anita"], n
    )
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" for c in courses]
    bookies = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds = np.random.uniform(2, 10, n).round(2)
    win_val = np.random.uniform(5, 30, n).round(1)

    df = pd.DataFrame({
        "Horse": horses,
        "Course": courses,
        "Country": countries,
        "Bookie": bookies,
        "Odds": odds,
        "Win_Value": win_val,
        "Place_Value": (win_val * 0.6).round(1)
    })
    df["Predicted Win %"]   = (1 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = (
        df["Predicted Win %"] * 0.6 + df["Win_Value"] * 0.4
    ).round(1)
    df["BetEdge Place %"]   = (
        df["Predicted Place %"] * 0.6 + df["Place_Value"] * 0.4
    ).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"] > 25, "✅",
        np.where(df["BetEdge Win %"] > 15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ---- MAIN MENU ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview", "Horse Racing", "Football", "EdgeBrain", "How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# ---- OVERVIEW PAGE ----
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive value models.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners", len(df))
    st.metric("Top Edge Value", f"{df['BetEdge Win %'].max()}%")

# ---- HORSE RACING PAGE ----
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Select Country", ["All","UK","USA"])
    with col2:
        bookie = st.selectbox("Select Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with col3:
        all_courses = df["Course"].unique().tolist()
        course_filter = st.multiselect("Select Courses", all_courses, default=all_courses)

    # Edge % slider
    min_val = int(df["BetEdge Win %"].min())
    max_val = int(df["BetEdge Win %"].max())
    edge_range = st.slider("🎯 Filter by BetEdge Win %", min_val, max_val, (min_val, max_val))

    # View toggle
    view_mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    # Apply filters
    filt = (
        ((df["Country"]==country)|(country=="All")) &
        ((df["Bookie"]==bookie)|(bookie=="All")) &
        (df["Course"].isin(course_filter)) &
        (df["BetEdge Win %"].between(*edge_range))
    )
    filtered_df = df[filt]
    df_win   = filtered_df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = filtered_df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if filtered_df.empty:
        st.warning("No horses match your filters.")
    else:
        if view_mode == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            win20 = df_win.head(20)
            win_chart = alt.Chart(win20).mark_bar().encode(
                x=alt.X("BetEdge Win %:Q"),
                y=alt.Y("Horse:N",
                        sort=alt.SortField("BetEdge Win %", order="descending"))
            ).properties(width=700, height=400)
            st.altair_chart(win_chart, use_container_width=True)

            st.subheader("Top 20 BetEdge Place %")
            place20 = df_place.head(20)
            place_chart = alt.Chart(place20).mark_bar().encode(
                x=alt.X("BetEdge Place %:Q"),
                y=alt.Y("Horse:N",
                        sort=alt.SortField("BetEdge Place %", order="descending"))
            ).properties(width=700, height=400)
            st.altair_chart(place_chart, use_container_width=True)

        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(df_win, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(df_place, use_container_width=True)

# ---- FOOTBALL PAGE ----
elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data – coming soon with EdgeBrain integration.")

# ---- EDGEBRAIN PAGE ----
elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.dataframe(df[["Horse","Course","Odds","BetEdge Win %","Risk"]], use_container_width=True)

# ---- HOW IT WORKS PAGE ----
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
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # Replace with your final video URL

# ---- FOOTER ----
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
