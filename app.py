import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu

# ---------------------------------------------
# 1) Clear any DataFrame/Series in session_state
# ---------------------------------------------
for _key in list(st.session_state.keys()):
    val = st.session_state[_key]
    if isinstance(val, (pd.DataFrame, pd.Series)):
        del st.session_state[_key]

# ---------------------------------------------
# 2) Page config & dark theme styling
# ---------------------------------------------
st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 3) Mock EdgeBrain data generator
# ---------------------------------------------
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
    df["Predicted Win %"]   = (1/df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"]              = np.where(
        df["BetEdge Win %"]>25, "✅",
        np.where(df["BetEdge Win %"]>15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ---------------------------------------------
# 4) Main sidebar navigation
# ---------------------------------------------
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast",
        default_index=0
    )

# ---------------------------------------------
# 5) Pages
# ---------------------------------------------
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive value models.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners", len(df))
    st.metric("Top Edge Value", f"{df['BetEdge Win %'].max()}%")

elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with c2:
        bookie = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with c3:
        courses = df["Course"].unique().tolist()
        course_filter = st.multiselect("Courses", courses, default=courses)

    # Edge % slider
    lo, hi = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
    edge_range = st.slider("🎯 BetEdge Win % Range", lo, hi, (lo,hi))

    # View toggle
    view_mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    # Apply
    filt = (
        ((df["Country"]==country)|(country=="All")) &
        ((df["Bookie"]==bookie)|(bookie=="All")) &
        df["Course"].isin(course_filter) &
        df["BetEdge Win %"].between(*edge_range)
    )
    sub = df[filt].reset_index(drop=True)
    win = sub.sort_values("BetEdge Win %", ascending=False)
    plc = sub.sort_values("BetEdge Place %", ascending=False)

    if sub.empty:
        st.warning("No horses match your filters.")
    else:
        if view_mode=="📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(win.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(plc.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(plc, use_container_width=True)

elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data – coming soon with EdgeBrain integration.")

elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("AI-enhanced simulated predictions using EdgeBrain logic.")
    st.dataframe(df[["Horse","Course","Odds","BetEdge Win %","Risk"]], use_container_width=True)

elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet uses a smart hybrid model combining:

    - ✅ Implied Probability from Odds  
    - 📊 Win & Place Value Metrics  
    - 🧠 Proprietary EdgeBrain Simulation  
    - 🔎 Risk Indicators and Smart Filtering
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # your explainer

# ---------------------------------------------
# 6) Footer
# ---------------------------------------------
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
