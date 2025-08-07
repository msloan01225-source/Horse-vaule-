import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu

# ─── PAGE CONFIG & THEME ───
st.set_page_config(page_title="EdgeBet – Phase 1", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
body { background-color: #111111; color: #EEE; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ─── MOCK EDGE BRAIN DATA ───
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"],
        n
    )
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" for c in courses]
    bookies = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds = np.random.uniform(2, 10, n).round(2)
    win_val = np.random.uniform(5, 30, n).round(1)

    df = pd.DataFrame({
        "Horse":        horses,
        "Course":       courses,
        "Country":      countries,
        "Bookie":       bookies,
        "Odds":         odds,
        "Win Value":    win_val,
        "Place Value":  (win_val * 0.6).round(1)
    })
    df["Predicted Win %"]   = (1/df["Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6) + (df["Win Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6) + (df["Place Value"]*0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"]>25, "✅",
        np.where(df["BetEdge Win %"]>15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ─── SIDEBAR MENU ───
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# ─── OVERVIEW ───
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive models.")
    st.metric("Active Sports", 1)
    st.metric("Total Runners", len(df))
    st.metric("Top Edge Value", f"{df['BetEdge Win %'].max()}%")

# ─── HORSE RACING ───
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with col2:
        bookie = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with col3:
        all_courses = df["Course"].unique().tolist()
        courses = st.multiselect("Courses", all_courses, default=all_courses)

    min_edge, max_edge = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
    edge_range = st.slider("BetEdge Win % range", min_edge, max_edge, (min_edge,max_edge))
    view_mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    filt = (
        ((df["Country"]==country)|(country=="All")) &
        ((df["Bookie"]==bookie)|(bookie=="All")) &
        (df["Course"].isin(courses)) &
        (df["BetEdge Win %"].between(*edge_range))
    )
    sub = df[filt]
    win = sub.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    plc = sub.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

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

# ─── FOOTBALL ───
elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data – coming soon with EdgeBrain integration.")

# ─── EDGEBRAIN ───
elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("AI-enhanced simulated predictions using EdgeBrain logic.")
    st.dataframe(df[["Horse","Course","Odds","BetEdge Win %","Risk"]], use_container_width=True)

# ─── HOW IT WORKS ───
elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet combines:
    - ✅ Implied Probability from Odds  
    - 📊 Win & Place Value Metrics  
    - 🧠 Proprietary EdgeBrain Simulation  
    - 🔎 Risk Indicators & Smart Filters  

    Result → your **BetEdge score**: % edge over the market.
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # placeholder

# ─── FOOTER ───
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
