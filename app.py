import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ---- PAGE CONFIG & THEME ----
st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- LIVE UK RACING SCRAPER ----
@st.cache_data(ttl=300)
def fetch_live_data(day="today"):
    """Scrape Racing Post time-order page for races today/tomorrow."""
    url = "https://www.racingpost.com/racecards/time-order/"
    if day == "tomorrow":
        tgt = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        url += f"?date={tgt}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        # find all time tags like "14:30"
        for t in soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$")):
            race_time = t.strip()
            course = t.find_next(string=True).strip() if t.find_next(string=True) else "Unknown"
            # the next <a> after the time is the race name; we'll treat that as our "horse" slot for live
            race_name = t.find_next("a").get_text(strip=True) if t.find_next("a") else "Race"
            rows.append({
                "Horse": race_name,
                "Course": course,
                "Time": race_time
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ---- MOCK EDGEBRAIN DATA ----
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" for c in courses]
    bookies = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds = np.random.uniform(2,10,n).round(2)
    win_val = np.random.uniform(5,30,n).round(1)
    df = pd.DataFrame({
        "Horse": horses,
        "Course": courses,
        "Country": countries,
        "Bookie": bookies,
        "Odds": odds,
        "Win_Value": win_val,
        "Place_Value": (win_val*0.6).round(1),
    })
    df["Predicted Win %"]   = (1/df["Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"]>25, "✅",
        np.where(df["BetEdge Win %"]>15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ---- BUILD FULL HORSE-RACING DATAFRAME (LIVE → FALLBACK MOCK) ----
def get_horse_df(day="today"):
    live = fetch_live_data(day)
    if not live.empty:
        # attach mock metrics onto live scraped rows
        live["Country"]    = "UK"
        live["Bookie"]     = "Live"
        live["Odds"]       = np.random.uniform(2,6,len(live)).round(2)
        live["Win_Value"]  = np.random.uniform(5,25,len(live)).round(1)
        live["Place_Value"]= (live["Win_Value"]*0.6).round(1)
        live["Predicted Win %"]   = (1/live["Odds"]*100).round(1)
        live["Predicted Place %"] = (live["Predicted Win %"]*0.6).round(1)
        live["BetEdge Win %"]     = ((live["Predicted Win %"]*0.6)+(live["Win_Value"]*0.4)).round(1)
        live["BetEdge Place %"]   = ((live["Predicted Place %"]*0.6)+(live["Place_Value"]*0.4)).round(1)
        live["Risk"] = np.where(
            live["BetEdge Win %"]>25, "✅",
            np.where(live["BetEdge Win %"]>15, "⚠️", "❌")
        )
        return live.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    else:
        return generate_mock_data(40)

# ---- NAVIGATION ----
with st.sidebar:
    page = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=["house","activity","soccer","robot","book"],
        menu_icon="cast", default_index=0
    )

# ---- PAGES ----
if page == "Overview":
    df0 = generate_mock_data(20)
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive value models.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners", len(df0))
    st.metric("Top Edge Value", f"{df0['BetEdge Win %'].max()}%")

elif page == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    day_opt = st.radio("Day", ["today","tomorrow"], horizontal=True)
    df = get_horse_df(day_opt)

    # filters
    c1,c2,c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", ["All"]+sorted(df["Country"].unique().tolist()))
    with c2:
        bookie  = st.selectbox("Bookmaker", ["All"]+sorted(df["Bookie"].unique().tolist()))
    with c3:
        courses = st.multiselect(
            "Courses", ["All"]+sorted(df["Course"].unique().tolist()),
            default=["All"]
        )

    # BetEdge slider
    lo, hi = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
    edge_lo, edge_hi = st.slider("BetEdge Win % Range", lo, hi, (lo,hi))

    # apply filters
    filt = df.copy()
    if country!="All": filt = filt[filt["Country"]==country]
    if bookie!="All":  filt = filt[filt["Bookie"] ==bookie]
    if "All" not in courses: filt = filt[filt["Course"].isin(courses)]
    filt = filt[filt["BetEdge Win %"].between(edge_lo,edge_hi)]
    df_win   = filt.sort_values("BetEdge Win %", ascending=False)
    df_place = filt.sort_values("BetEdge Place %", ascending=False)

    view = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)
    if filt.empty:
        st.warning("No runners match your filters.")
    elif view.startswith("📊"):
        st.subheader("Top 20 BetEdge Win %")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("Top 20 BetEdge Place %")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
    else:
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win, use_container_width=True)
        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place, use_container_width=True)

elif page == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data – coming soon with EdgeBrain integration.")

elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("AI-enhanced simulated predictions using EdgeBrain logic.")
    st.dataframe(
        get_horse_df("today")[["Horse","Course","Odds","BetEdge Win %","Risk"]],
        use_container_width=True
    )

else:  # How It Works
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet uses a smart hybrid model combining:

    - ✅ Implied Probability from Odds  
    - 📊 Win & Place Value Metrics  
    - 🧠 Proprietary EdgeBrain Simulation  
    - 🔎 Risk Indicators and Smart Filtering

    This generates a **BetEdge score** – your % edge over the market.
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # placeholder

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
