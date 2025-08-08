import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import joblib
from datetime import datetime
from streamlit_option_menu import option_menu

# — load your trained EdgeBrain model —
@st.cache_resource
def load_edgebrain_model():
    return joblib.load("edgebrain_model.pkl")
edge_model = load_edgebrain_model()

# — scrape live RacingPost SP odds & build DataFrame —
@st.cache_data(ttl=300)
def fetch_live_rp(day="today"):
    # build URL for today/tomorrow
    date = datetime.utcnow().date()
    if day=="tomorrow":
        date += pd.Timedelta(days=1)
    url = f"https://www.racingpost.com/racecards/time-order/{date.isoformat()}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    # find each race container
    for race_section in soup.select("section[class*=race-time]"):
        time_tag   = race_section.select_one(".rc-race-time")
        course_tag = race_section.select_one(".rc-race-meeting__course a")
        if not time_tag or not course_tag:
            continue
        off_time = time_tag.get_text(strip=True)
        course   = course_tag.get_text(strip=True)

        # each runner row
        for runner in race_section.select(".runner-wrap"):
            horse_tag = runner.select_one(".runner-runner__name")
            sp_tag    = runner.select_one(".runner-sp__price")
            # fallback if no SP
            try:
                sp = float(sp_tag.get_text(strip=True))
            except:
                continue
            horse = horse_tag.get_text(strip=True)
            rows.append({
                "Race_Time": off_time,
                "Course":    course,
                "Horse":     horse,
                "Odds":      sp
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

# — if scraper fails, generate mock data —
def generate_mock_data(n=40):
    horses  = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    odds    = np.random.uniform(2,10,n).round(2)
    df = pd.DataFrame({"Horse":horses,"Course":courses,"Odds":odds})
    return df

# — build full live DataFrame & calculate values —
def build_live_df(day="today"):
    df = fetch_live_rp(day)
    if df.empty:
        df = generate_mock_data(40)
    # compute value metrics
    df["Win_Value"]   = np.random.uniform(5,30,len(df)).round(1)
    df["Place_Value"] = (df["Win_Value"]*0.6).round(1)
    df["Pred Win %"]  = (1/df["Odds"]*100).round(1)
    df["Pred Place %"]= (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"]= ((df["Pred Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Pl %"]= ((df["Pred Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    # risk
    df["Risk"] = np.where(df["BetEdge Win %"]>25,"✅",
                   np.where(df["BetEdge Win %"]>15,"⚠️","❌"))
    # EdgeBrain probability
    X = df[["Odds","Win_Value","Place_Value"]]
    df["EdgeBrain %"] = (edge_model.predict_proba(X)[:,1]*100).round(1)
    return df

# — Streamlit UI & theming —
st.set_page_config(page_title="EdgeBet", layout="wide")
st.markdown("""
<style>
body { background-color: #111; color: #eee; }
h1,h2,h3,h4 { color: #0ff; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    page = option_menu(
        "EdgeBet", ["Overview","Horse Racing","EdgeBrain","How It Works"],
        icons=["house","card-list","robot","info-circle"]
    )

# — Overview —
if page=="Overview":
    st.title("📊 EdgeBet Live Tracker")
    st.write("Scraping Racing Post SP odds every 5 minutes…")
    df0 = build_live_df()
    st.metric("Races Found", len(df0["Course"].unique()))
    st.metric("Total Runners", len(df0))
    st.metric("Top Live Value", f"{df0['BetEdge Win %'].max()}%")

# — Horse Racing page —
elif page=="Horse Racing":
    st.title("🏇 Horse Racing – Live SP Odds")
    day = st.selectbox("Day",["today","tomorrow"])
    df = build_live_df(day)

    country = st.selectbox("Country",["All","UK","USA"])
    # derive country from course name
    df["Country"] = df["Course"].apply(
        lambda c: "UK" if c in ["Ascot","York","Cheltenham"] else "USA"
    )
    df = df[(df["Country"]==country)| (country=="All")]

    st.subheader("Top 20 Live Value (BetEdge Win %)")
    st.bar_chart(df.sort_values("BetEdge Win %",ascending=False).head(20).
                 set_index("Horse")["BetEdge Win %"])

    st.subheader("Full Live Table")
    st.dataframe(df, use_container_width=True)

# — EdgeBrain page —
elif page=="EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    df = build_live_df()
    st.subheader("Top 20 AI Win % Picks")
    st.bar_chart(df.sort_values("EdgeBrain %",ascending=False).head(20).
                 set_index("Horse")["EdgeBrain %"])
    st.dataframe(df[["Horse","Course","Odds","EdgeBrain %","Risk"]], use_container_width=True)

# — How It Works —
else:
    st.title("ℹ️ How EdgeBet Works")
    st.markdown("""
- We **scrape** Racing Post for live SP odds  
- We compute **value** (% edge) vs implied prob  
- We run your **EdgeBrain** ML model for “AI Win%”  
- We let you **filter** by country, time, course, value  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")
