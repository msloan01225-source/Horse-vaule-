import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import joblib
from datetime import datetime
from streamlit_option_menu import option_menu

st.set_page_config(page_title="EdgeBet", layout="wide")

# --- Load EdgeBrain model, with fallback stub ---
@st.cache_resource
def load_edgebrain_model():
    try:
        return joblib.load("edgebrain_model.pkl")
    except Exception:
        st.warning("⚠️ edgebrain_model.pkl not found — using stub model.")
        return None

edge_model = load_edgebrain_model()

def run_edgebrain_stub(df):
    # simply echoes BetEdge Win % if no model
    return df["BetEdge Win %"]

def predict_edgebrain(df):
    if edge_model:
        X = df[["Odds","Win_Value","Place_Value"]]
        return (edge_model.predict_proba(X)[:,1] * 100).round(1)
    else:
        return run_edgebrain_stub(df)

# --- Scrape RacingPost SP odds ---
@st.cache_data(ttl=300)
def fetch_live_rp(day="today"):
    date = datetime.utcnow().date()
    if day=="tomorrow":
        date += pd.Timedelta(days=1)
    url = f"https://www.racingpost.com/racecards/time-order/{date.isoformat()}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for race_section in soup.select("section[class*=race-time]"):
        t = race_section.select_one(".rc-race-time")
        c = race_section.select_one(".rc-race-meeting__course a")
        if not t or not c: continue
        off, course = t.text.strip(), c.text.strip()
        for rn in race_section.select(".runner-wrap"):
            h = rn.select_one(".runner-runner__name")
            sp = rn.select_one(".runner-sp__price")
            try:
                odds = float(sp.text.strip())
            except:
                continue
            rows.append({"Race_Time":off,"Course":course,"Horse":h.text.strip(),"Odds":odds})
    return pd.DataFrame(rows)

def generate_mock_data(n=40):
    horses  = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n)
    odds    = np.random.uniform(2,10,n).round(2)
    return pd.DataFrame({"Horse":horses,"Course":courses,"Odds":odds})

def build_live_df(day="today"):
    df = fetch_live_rp(day)
    if df.empty:
        df = generate_mock_data()
    df["Win_Value"]    = np.random.uniform(5,30,len(df)).round(1)
    df["Place_Value"]  = (df["Win_Value"]*0.6).round(1)
    df["Pred Win %"]   = (1/df["Odds"]*100).round(1)
    df["Pred Place %"] = (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"]= ((df["Pred Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Pl %"] = ((df["Pred Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"]         = np.where(df["BetEdge Win %"]>25,"✅",
                           np.where(df["BetEdge Win %"]>15,"⚠️","❌"))
    df["EdgeBrain %"]  = predict_edgebrain(df)
    return df

# --- Sidebar navigation ---
with st.sidebar:
    page = option_menu("EdgeBet", ["Overview","Horse Racing","EdgeBrain","How It Works"],
                       icons=["house","card-list","robot","info-circle"])

# --- Pages ---
if page=="Overview":
    st.title("📊 EdgeBet Live Tracker")
    df0 = build_live_df()
    st.metric("Races Found", df0["Course"].nunique())
    st.metric("Total Runners", len(df0))
    st.metric("Top Live Value", f"{df0['BetEdge Win %'].max()}%")

elif page=="Horse Racing":
    day = st.selectbox("Day",["today","tomorrow"])
    df = build_live_df(day)
    df["Country"] = df["Course"].apply(lambda c: "UK" if c in ["Ascot","York","Cheltenham"] else "USA")
    country = st.selectbox("Country", ["All","UK","USA"])
    df = df[(df["Country"]==country)| (country=="All")]
    st.subheader("Top 20 Live Value")
    st.bar_chart(df.sort_values("BetEdge Win %",ascending=False).head(20)
                     .set_index("Horse")["BetEdge Win %"])
    st.dataframe(df, use_container_width=True)

elif page=="EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    df = build_live_df()
    st.bar_chart(df.sort_values("EdgeBrain %",ascending=False).head(20)
                     .set_index("Horse")["EdgeBrain %"])
    st.dataframe(df[["Horse","Course","Odds","EdgeBrain %","Risk"]], use_container_width=True)

else:
    st.title("ℹ️ How EdgeBet Works")
    st.markdown("""
- Scrape live SP odds from Racing Post  
- Compute live “BetEdge” value metrics  
- Run the **EdgeBrain** ML model (or stub)  
- Filter and chart your top picks in real-time  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")
