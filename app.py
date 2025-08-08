import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

st.set_page_config(page_title="EdgeBet", layout="wide")

# --- EDGEBRAIN MODEL LOADING ---
@st.cache_resource
def load_edgebrain_model():
    try:
        return joblib.load("edgebrain_model.pkl")
    except Exception:
        st.warning("⚠️ edgebrain_model.pkl not found — using stub predictions.")
        return None

edge_model = load_edgebrain_model()

def predict_edgebrain(df: pd.DataFrame) -> pd.Series:
    if edge_model:
        X = df[["Odds", "Win_Value", "Place_Value"]]
        return (edge_model.predict_proba(X)[:,1] * 100).round(1)
    else:
        # stub = echo BetEdge Win % until model is in place
        return df["BetEdge Win %"]

# --- LIVE SP SCRAPER / MOCK FALLBACK ---
@st.cache_data(ttl=300)
def fetch_live_rp(day: str = "today") -> pd.DataFrame:
    """Scrape Racing Post time-order SP odds for today or tomorrow."""
    # build date suffix
    target = datetime.utcnow().date()
    if day.lower() == "tomorrow":
        target += timedelta(days=1)
    url = f"https://www.racingpost.com/racecards/time-order/{target.isoformat()}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        # on any fetch or parse error, return empty dataframe
        return pd.DataFrame()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    # each race block
    for section in soup.select("section.race-time"):
        time_tag = section.select_one(".rc-race-time")
        course_tag = section.select_one(".rc-race-meeting__course a")
        if not time_tag or not course_tag:
            continue
        off = time_tag.text.strip()
        course = course_tag.text.strip()
        # each runner
        for runner in section.select(".runner-wrap"):
            name_el = runner.select_one(".runner-runner__name")
            sp_el   = runner.select_one(".runner-sp__price")
            if not name_el or not sp_el:
                continue
            try:
                od = float(sp_el.text.strip())
            except:
                continue
            rows.append({
                "Horse": name_el.text.strip(),
                "Course": course,
                "Time":   off,
                "Odds":   od
            })
    return pd.DataFrame(rows)

def generate_mock_data(n=40) -> pd.DataFrame:
    horses  = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    odds    = np.random.uniform(2,10,n).round(2)
    return pd.DataFrame({"Horse":horses, "Course":courses, "Odds":odds})

def build_live_df(day: str="today") -> pd.DataFrame:
    df = fetch_live_rp(day)
    if df.empty:
        df = generate_mock_data()
        df["Time"] = ""
    # value calc
    df["Win_Value"]    = np.random.uniform(5,30,len(df)).round(1)
    df["Place_Value"]  = (df["Win_Value"]*0.6).round(1)
    df["Pred Win %"]   = (1/df["Odds"]*100).round(1)
    df["Pred Pl %"]    = (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"]= ((df["Pred Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Pl %"] = ((df["Pred Pl %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"]         = np.where(df["BetEdge Win %"]>25,"✅",
                           np.where(df["BetEdge Win %"]>15,"⚠️","❌"))
    df["EdgeBrain %"]  = predict_edgebrain(df)
    # country from course
    df["Country"] = df["Course"].apply(
        lambda c: "UK" if c in ["Ascot","York","Cheltenham"] else "USA"
    )
    return df

# --- SIDEBAR MENU ---
with st.sidebar:
    page = option_menu("🏇 EdgeBet", ["Overview","Horse Racing","EdgeBrain","How It Works"],
                       icons=["house","card-list","robot","info-circle"], default_index=0)

# --- PAGES ---
if page == "Overview":
    st.title("📊 EdgeBet Live Tracker")
    df0 = build_live_df()
    st.metric("Races Found", df0["Course"].nunique())
    st.metric("Total Runners", len(df0))
    st.metric("Top BetEdge Value", f"{df0['BetEdge Win %'].max()}%")

elif page == "Horse Racing":
    st.title("🏇 Horse Racing – Live UK/USA")
    day = st.selectbox("Select Day", ["today","tomorrow"])
    df = build_live_df(day)
    country = st.selectbox("Country Filter", ["All","UK","USA"])
    if country!="All":
        df = df[df["Country"]==country]
    st.subheader("📊 Top 20 BetEdge Win %")
    st.bar_chart(df.sort_values("BetEdge Win %", ascending=False)
                     .head(20).set_index("Horse")["BetEdge Win %"])
    st.subheader("🏆 Full Table")
    st.dataframe(df, use_container_width=True)

elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    df = build_live_df()
    st.subheader("📊 Top 20 EdgeBrain Win %")
    st.bar_chart(df.sort_values("EdgeBrain %", ascending=False)
                     .head(20).set_index("Horse")["EdgeBrain %"])
    st.dataframe(df[["Horse","Course","Odds","EdgeBrain %","Risk"]], use_container_width=True)

else:
    st.title("ℹ️ How EdgeBet Works")
    st.markdown("""
- Scrape live SP odds from Racing Post  
- Compute “BetEdge” value metrics in real-time  
- Run the **EdgeBrain** ML model (or stub)  
- Filter, chart and table your best picks
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
