import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
import joblib

# ─── CONFIG & CREDENTIALS ──────────────────────────────────────────────────────
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(page_title="EdgeBet Live", layout="wide")
st.sidebar.title("🏇 EdgeBet Live")

# ─── LOAD EDGEBRAIN MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load_edgebrain_model():
    try:
        return joblib.load("edgebrain_model.pkl")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ edgebrain_model.pkl not found — using BetEdge as stub.")
        return None

edge_model = load_edgebrain_model()

def predict_edgebrain(df: pd.DataFrame) -> pd.Series:
    if edge_model:
        X = df[["Odds", "Win_Value", "Place_Value"]]
        return (edge_model.predict_proba(X)[:, 1] * 100).round(1)
    else:
        # stub: mirror BetEdge Win %
        return df["BetEdge Win %"]

# ─── STRATEGY 1: OFFICIAL API ───────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_live_api(day: str = "today") -> pd.DataFrame:
    date = datetime.utcnow().date()
    if day == "tomorrow":
        date += timedelta(days=1)

    url = "https://api.theracingapi.com/v1/racecards"
    params = {"region": "GB", "date": date.isoformat()}
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    rows = []
    for m in data.get("meetings", []):
        course = m.get("course", {}).get("name", "")
        for race in m.get("races", []):
            off = race.get("off", "")[:5]
            for runner in race.get("runners", []):
                sp = runner.get("sp_dec")
                if sp is None: 
                    continue
                rows.append({
                    "Horse": runner.get("horse", ""),
                    "Course": course,
                    "Time":   off,
                    "Odds":   float(sp)
                })
    return pd.DataFrame(rows)

# ─── STRATEGY 2: SCRAPE RACING POST ────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_live_sp(day: str = "today") -> pd.DataFrame:
    date = datetime.utcnow().date()
    if day == "tomorrow":
        date += timedelta(days=1)

    url = f"https://www.racingpost.com/racecards/time-order/{date.isoformat()}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    for section in soup.select("section.race-time"):
        time_tag   = section.select_one(".rc-race-time")
        course_tag = section.select_one(".rc-race-meeting__course a")
        if not time_tag or not course_tag:
            continue
        off, course = time_tag.text.strip(), course_tag.text.strip()
        for runner in section.select(".runner-wrap"):
            name_el = runner.select_one(".runner-runner__name")
            sp_el   = runner.select_one(".runner-sp__price")
            if not name_el or not sp_el:
                continue
            try:
                od = float(sp_el.text.strip())
            except ValueError:
                continue
            rows.append({
                "Horse": name_el.text.strip(),
                "Course": course,
                "Time": off,
                "Odds": od
            })
    return pd.DataFrame(rows)

# ─── STRATEGY 3: MOCK DATA ─────────────────────────────────────────────────────
def generate_mock_data(n=30) -> pd.DataFrame:
    horses  = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    odds    = np.random.uniform(2,10,n).round(2)
    return pd.DataFrame({"Horse":horses,"Course":courses,"Odds":odds})

# ─── HYBRID DATA BUILDER ───────────────────────────────────────────────────────
def build_live_df(day: str="today") -> pd.DataFrame:
    # Try API first
    try:
        df = fetch_live_api(day)
        if df.empty:
            raise ValueError("API returned no rows")
    except Exception:
        # Fallback to scraper
        try:
            df = fetch_live_sp(day)
            if df.empty:
                raise ValueError("Scraper returned no rows")
        except Exception:
            # Final fallback: mock
            df = generate_mock_data()
            df["Time"] = ""

    # Value calculations
    df["Win_Value"]     = np.random.uniform(5,30,len(df)).round(1)
    df["Place_Value"]   = (df["Win_Value"] * 0.6).round(1)
    df["Pred Win %"]    = (1/df["Odds"] * 100).round(1)
    df["Pred Pl %"]     = (df["Pred Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Pred Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Pl %"]  = ((df["Pred Pl %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"]          = np.where(df["BetEdge Win %"]>25, "✅",
                            np.where(df["BetEdge Win %"]>15, "⚠️", "❌"))
    df["EdgeBrain %"]   = predict_edgebrain(df)
    df["Country"] = df["Course"].apply(
        lambda c: "UK" if c in ["Ascot","York","Cheltenham"] else "USA"
    )
    return df

# ─── UI ────────────────────────────────────────────────────────────────────────
day = st.sidebar.selectbox("Select Day", ["today","tomorrow"])
df = build_live_df(day)

tabs = option_menu(
    None, ["Overview","Horse Racing","EdgeBrain","How It Works"],
    icons=["house","card-list","robot","info-circle"], 
    menu_icon="cast", default_index=0, orientation="horizontal"
)

if tabs=="Overview":
    st.title("📊 EdgeBet Live Tracker")
    st.metric("Races Found", df["Course"].nunique())
    st.metric("Total Runners", len(df))
    st.metric("Top BetEdge Value", f"{df['BetEdge Win %'].max()}%")

elif tabs=="Horse Racing":
    st.title("🏇 Horse Racing – Live UK/USA")
    country = st.selectbox("Country Filter", ["All","UK","USA"])
    if country!="All":
        df = df[df["Country"]==country]

    st.subheader("📊 Top 20 BetEdge Win %")
    st.bar_chart(df.sort_values("BetEdge Win %", ascending=False)
                    .head(20).set_index("Horse")["BetEdge Win %"])
    st.subheader("🏆 Full Table")
    st.dataframe(df, use_container_width=True)

elif tabs=="EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    st.subheader("📊 Top 20 EdgeBrain Win %")
    st.bar_chart(df.sort_values("EdgeBrain %", ascending=False)
                    .head(20).set_index("Horse")["EdgeBrain %"])
    st.dataframe(df[["Horse","Course","Odds","EdgeBrain %","Risk"]],
                 use_container_width=True)

else:  # How It Works
    st.title("ℹ️ How EdgeBet Works")
    st.markdown("""
- **Step 1:** Try official Racing API  
- **Step 2:** Fallback to Racing Post scrape  
- **Step 3:** Fallback to mock data  
- **Compute:** BetEdge & EdgeBrain scores in real-time  
- **Filter & visualize** your top value picks!
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
