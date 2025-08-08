import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

st.set_page_config(page_title="EdgeBet", layout="wide")

# --- Load EdgeBrain Model if available ---
@st.cache_resource
def load_edgebrain_model():
    try:
        return joblib.load("edgebrain_model.pkl")
    except:
        return None

edge_model = load_edgebrain_model()

def predict_edgebrain(df):
    if edge_model:
        X = df[["Odds","Win_Value","Place_Value"]]
        return (edge_model.predict_proba(X)[:,1]*100).round(1)
    else:
        # stub: echo BetEdge Win %
        return df["BetEdge Win %"]

# --- Live scrape Racing Post SP with fallback to mock ---
@st.cache_data(ttl=300)
def fetch_live(day="today"):
    date = datetime.utcnow().date()
    if day=="tomorrow":
        date += timedelta(days=1)
    url = f"https://www.racingpost.com/racecards/time-order/{date.isoformat()}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except:
        return pd.DataFrame()
    soup = BeautifulSoup(r.text,"html.parser")
    rows=[]
    for sec in soup.select("section.race-time"):
        time_tag = sec.select_one(".rc-race-time")
        course_tag = sec.select_one(".rc-race-meeting__course a")
        if not time_tag or not course_tag:
            continue
        off = time_tag.text.strip()
        course = course_tag.text.strip()
        for runner in sec.select(".runner-wrap"):
            name = runner.select_one(".runner-runner__name")
            sp   = runner.select_one(".runner-sp__price")
            if not name or not sp: continue
            try:
                odds = float(sp.text.strip())
            except:
                continue
            rows.append({
                "Horse":name.text.strip(),
                "Course":course,
                "Time":off,
                "Odds":odds
            })
    return pd.DataFrame(rows)

def generate_mock(n=30):
    horses  = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"],n)
    odds    = np.random.uniform(2,10,n).round(2)
    return pd.DataFrame({"Horse":horses,"Course":courses,"Odds":odds})

def build_df(day="today"):
    df = fetch_live(day)
    if df.empty:
        df = generate_mock()
        df["Time"] = ""
    # value metrics
    df["Win_Value"]    = np.random.uniform(5,30,len(df)).round(1)
    df["Place_Value"]  = (df["Win_Value"]*0.6).round(1)
    df["Pred Win %"]   = (1/df["Odds"]*100).round(1)
    df["Pred Pl %"]    = (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"]= ((df["Pred Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Pl %"] = ((df["Pred Pl %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"]         = np.where(df["BetEdge Win %"]>25,"✅",
                           np.where(df["BetEdge Win %"]>15,"⚠️","❌"))
    df["EdgeBrain %"]  = predict_edgebrain(df)
    df["Country"]      = df["Course"].apply(
        lambda c: "UK" if c in ["Ascot","York","Cheltenham"] else "USA")
    df["Bookie"]       = np.random.choice(
        ["SkyBet","Bet365","Betfair","PaddyPower","WilliamHill"], len(df))
    return df

# --- Sidebar Menu ---
with st.sidebar:
    page = option_menu("🏇 EdgeBet", 
                       ["Overview","Horse Racing","EdgeBrain","How It Works"],
                       icons=["house","card-list","robot","info-circle"],
                       default_index=0)

# --- Overview ---
if page=="Overview":
    st.title("📊 EdgeBet Live Overview")
    df0 = build_df()
    st.metric("Races",     df0["Course"].nunique())
    st.metric("Runners",   len(df0))
    st.metric("Top Value", f"{df0['BetEdge Win %'].max()}%")

# --- Horse Racing ---
elif page=="Horse Racing":
    st.title("🏇 Horse Racing")
    day = st.selectbox("Day",["today","tomorrow"])
    df = build_df(day)
    # filters
    country = st.selectbox("Country",["All","UK","USA"])
    bookie  = st.selectbox("Bookie", ["All"]+df["Bookie"].unique().tolist())
    courses = st.multiselect("Courses", df["Course"].unique(), default=df["Course"].unique())
    rng = st.slider("BetEdge Win %", int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max()),
                    (int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())))
    view = st.radio("View",["Charts","Tables"], horizontal=True)

    # apply
    mask = ((df["Country"]==country)|(country=="All")) & \
           ((df["Bookie"]==bookie)|(bookie=="All")) & \
           (df["Course"].isin(courses)) & \
           (df["BetEdge Win %"].between(*rng))
    df2 = df[mask]

    if df2.empty:
        st.warning("No runners after filtering.")
    else:
        if view=="Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(df2.sort_values("BetEdge Win %",ascending=False)
                              .head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Pl %")
            st.bar_chart(df2.sort_values("BetEdge Pl %",ascending=False)
                              .head(20).set_index("Horse")["BetEdge Pl %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(df2.reset_index(drop=True), use_container_width=True)

# --- EdgeBrain ---
elif page=="EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    df = build_df()
    st.subheader("Top 20 EdgeBrain %")
    st.bar_chart(df.sort_values("EdgeBrain %",ascending=False)
                    .head(20).set_index("Horse")["EdgeBrain %"])
    st.dataframe(df[["Horse","Course","Odds","EdgeBrain %","Risk"]], use_container_width=True)

# --- How It Works ---
else:
    st.title("ℹ️ How It Works")
    st.markdown("""
- **Live SP scrape** from Racing Post (fallback to mock)  
- **BetEdge** value = 0.6·Implied% + 0.4·WinValue  
- **EdgeBrain** ML model pulse (or stub)  
- Full **country**, **bookie**, **course**, **%** filters  
- Charts & Tables side by side  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
