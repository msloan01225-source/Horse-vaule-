# app.py
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

st.set_page_config(
    page_title="BetEdge Live",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── INJECT CUSTOM CSS ──────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* override Streamlit default header/footer */
    header, footer {visibility: hidden;}
    /* main background */
    .css-1d391kg { background-color: #041C32 !important; }
    /* sidebar */
    .css-1d391kg .css-1d391kg { background-color: #0B1E2D !important; }
    /* table header */
    thead th { background-color: #00A5CF !important; color:#041C32 !important; }
    /* table rows */
    tbody tr:nth-child(even) { background-color: #0B2A44 !important; }
    tbody tr:nth-child(odd)  { background-color: #0B1E2D !important; }
    /* metrics boxes */
    .stMetric > div { background-color: #0B2A44 !important; border: 1px solid #00A5CF; }
    </style>
""", unsafe_allow_html=True)

# ─── LOAD EDGE-BRAIN MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load_edgebrain_model():
    try:
        return joblib.load("edgebrain_model.pkl")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ edgebrain_model.pkl not found — using BetEdge stub.")
        return None

edge_model = load_edgebrain_model()

def predict_edgebrain(df: pd.DataFrame) -> pd.Series:
    if edge_model:
        X = df[["Odds","Win_Value","Place_Value"]]
        return (edge_model.predict_proba(X)[:,1]*100).round(1)
    return df["BetEdge Win %"]

# ─── LIVE DATA FALLBACK ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_live_api(day="today"):
    date = datetime.utcnow().date() + (timedelta(days=1) if day=="tomorrow" else timedelta())
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"region":"GB","date":date.isoformat()}
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME,PASSWORD), params=params, timeout=8)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def fetch_live_sp(day="today"):
    date = datetime.utcnow().date() + (timedelta(days=1) if day=="tomorrow" else timedelta())
    url = f"https://www.racingpost.com/racecards/time-order/{date.isoformat()}"
    r = requests.get(url, timeout=8); r.raise_for_status()
    soup = BeautifulSoup(r.text,"html.parser")
    rows=[]
    for sec in soup.select("section.race-time"):
        time_el = sec.select_one(".rc-race-time"); course_el = sec.select_one(".rc-race-meeting__course a")
        if not time_el or not course_el: continue
        off, course = time_el.text.strip(), course_el.text.strip()
        for runner in sec.select(".runner-wrap"):
            nm = runner.select_one(".runner-runner__name"); sp = runner.select_one(".runner-sp__price")
            if nm and sp:
                try: od=float(sp.text.strip())
                except: continue
                rows.append({"Horse":nm.text.strip(),"Course":course,"Time":off,"Odds":od})
    return pd.DataFrame(rows)

def gen_mock(n=30):
    horses=[f"H{i+1}" for i in range(n)]
    courses=np.random.choice(["Ascot","York","Cheltenham","Santa Anita"],n)
    odds=np.random.uniform(2,10,n).round(2)
    return pd.DataFrame({"Horse":horses,"Course":courses,"Odds":odds,"Time":[""]*n})

def build_live_df(day="today"):
    # try API
    try:
        raw = fetch_live_api(day)
        rows=[]
        for m in raw.get("meetings",[]):
            course=m.get("course",{}).get("name","")
            for race in m.get("races",[]):
                off=race.get("off","")[:5]
                for rnr in race.get("runners",[]):
                    sp=rnr.get("sp_dec")
                    if sp is None: continue
                    rows.append({"Horse":rnr["horse"],"Course":course,"Time":off,"Odds":float(sp)})
        df=pd.DataFrame(rows)
        if df.empty: raise ValueError
    except:
        # try scrape
        df = fetch_live_sp(day)
        if df.empty:
            # fallback mock
            df = gen_mock(30)

    # value & EdgeBrain calc
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
                            lambda c: "UK" if c in ["Ascot","York","Cheltenham"] else "USA"
                        )
    return df

# ─── UI ────────────────────────────────────────────────────────────────────────
day = st.sidebar.selectbox("▶️ Day", ["today","tomorrow"])
df  = build_live_df(day)

tabs = option_menu(
    None,
    ["Overview","Horse Racing","EdgeBrain","How It Works"],
    icons=["house","card-list","robot","info-circle"],
    menu_icon="cast", default_index=0, orientation="horizontal"
)

if tabs=="Overview":
    st.title("📊 Live EdgeBet Tracker")
    col1,col2,col3 = st.columns(3)
    col1.metric("Races", df["Course"].nunique())
    col2.metric("Runners", len(df))
    col3.metric("Top BetEdge", f"{df['BetEdge Win %'].max()}%")

elif tabs=="Horse Racing":
    st.title("🏇 Horse Racing – UK/USA Live")
    country = st.selectbox("Country",["All","UK","USA"])
    if country!="All": df=df[df["Country"]==country]
    st.subheader("📊 Top 20 BetEdge Win %")
    st.bar_chart(df.sort_values("BetEdge Win %",ascending=False)
                    .head(20).set_index("Horse")["BetEdge Win %"])
    st.subheader("🏆 Full Table")
    st.dataframe(df, use_container_width=True)

elif tabs=="EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    st.subheader("📊 Top 20 EdgeBrain Win %")
    st.bar_chart(df.sort_values("EdgeBrain %",ascending=False)
                    .head(20).set_index("Horse")["EdgeBrain %"])
    st.dataframe(df[["Horse","Course","Odds","EdgeBrain %","Risk"]], use_container_width=True)

else:
    st.title("ℹ️ How It Works")
    st.markdown("""
  1. **API → Scrape → Mock**  
  2. Compute **BetEdge** & **EdgeBrain** in real-time  
  3. Filter, chart, table your top value picks
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
