import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ─── PAGE CONFIG & THEME ──────────────────────────────────────────────────────
st.set_page_config(page_title="EdgeBet – Phase 1", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
body { background-color: #111111; color: #EFEFEF; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ─── SCRAPERS ─────────────────────────────────────────────────────────────────
def get_racingpost_data(day="Today"):
    """Scrape Racing Post time-order racecards."""
    url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower()=="tomorrow":
        d = (datetime.utcnow()+timedelta(days=1)).strftime("%Y-%m-%d")
        url += f"?date={d}"
    try:
        r = requests.get(url, timeout=15); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows=[]
        for tag in soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$")):
            race_time = tag.strip()
            course = tag.find_next(string=True).strip()
            name = tag.find_next("a").get_text(strip=True)
            rows.append({"Course":course, "Race":f"{course} {race_time} – {name}",
                         "Time":race_time, "Horse":name,
                         "Win_Value":0.0, "Place_Value":0.0})
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame(columns=["Course","Race","Time","Horse","Win_Value","Place_Value"])

def get_timeform_data(day="Today"):
    """Scrape Timeform racecards."""
    url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower()=="tomorrow":
        d = (datetime.utcnow()+timedelta(days=1)).strftime("%Y-%m-%d")
        url += f"?date={d}"
    try:
        r = requests.get(url, timeout=15); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows=[]
        for tag in soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$")):
            race_time = tag.strip()
            course = tag.find_next(string=True).strip()
            name   = tag.find_next("a").get_text(strip=True)
            rows.append({"Course":course, "Race":f"{course} {race_time} – {name}",
                         "Time":race_time, "Horse":name,
                         "Win_Value":0.0, "Place_Value":0.0})
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame(columns=["Course","Race","Time","Horse","Win_Value","Place_Value"])

# ─── DATA LOADER & VALUE CALCS ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(day="Today"):
    rp = get_racingpost_data(day)
    tf = get_timeform_data(day)
    if rp.empty and tf.empty:
        return pd.DataFrame()
    try:
        df = pd.merge(rp, tf, on=["Race","Horse"], how="inner",
                      suffixes=("_RP","_TF"))
    except:
        df = pd.concat([rp,tf], ignore_index=True)
    # Merge Win/Place values
    if {"Win_Value_RP","Win_Value_TF"}.issubset(df.columns):
        df["Win_Value"] = df[["Win_Value_RP","Win_Value_TF"]].mean(axis=1)
    df["Win_Value"] = pd.to_numeric(df.get("Win_Value",0),errors="coerce").fillna(0)
    if {"Place_Value_RP","Place_Value_TF"}.issubset(df.columns):
        df["Place_Value"] = df[["Place_Value_RP","Place_Value_TF"]].mean(axis=1)
    df["Place_Value"] = pd.to_numeric(df.get("Place_Value",0),errors="coerce").fillna(0)
    # Mock Best Odds
    df["Best Odds"] = np.random.uniform(2,6,len(df)).round(2)
    df["Predicted Win %"]   = (1/df["Best Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    # BetEdge algorithm
    df["BetEdge Win %"]   = ((df["Predicted Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    # Country & Bookie mocks
    df["Country"] = np.where(df["Course"].isin(["Ascot","York","Cheltenham"]),"UK","USA")
    df["Bookie"]  = np.random.choice(["SkyBet","Bet365","Betfair","Paddy","William Hill"], len(df))
    return df.reset_index(drop=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
day = st.selectbox("📅 Select Day:", ["Today","Tomorrow"], key="day")
df = load_data(day)
if df.empty:
    st.warning("No race data available for that day.")
else:
    # Sidebar navigation
    with st.sidebar:
        sel = option_menu("🏠 Main Menu",
            ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
            icons=["house","activity","soccer","robot","book"],
            menu_icon="cast", default_index=1)

    # Colour function
    def color_val(v):
        if v>20: return 'background-color:#58D68D;color:black'
        if v>10: return 'background-color:#F9E79F;color:black'
        return 'background-color:#F5B7B1;color:black'

    # ─── OVERVIEW ────────────────────────────────────────────────────────────
    if sel=="Overview":
        st.title("📊 EdgeBet Overview")
        st.metric("Active Sports", 2)
        st.metric("Total Runners", len(df))
        st.metric("Top Edge Value", f"{df['BetEdge Win %'].max()}%")
        st.write("Use the sidebar to drill into Horse Racing, Football, or our EdgeBrain module.")

    # ─── HORSE RACING ────────────────────────────────────────────────────────
    elif sel=="Horse Racing":
        st.title("🏇 Horse Racing – UK & USA")
        # filters
        c1,c2,c3 = st.columns(3)
        with c1:
            country   = st.selectbox("Country", ["All","UK","USA"])
        with c2:
            bookie    = st.selectbox("Bookmaker", ["All"]+sorted(df["Bookie"].unique()))
        with c3:
            courses   = st.multiselect("Courses", sorted(df["Course"].unique()), default=None)
        # BetEdge filter
        min_v,max_v = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
        edge_range = st.slider("🎯 BetEdge Win % Range", min_v, max_v, (min_v,max_v))
        # view mode
        view_mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

        # apply filters
        f = df.copy()
        if country!="All": f=f[f.Country==country]
        if bookie!="All":  f=f[f.Bookie==bookie]
        if courses:       f=f[f.Course.isin(courses)]
        f = f[f["BetEdge Win %"].between(*edge_range)]

        if f.empty:
            st.warning("No horses match those filters.")
        else:
            win_df    = f.sort_values("BetEdge Win %", ascending=False)
            place_df  = f.sort_values("BetEdge Place %", ascending=False)
            if view_mode=="📊 Charts":
                st.subheader("Top 20 BetEdge Win %")
                st.bar_chart(win_df.head(20).set_index("Horse")["BetEdge Win %"])
                st.subheader("Top 20 BetEdge Place %")
                st.bar_chart(place_df.head(20).set_index("Horse")["BetEdge Place %"])
            else:
                st.subheader("🏆 Win Rankings")
                st.dataframe(win_df.style.applymap(color_val, subset=["BetEdge Win %"]), use_container_width=True)
                st.subheader("🥈 Place Rankings")
                st.dataframe(place_df.style.applymap(color_val, subset=["BetEdge Place %"]), use_container_width=True)

    # ─── FOOTBALL (MOCK) ─────────────────────────────────────────────────────
    elif sel=="Football":
        st.title("⚽ Football Value Picks")
        st.info("🚧 Coming in Step 2 with live data & EdgeBrain integration…")

    # ─── EDGEBRAIN (MOCK) ────────────────────────────────────────────────────
    elif sel=="EdgeBrain":
        st.title("🧠 EdgeBrain – Smart Predictions")
        st.write("Simulated picks with risk indicators:")
        display = df[["Horse","Course","Odds","BetEdge Win %"]].copy()
        display["Risk"] = np.where(display["BetEdge Win %"]>25,"✅",
                             np.where(display["BetEdge Win %"]>15,"⚠️","❌"))
        st.dataframe(display, use_container_width=True)

    # ─── HOW IT WORKS ────────────────────────────────────────────────────────
    else:
        st.title("📚 How EdgeBet Works")
        st.markdown("""
        EdgeBet combines:
        - 🤖 **EdgeBrain** simulations  
        - 📊 **Value Metrics** from Odds & historical Value  
        - 🎯 **Risk Indicators** to guide your stake  
        - 🔄 **Live racecards** from Racing Post & Timeform  

        Follow Phase 2 to hook in The Racing API and Phase 3 for Football & advanced analytics.
        """)

    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
