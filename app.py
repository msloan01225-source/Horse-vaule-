import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ─── THEME & PAGE SETUP ───────────────────────────────────────────────────────
st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")
st.markdown("""
  <style>
    body { background-color: #111111; color: white; }
    h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
    .css-1d391kg { background-color: #222 !important; }
  </style>
""", unsafe_allow_html=True)

# ─── SCRAPERS ────────────────────────────────────────────────────────────────
def get_racingpost_data(day="today"):
    url = "https://www.racingpost.com/racecards/time-order/"
    if day == "tomorrow":
        url += f"?date={(datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')}"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        for t in soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$")):
            tm     = t.strip()
            course = t.find_next(string=True).strip()
            horse  = t.find_next("a").get_text(strip=True)
            rows.append({"Horse": horse, "Course": course, "Time": tm})
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def get_timeform_data(day="today"):
    url = "https://www.timeform.com/horse-racing/racecards"
    if day == "tomorrow":
        url += f"?date={(datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')}"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        for t in soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$")):
            tm     = t.strip()
            course = t.find_next(string=True).strip()
            horse  = t.find_next("a").get_text(strip=True)
            rows.append({"Horse": horse, "Course": course, "Time": tm})
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_data(day="today"):
    rp = get_racingpost_data(day)
    tf = get_timeform_data(day)
    df = pd.concat([rp, tf], ignore_index=True).drop_duplicates(subset=["Horse","Time","Course"])
    if df.empty:
        return df

    # ─── MOCK ODDS & VALUE (replace with your API later) ──────────────────────
    df["Odds"]        = np.random.uniform(2, 10, len(df)).round(2)
    df["Win_Value"]   = np.random.uniform(5, 30, len(df)).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    # ──────────────────────────────────────────────────────────────────────────

    # compute predictions & BetEdge scores
    df["Predicted Win %"]   = (1/df["Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"]>25, "✅",
                  np.where(df["BetEdge Win %"]>15, "⚠️", "❌"))
    return df

# ─── SIDEBAR MENU ────────────────────────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# ─── PAGES ──────────────────────────────────────────────────────────────────
if selected=="Overview":
    df0 = load_data("today")
    st.title("📊 Welcome to EdgeBet")
    st.metric("Active Sports",2)
    st.metric("Total Runners",len(df0))
    st.metric("Top Edge Value",f"{df0['BetEdge Win %'].max() if not df0.empty else 0}%")

elif selected=="Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    day = st.radio("Race Day",["today","tomorrow"],horizontal=True)
    df  = load_data(day)

    country   = st.selectbox("Country",["All","UK","USA"])
    bookie    = st.selectbox("Bookmaker",["All","SkyBet","Bet365","Betfair"])
    courses   = df["Course"].unique().tolist() if not df.empty else []
    course_f  = st.multiselect("Courses",courses,default=courses)
    mi,ma     = int(df["BetEdge Win %"].min() if not df.empty else 0), int(df["BetEdge Win %"].max() if not df.empty else 0)
    edge_rng  = st.slider("BetEdge % range",mi,ma,(mi,ma))
    view_mode = st.radio("View Mode",["📊 Charts","📋 Tables"],horizontal=True)

    filt = (
      ((df["Country"]==country)|(country=="All")) &
      (df["Course"].isin(course_f)) &
      (df["BetEdge Win %"].between(*edge_rng))
    )
    df_f = df[filt] if not df.empty else df

    if df_f.empty:
        st.warning("No races match your filters.")
    else:
        win_sorted   = df_f.sort_values("BetEdge Win %",ascending=False).head(20)
        place_sorted = df_f.sort_values("BetEdge Place %",ascending=False).head(20)
        if view_mode=="📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(win_sorted.set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(place_sorted.set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win_sorted,use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(place_sorted,use_container_width=True)

elif selected=="Football":
    st.title("⚽ Football Value Picks")
    st.info("Coming soon – mock data for now.")

elif selected=="EdgeBrain":
    st.title("🧠 EdgeBrain")
    df0 = load_data("today")
    st.dataframe(df0[["Horse","Odds","BetEdge Win %","Risk"]],use_container_width=True)

elif selected=="How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
      - ✅ Implied Probability  
      - 📊 Value Metrics  
      - 🧠 Proprietary Simulation  
      - 🔎 Risk Filters  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # your host'd intro

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
