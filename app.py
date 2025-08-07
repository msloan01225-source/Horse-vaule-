import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# --- Auth credentials (keep same) ---
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(
    page_title="EdgeBet – Phase 1 & 3",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---- DARK THEME ----
st.markdown("""
    <style>
      body { background-color: #111; color: #EEE; }
      h1, h2, h3, h4, h5, h6 { color: #00FFCC; }
      .css-1d391kg { background-color: #222 !important; }
    </style>
""", unsafe_allow_html=True)

# ---- LIVE UK RACING API ----
@st.cache_data(ttl=300)
def fetch_meetings(day: str = "today"):
    d = datetime.utcnow().date()
    if day == "tomorrow":
        d += timedelta(days=1)
    params = {
        "from": d.strftime("%Y-%m-%d"),
        "to":   d.strftime("%Y-%m-%d"),
        "countryCodes": "GB",
    }
    url = "https://api.theracingapi.com/v1/meetings"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, timeout=10)
    r.raise_for_status()
    return r.json().get("meetings", [])

@st.cache_data(ttl=300)
def fetch_racecards_for(meeting_uid: str):
    url = f"https://api.theracingapi.com/v1/racecards/{meeting_uid}"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=10)
    r.raise_for_status()
    return r.json().get("runners", [])

def load_live_uk(day="today"):
    meetings = fetch_meetings(day)
    rows = []
    for m in meetings:
        course = m.get("course",{}).get("name","Unknown")
        for runner in fetch_racecards_for(m["meeting_uid"]):
            horse = runner.get("horse", runner.get("horse_id","Unknown"))
            odds  = float(runner.get("sp_dec", np.random.uniform(2,6)))
            # these two would come from the API if provided, else mock
            win_val   = np.random.uniform(5,25)
            place_val = win_val * 0.6
            rows.append({
                "Course": course,
                "Time":   runner.get("race_time","")[:5],
                "Horse":  horse,
                "Best Odds": round(odds,2),
                "Win_Value": round(win_val,1),
                "Place_Value": round(place_val,1),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Predicted Win %"]   = (1/df["Best Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ---- MOCK EDGEBRAIN (Phase 3) ----
def generate_mock_data(n=20):
    horses = [f"Horse {i+1}" for i in range(n)]
    odds    = np.random.uniform(2,10,n).round(2)
    win_val = np.random.uniform(5,30,n).round(1)
    df = pd.DataFrame({
        "Horse": horses,
        "Odds": odds,
        "Win_Value": win_val,
        "Place_Value": (win_val*0.6).round(1),
    })
    df["Predicted Win %"]   = (1/df["Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"]>25, "✅",
                  np.where(df["BetEdge Win %"]>15, "⚠️","❌"))
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

mock_df = generate_mock_data()

# ---- SIDEBAR NAVIGATION ----
with st.sidebar:
    page = option_menu(
      "🏠 Main Menu",
      ["Overview","Horse Racing","EdgeBrain","How It Works"],
      icons=["house","activity","robot","book"],
      menu_icon="cast", default_index=0
    )

# ---- PAGES ----
if page == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Live UK racing value + our EdgeBrain proof-of-concept.")
    st.metric("Live Races (UK)", len(fetch_meetings("today")))
    st.metric("Mock EdgeBrain Runners", len(mock_df))
    st.metric("Top Live Edge %", f"{load_live_uk('today')['BetEdge Win %'].max()}%")

elif page == "Horse Racing":
    st.title("🏇 Horse Racing – Live UK")
    df_live = load_live_uk("today")
    if df_live.empty:
        st.warning("No live UK races found for today.")
    else:
        # simple table + bar chart
        st.subheader("🏆 Top Live Edge Picks")
        st.dataframe(df_live[["Course","Horse","Best Odds","BetEdge Win %","BetEdge Place %"]])
        st.subheader("📊 BetEdge Win % Chart")
        st.bar_chart(df_live.head(20).set_index("Horse")["BetEdge Win %"])

elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain – Mock Historical")
    st.dataframe(mock_df)

elif page == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    1. **Live Implied Probability** from UK API  
    2. **Value Metrics** (Win & Place)  
    3. **Hybrid BetEdge Score**  
    4. **Risk Indicators**  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
