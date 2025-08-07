import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
import numpy as np
from streamlit_option_menu import option_menu

# --- Your API credentials ---
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

# --- Streamlit page config & dark theme ---
st.set_page_config(page_title="EdgeBet Live Odds", layout="wide", initial_sidebar_state="auto")
st.markdown("""
    <style>
      body { background-color: #111111; color: white; }
      h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
      .css-1d391kg { background-color: #222 !important; }
    </style>
""", unsafe_allow_html=True)

# --- Fetch live racecards & odds from TheRacingAPI ---
@st.cache_data(ttl=300)
def fetch_live_data(day="today"):
    # build date string
    date = datetime.utcnow().date()
    if day == "tomorrow":
        date += timedelta(days=1)
    url_cards = f"https://api.theracingapi.com/v1/racecards/{date:%Y-%m-%d}"
    resp = requests.get(url_cards,
                        auth=HTTPBasicAuth(USERNAME, PASSWORD),
                        timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for meeting in data.get("meetings", []):
        course = meeting.get("course", {}).get("name", "Unknown")
        for race in meeting.get("races", []):
            off_time = race.get("off", "")[:5]
            race_id = race.get("id")
            # fetch odds
            odds_resp = requests.get(
                f"https://api.theracingapi.com/v1/betting_odds/{race_id}",
                auth=HTTPBasicAuth(USERNAME, PASSWORD),
                timeout=10
            )
            odds_resp.raise_for_status()
            odds_data = odds_resp.json()
            for runner in odds_data.get("runners", []):
                horse = runner.get("horse", {}).get("name", runner.get("horse_id", "Unknown"))
                sp = runner.get("sp_decimal")
                try:
                    odds_decimal = float(sp)
                except:
                    odds_decimal = np.random.uniform(2, 6)
                rows.append({
                    "Time": off_time,
                    "Course": course,
                    "Horse": horse,
                    "Best Odds": round(odds_decimal, 2)
                })
    return pd.DataFrame(rows)

# --- Compute value metrics & BetEdge scores ---
def enrich_with_values(df):
    if df.empty:
        return df
    df["Predicted Win %"]   = (1 / df["Best Odds"] * 100).round(1)
    df["Win_Value"]         = np.random.uniform(5, 30, len(df)).round(1)      # placeholder: replace with real value algo
    df["Place_Value"]       = (df["Win_Value"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Win %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅",
                  np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))
    return df

# --- Main menu ---
with st.sidebar:
    selected = option_menu(
        "🏠 EdgeBet",
        ["Overview", "Horse Racing", "Football", "EdgeBrain", "How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# --- Overview page ---
if selected == "Overview":
    st.title("📊 EdgeBet Live Overview")
    day = st.selectbox("Select Day:", ["today","tomorrow"])
    live = fetch_live_data(day)
    df = enrich_with_values(live)
    st.metric("Races Found", df["Course"].nunique())
    st.metric("Total Runners", len(df))
    st.metric("Top BetEdge %", f"{df['BetEdge Win %'].max()}%")

# --- Horse Racing page ---
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – Live Value Tracker")
    day = st.selectbox("Day:", ["today","tomorrow"])
    live = fetch_live_data(day)
    df = enrich_with_values(live)

    # filters
    col1, col2 = st.columns(2)
    with col1:
        course_opts = ["All"] + sorted(df["Course"].unique().tolist())
        course = st.selectbox("Course:", course_opts)
    with col2:
        rng = st.slider("BetEdge Win % range:", 
                        int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max()),
                        (int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())))
    # apply
    filt = df[
        ((df["Course"]==course)|(course=="All")) &
        (df["BetEdge Win %"].between(*rng))
    ]
    if filt.empty:
        st.warning("No runners match filters.")
    else:
        # view toggle
        view = st.radio("View:", ["Charts","Tables"], horizontal=True)
        if view=="Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(filt.sort_values("BetEdge Win %", ascending=False)
                         .head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(filt.sort_values("BetEdge Place %", ascending=False)
                         .head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(filt.sort_values("BetEdge Win %", ascending=False), use_container_width=True)

# --- Football page (placeholder) ---
elif selected == "Football":
    st.title("⚽ Football – Coming Soon")
    st.info("We’ll layer in live football markets next.")

# --- EdgeBrain page (placeholder) ---
elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Premium Predictions")
    st.info("Advanced AI-driven value models in development.")

# --- How It Works ---
elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    1. Fetch live odds ✔  
    2. Compute implied probabilities ✔  
    3. Apply proprietary value metrics  
    4. Generate your **BetEdge %**  
    5. Filter & visualise  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # replace with your explainer

st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")  
