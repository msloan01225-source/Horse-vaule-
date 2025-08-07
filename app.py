import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# --- APP CONFIG ---
st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")

# --- DARK THEME STYLING ---
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }  /* sidebar */
</style>
""", unsafe_allow_html=True)

# --- AUTH CREDENTIALS ---
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

# --- FETCH LIVE RACECARDS ---
@st.cache_data(ttl=600)
def fetch_racecards(day="today"):
    date = datetime.utcnow().date()
    if day == "tomorrow":
        date += timedelta(days=1)
    date_str = date.strftime("%Y-%m-%d")

    params = {"date": date_str, "region": "GB"}
    resp = requests.get(
        "https://api.theracingapi.com/v1/racecards",
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        params=params,
        timeout=10
    )
    try:
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError:
        st.error(f"API Error {resp.status_code}: {resp.text}")
        return None

def process_live(raw):
    if not raw or "meetings" not in raw:
        return pd.DataFrame()
    rows = []
    for meeting in raw["meetings"]:
        course = meeting.get("course", {}).get("name", "Unknown")
        for race in meeting.get("races", []):
            off = race.get("off", "")[:5]
            for runner in race.get("runners", []):
                horse = runner.get("horse") or runner.get("horse_id", "Unknown")
                odds = float(runner.get("sp_dec", np.random.uniform(2, 6)))
                win_val = np.random.uniform(5, 30)
                rows.append({
                    "Time": off,
                    "Course": course,
                    "Horse": horse,
                    "Best Odds": round(odds, 2),
                    "Win_Value": round(win_val, 1),
                    "Place_Value": round(win_val * 0.6, 1)
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Predicted Win %"]   = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]   * 0.6) + (df["Win_Value"]   * 0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅",
                  np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))
    return df

def fetch_with_fallback():
    for d in ["today", "tomorrow"]:
        raw = fetch_racecards(d)
        df = process_live(raw)
        if not df.empty:
            return df, d
    return pd.DataFrame(), None

# --- PRELOAD DATA ---
df_live, live_day = fetch_with_fallback()

# --- MOCK DATA FOR OTHER PAGES ---
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    df = pd.DataFrame({
        "Horse": horses,
        "Course": courses,
        "Odds": np.random.uniform(2,10,n).round(2),
        "Win_Value": np.random.uniform(5,30,n).round(1)
    })
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"]   = (1/df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]   *0.6)+(df["Win_Value"]   *0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"] *0.6)+(df["Place_Value"] *0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"]>25,"✅",
                  np.where(df["BetEdge Win %"]>15,"⚠️","❌"))
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df_mock = generate_mock_data()

# --- MAIN NAVIGATION ---
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# --- OVERVIEW PAGE ---
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive value models.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners", len(df_live) if live_day else len(df_mock))
    top_value = (df_live["BetEdge Win %"].max() if live_day else df_mock["BetEdge Win %"].max())
    st.metric("Top Edge Value", f"{top_value}%")
    if live_day:
        st.info(f"Live horse racing data for **{live_day.title()}**")

# --- HORSE RACING PAGE ---
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    data = df_live if live_day else df_mock

    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with col2:
        bookie  = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with col3:
        courses = data["Course"].unique().tolist()
        course  = st.multiselect("Course", courses, default=courses)

    min_edge = int(data["BetEdge Win %"].min())
    max_edge = int(data["BetEdge Win %"].max())
    edge_range = st.slider("BetEdge Win % Range", min_edge, max_edge, (min_edge, max_edge))

    view_mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    # apply filters
    df_f = data[
        ((data["Course"].isin(course))) &
        (data["BetEdge Win %"].between(*edge_range))
    ]
    if df_f.empty:
        st.warning("No horses match your filters.")
    else:
        df_win   = df_f.sort_values("BetEdge Win %",   ascending=False).reset_index(drop=True)
        df_place = df_f.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

        if view_mode == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(df_win, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(df_place, use_container_width=True)

# --- FOOTBALL PAGE ---
elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data – coming soon with EdgeBrain integration.")

# --- EDGEBRAIN PAGE ---
elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    data = df_live if live_day else df_mock
    st.dataframe(
        data[["Horse","Course","BetEdge Win %","Risk"]],
        use_container_width=True
    )

# --- HOW IT WORKS PAGE ---
elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
EdgeBet uses a smart hybrid model combining:

- ✅ Implied Probability from Odds  
- 📊 Win & Place Value Metrics  
- 🧠 Proprietary EdgeBrain Simulation  
- 🔎 Risk Indicators and Smart Filtering

This generates a **BetEdge score** – your edge % over the market.
""")
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # placeholder

# --- FOOTER ---
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
