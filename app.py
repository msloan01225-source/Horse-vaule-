import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ——— Auth & Config ———
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide",
                   initial_sidebar_state="auto")

# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- MOCK EDGE BRAIN ----
def generate_mock_data(n=40):
    horses   = [f"Horse {i+1}" for i in range(n)]
    courses  = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    countries = ["UK" if c in ("Ascot","York","Cheltenham") else "USA"
                 for c in courses]
    bookies  = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds     = np.random.uniform(2,10,n).round(2)
    win_val  = np.random.uniform(5,30,n).round(1)

    df = pd.DataFrame({
        "Horse":       horses,
        "Course":      courses,
        "Country":     countries,
        "Bookie":      bookies,
        "Odds":        odds,
        "Win_Value":   win_val,
        "Place_Value": (win_val * 0.6).round(1)
    })

    df["Predicted Win %"]    = (1/df["Odds"] * 100).round(1)
    df["Predicted Place %"]  = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]      = ((df["Predicted Win %"]*0.6) + 
                                 (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]    = ((df["Predicted Place %"]*0.6) + 
                                 (df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"]>25, "✅",
        np.where(df["BetEdge Win %"]>15, "⚠️","❌")
    )

    return df.sort_values("BetEdge Win %", ascending=False)\
             .reset_index(drop=True)

# ---- LIVE DATA FETCH via TheRacingAPI ----
@st.cache_data(ttl=300)
def fetch_live_data(day="today"):
    # pick date
    d = datetime.utcnow().date()
    if day=="tomorrow":
        d += timedelta(days=1)
    iso = d.isoformat()

    # 1) get all stages (meetings) for that date
    stages = []
    try:
        resp = requests.get(
            "https://api.theracingapi.com/v1/stages",
            auth=HTTPBasicAuth(USERNAME,PASSWORD),
            params={"from":iso,"to":iso,"countryCodes":"GB"},
            timeout=10
        )
        resp.raise_for_status()
        stages = resp.json().get("stages",[])
    except Exception as e:
        return pd.DataFrame()  # trigger fallback

    rows = []
    # 2) for each meeting pull its racecard
    for s in stages:
        stage_id = s["id"]
        course   = s["course"]["name"]
        try:
            rc = requests.get(
                f"https://api.theracingapi.com/v1/racecards/{stage_id}",
                auth=HTTPBasicAuth(USERNAME,PASSWORD),
                timeout=10
            )
            rc.raise_for_status()
            rcj = rc.json()
        except:
            continue

        for race in rcj.get("races",[]):
            off = race.get("off","")[:5]
            for rnr in race.get("runners",[]):
                sp = rnr.get("sp_dec")
                if sp is None: 
                    continue
                odds = float(sp)
                # placeholder value calc:
                win_val = np.random.uniform(5,25)

                rows.append({
                    "Horse":       rnr.get("horse","Unknown"),
                    "Course":      course,
                    "Country":     "UK" if course in 
                                   ("Ascot","York","Cheltenham") else "USA",
                    "Bookie":      "All",
                    "Time":        off,
                    "Odds":        round(odds,2),
                    "Win_Value":   round(win_val,1),
                    "Place_Value": round(win_val*0.6,1)
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # compute metrics
    df["Predicted Win %"]    = (1/df["Odds"] * 100).round(1)
    df["Predicted Place %"]  = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"]      = ((df["Predicted Win %"]*0.6)+
                                 (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]    = ((df["Predicted Place %"]*0.6)+
                                 (df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"]>25, "✅",
        np.where(df["BetEdge Win %"]>15, "⚠️","❌")
    )

    return df.sort_values("BetEdge Win %", ascending=False)\
             .reset_index(drop=True)

# 3) fallback loader
def load_data(day="today"):
    live = fetch_live_data(day)
    if not live.empty:
        return live
    st.warning("Live API fetch failed — using mock data.")
    return generate_mock_data()

# ---- NAVIGATION ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# ---- PAGES ----
if selected=="Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over bookmakers with our hybrid value engine.")
    df0 = load_data("today")
    st.metric("Active Sports", 1)
    st.metric("Total Runners", len(df0))
    st.metric("Top Edge Value", f"{df0['BetEdge Win %'].max():.1f}%")

elif selected=="Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    df_hr = load_data("today")

    # filters
    c1,c2,c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with c2:
        bookie  = st.selectbox("Bookie", ["All","All"])
    with c3:
        courses = sorted(df_hr["Course"].unique())
        flt     = st.multiselect("Courses", courses, default=courses)

    rmin = int(df_hr["BetEdge Win %"].min())
    rmax = int(df_hr["BetEdge Win %"].max())
    rng  = st.slider("Edge % filter", rmin, rmax, (rmin,rmax))

    view = st.radio("View Mode", ["Charts","Tables"], horizontal=True)

    f = df_hr[
        ((df_hr["Country"]==country)|(country=="All")) &
        (df_hr["Course"].isin(flt)) &
        (df_hr["BetEdge Win %"].between(*rng))
    ]
    win = f.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    plc = f.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if f.empty:
        st.warning("No runners match filters.")
    else:
        if view=="Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(win.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(plc.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(plc, use_container_width=True)

elif selected=="Football":
    st.title("⚽ Football Value Picks")
    st.info("Coming soon…")

elif selected=="EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("Under the hood: our hybrid simulation + ML pipeline.")
    df_sb = load_data("today")
    st.dataframe(df_sb[["Horse","Course","Odds","BetEdge Win %","Risk"]],
                 use_container_width=True)

elif selected=="How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    - 🏇 Live odds → implied probabilities  
    - 📊 Value overlays (Win/Place)  
    - 🤖 EdgeBrain hybrid simulation  
    - 🔎 Drill-down filters & risk flags  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

# ---- FOOTER ----
st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
