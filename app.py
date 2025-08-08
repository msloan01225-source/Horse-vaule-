import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ——— Your Racing API credentials ———
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(page_title="EdgeBet", layout="wide", initial_sidebar_state="auto")

# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- MOCK EDGE BRAIN (for fallback) ----
def generate_mock_data(n=40):
    horses  = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" for c in courses]
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
    df["Predicted Win %"]   = (1/df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6) + (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6) + (df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"] > 25, "✅",
        np.where(df["BetEdge Win %"] > 15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ---- LIVE API FETCH ----
@st.cache_data(ttl=300)
def fetch_live_data(day="today"):
    # API uses `from`/`to` & `countryCodes=GB`
    d = datetime.utcnow().date() + (timedelta(days=1) if day=="tomorrow" else timedelta())
    url = "https://api.theracingapi.com/v1/racecards"
    params = {
        "from":        d.isoformat(),
        "to":          d.isoformat(),
        "countryCodes":"GB"
    }
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def live_to_df(raw):
    rows = []
    for m in raw.get("meetings", []):
        course = m.get("course", {}).get("name", "Unknown")
        for race in m.get("races", []):
            off = race.get("off", "")[:5]
            for runner in race.get("runners", []):
                sp = runner.get("sp_dec")
                if sp is None: continue
                odds = float(sp)
                # simple placeholder value calc; swap in your real logic if available
                win_val = np.random.uniform(5, 25)
                rows.append({
                    "Horse":       runner.get("horse", "Unknown"),
                    "Course":      course,
                    "Country":     "UK" if course in ["Ascot","York","Cheltenham"] else "USA",
                    "Bookie":      "All",
                    "Time":        off,
                    "Odds":        round(odds,2),
                    "Win_Value":   round(win_val,1),
                    "Place_Value": round(win_val*0.6,1)
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Predicted Win %"]   = (1/df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6) + (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6) + (df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"] > 25, "✅",
        np.where(df["BetEdge Win %"] > 15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ---- UNIFIED DATA LOADER ----
def load_data(day="today"):
    try:
        raw = fetch_live_data(day)
        df  = live_to_df(raw)
        if df.empty:
            raise ValueError("no runners")
        return df
    except Exception as e:
        st.warning(f"Live API fetch failed: {e} — using mock data.")
        return generate_mock_data()

# ---- MAIN MENU ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# ---- PAGES ----
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the market with cutting-edge predictive models.")
    # show today's data summary
    df0 = load_data("today")
    st.metric("Races Today", df0["Course"].nunique())
    st.metric("Runners Today", len(df0))
    st.metric("Top Edge Value", f"{df0['BetEdge Win %'].max():.1f}%")

elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    day = st.radio("Select Day", ["today","tomorrow"], horizontal=True)
    df = load_data(day)

    # filters
    c1,c2,c3 = st.columns(3)
    with c1:
        country       = st.selectbox("Country",    ["All","UK","USA"])
    with c2:
        bookie        = st.selectbox("Bookmaker",  sorted(df["Bookie"].unique()))
    with c3:
        courses_all   = sorted(df["Course"].unique())
        course_filter = st.multiselect("Courses", courses_all, default=courses_all)

    # value slider
    mn, mx = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
    edge_range = st.slider("BetEdge Win % Range", mn, mx, (mn, mx))

    # view mode
    view_mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    # apply
    flt = df[
        ((df["Country"]==country)|(country=="All")) &
        (df["Bookie"]==bookie) &
        (df["Course"].isin(course_filter)) &
        (df["BetEdge Win %"].between(*edge_range))
    ]
    df_win   = flt.sort_values("BetEdge Win %", ascending=False)
    df_place = flt.sort_values("BetEdge Place %", ascending=False)

    if flt.empty:
        st.warning("No horses match those filters.")
    else:
        if view_mode=="📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(df_win,   use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(df_place, use_container_width=True)

elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Coming soon – EdgeBrain integration.")

elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.dataframe(
        load_data("today")[["Horse","Course","Odds","BetEdge Win %","Risk"]],
        use_container_width=True
    )

elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet combines:
    - ✅ Implied Probability from Odds  
    - 📊 Win & Place Value Metrics  
    - 🧠 Proprietary EdgeBrain Simulation  
    - 🔎 Risk Indicators & Smart Filters  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # placeholder

st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
