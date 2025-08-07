import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ───── CONFIG ─────
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"
API_URL  = "https://api.theracingapi.com/v1/racecards"

st.set_page_config(
    page_title="EdgeBet – Phase 4: Live Data", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ───── DARK THEME ─────
st.markdown("""
    <style>
    body { background-color: #111111; color: white; }
    h1,h2,h3,h4,h5,h6 { color: #00ffcc; }
    .css-1d391kg { background-color: #222 !important; }
    </style>
""", unsafe_allow_html=True)

# ───── MOCK DATA GENERATOR ─────
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" 
                 for c in courses]
    bookies = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds = np.random.uniform(2, 10, n).round(2)
    win_val = np.random.uniform(5, 30, n).round(1)
    df = pd.DataFrame({
        "Time":      "--:--",
        "Country":   countries,
        "Course":    courses,
        "Bookie":    bookies,
        "Horse":     horses,
        "Best Odds": odds,
        "Win_Value": win_val,
        "Place_Value": (win_val*0.6).round(1)
    })
    df["Predicted Win %"]   = (1/df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6) + (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6) + (df["Place_Value"]*0.4)).round(1)
    df["Risk"]              = np.where(df["BetEdge Win %"]>25, "✅",
                                np.where(df["BetEdge Win %"]>15, "⚠️","❌"))
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ───── LIVE DATA FETCHER ─────
@st.cache_data(ttl=300)
def fetch_live_racecards(day: str):
    """
    Fetch today or tomorrow racecards from TheRacingAPI
    """
    date = datetime.utcnow().date()
    if day=="tomorrow":
        date += timedelta(days=1)
    params = {
        "region_codes": ["GB"],
        "date": date.strftime("%Y-%m-%d")
    }
    resp = requests.get(
        API_URL,
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        params=params,
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()

def live_to_df(raw: dict) -> pd.DataFrame:
    """
    Parse JSON from /racecards into our DataFrame schema.
    """
    rows = []
    for meeting in raw.get("meetings", []):
        course = meeting.get("course", {}).get("name","Unknown")
        for race in meeting.get("races", []):
            off = race.get("off","")[:5]
            for runner in race.get("runners", []):
                horse = runner.get("horse", runner.get("horse_id","Unknown"))
                # sometimes sp_dec is null → fallback random odds
                odds = (runner.get("sp_dec") or np.random.uniform(2,6))
                odds = float(odds)
                # placeholder value metrics—replace with real logic later
                win_val   = np.random.uniform(5,25)
                place_val = win_val * 0.6
                rows.append({
                    "Time":       off,
                    "Country":    "UK",        # API is region_code GB only
                    "Course":     course,
                    "Bookie":     "BestExchange",
                    "Horse":      horse,
                    "Best Odds":  round(odds,2),
                    "Win_Value":  round(win_val,1),
                    "Place_Value": round(place_val,1)
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # compute percentages & Edge scores
    df["Predicted Win %"]   = (1/df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"]              = np.where(df["BetEdge Win %"]>25, "✅",
                                np.where(df["BetEdge Win %"]>15, "⚠️","❌"))
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ───── Fallback logic ─────
def get_data(day: str, source: str):
    if source=="Live":
        try:
            raw = fetch_live_racecards(day)
            df  = live_to_df(raw)
            if not df.empty:
                return df, day
        except Exception as e:
            st.sidebar.error(f"Live API error: {e}")
    # if live fails or no races, fallback to mock
    return generate_mock_data(), "mock"

# ───── SIDEBAR ─────
with st.sidebar:
    st.title("🎛️ Data Source")
    source = st.radio("", ["Live", "Mock"])
    st.divider()
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=1
    )

# ───── LOAD DATA ─────
# we only do today/tomorrow filter on Horse Racing page
df_today, day_today = get_data("today", source)
df_tomo,  day_tomo  = get_data("tomorrow", source)

# ───── PAGES ─────
if selected=="Overview":
    st.title("📊 EdgeBet – Overview")
    st.metric("Data Source", source)
    st.metric("Today’s Runners", len(df_today))
    st.metric("Tomorrow’s Runners", len(df_tomo))
    st.metric("Top Edge % (Today)", f"{df_today['BetEdge Win %'].max()}%")
    st.metric("Last Update (UTC)", datetime.utcnow().strftime("%H:%M:%S"))

elif selected=="Horse Racing":
    st.title("🏇 Horse Racing – GB")
    day = st.selectbox("View Day", [("Today",df_today),("Tomorrow",df_tomo)],
                       format_func=lambda x: x[0])[1]
    st.write(f"Source: **{source}**  •  Day: **{day_today if day is df_today else day_tomo}**")
    # filters
    col1,col2,col3 = st.columns(3)
    with col1:
        bookie = st.selectbox("Bookmaker", ["All"]+day["Bookie"].unique().tolist())
    with col2:
        course = st.multiselect("Courses", ["All"]+day["Course"].unique().tolist(), default=["All"])
    with col3:
        mins, maxs = int(day["BetEdge Win %"].min()), int(day["BetEdge Win %"].max())
        win_range = st.slider("Edge% Range", mins, maxs, (mins, maxs))
    # apply filters
    df = day.copy()
    if bookie!="All":  df=df[df["Bookie"]==bookie]
    if "All" not in course: df=df[df["Course"].isin(course)]
    df = df[df["BetEdge Win %"].between(*win_range)]
    if df.empty:
        st.warning("No races match those filters.")
    else:
        view = st.radio("Display As", ["Charts","Tables"], horizontal=True)
        if view=="Charts":
            st.subheader("📊 Top 20 – Edge%")
            st.bar_chart(df.head(20).set_index("Horse")["BetEdge Win %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(df, use_container_width=True)

elif selected=="Football":
    st.title("⚽ Football – Coming Soon")
    st.info("We’ll hook up live football markets next.")

elif selected=="EdgeBrain":
    st.title("🧠 EdgeBrain Feed")
    st.write("Analytics layer & back-test interface coming in Phase 5!")

else:  # How It Works
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    - **Live vs Mock** data sourcing  
    - **Value Metrics** (Win & Place)  
    - **Edge Score** = 60% implied probability + 40% value metric  
    - **Risk Icons**: ✅⚠️❌  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
