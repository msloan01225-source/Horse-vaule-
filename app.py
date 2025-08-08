import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ─── PAGE CONFIG & THEME ────────────────────────────────────────────────────────
st.set_page_config(page_title="EdgeBet", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  body { background-color: #111111; color: white; }
  h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
  .css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ─── API CREDENTIALS ────────────────────────────────────────────────────────────
API_USER = st.secrets.get("racingapi_user", "edypVknQtk8n3artYstntbPu")
API_PW   = st.secrets.get("racingapi_pw",   "DIDUKRnNjVtP1tvQOpbcCGC7")

# ─── LIVE DATA FETCH WITH FALLBACK ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_live_data(day: str = "today"):
    date = datetime.utcnow().date() + (timedelta(days=1) if day=="tomorrow" else timedelta())
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"date": date.isoformat(), "region": "GB"}
    try:
        r = requests.get(url, auth=HTTPBasicAuth(API_USER, API_PW), params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"⚠️ Live API error: {e}")
        return None

def live_to_df(raw):
    rows = []
    if not raw or "meetings" not in raw:
        return pd.DataFrame()
    for m in raw["meetings"]:
        course = m.get("course",{}).get("name","")
        for race in m.get("races",[]):
            off = race.get("off","")[:5]
            for rnr in race.get("runners",[]):
                sp = rnr.get("sp_dec")
                if sp is None: 
                    continue
                odds = float(sp)
                implied = rnr.get("implied_prob",0)*100
                win_val = max((1/odds*100) - implied, 0)
                rows.append({
                    "RaceTime": f"{course} {off}",
                    "Horse": rnr.get("horse",""),
                    "Course": course,
                    "Bookie": rnr.get("bookmaker","Unknown"),
                    "Odds": odds,
                    "Win_Value": round(win_val,1),
                    "Place_Value": round(win_val*0.6,1)
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Pred Win %"]       = (1/df["Odds"]*100).round(1)
    df["Pred Place %"]     = (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"]    = ((df["Pred Win %"]*0.6) + (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]  = ((df["Pred Place %"]*0.6) + (df["Place_Value"]*0.4)).round(1)
    df["Risk"]             = np.where(df["BetEdge Win %"]>25, "✅",
                              np.where(df["BetEdge Win %"]>15, "⚠️","❌"))
    return df

# ─── MOCK DATA (fallback) ──────────────────────────────────────────────────────
def generate_mock_data(n=40):
    horses  = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n)
    bookies = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds    = np.random.uniform(2,10,n).round(2)
    win_val = np.random.uniform(5,30,n).round(1)
    df = pd.DataFrame({
        "Horse": horses,
        "Course": courses,
        "Bookie": bookies,
        "Odds": odds,
        "Win_Value": win_val
    })
    df["Place_Value"]      = (df["Win_Value"]*0.6).round(1)
    df["Pred Win %"]       = (1/df["Odds"]*100).round(1)
    df["Pred Place %"]     = (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"]    = ((df["Pred Win %"]*0.6) + (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]  = ((df["Pred Place %"]*0.6) + (df["Place_Value"]*0.4)).round(1)
    df["Risk"]             = np.where(df["BetEdge Win %"]>25, "✅",
                              np.where(df["BetEdge Win %"]>15, "⚠️","❌"))
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ─── HISTORICAL CSV UPLOAD ──────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_history(uploaded):
    if not uploaded:
        return None
    df = pd.read_csv(uploaded)
    df["Stake"]  = pd.to_numeric(df["Stake"],errors="coerce").fillna(0)
    df["Return"] = pd.to_numeric(df["Return"],errors="coerce").fillna(0)
    return df

# ─── SIDEBAR MENU ───────────────────────────────────────────────────────────────
with st.sidebar:
    tab = option_menu(
        "🏠 Main Menu",
        ["Overview", "Horse Racing", "Performance", "EdgeBrain", "How It Works"],
        icons=["house","activity","bar-chart","robot","book"],
        menu_icon="cast",
        default_index=0
    )

# ─── PAGE: Overview ────────────────────────────────────────────────────────────
if tab == "Overview":
    st.title("📊 Welcome to EdgeBet")
    raw = fetch_live_data("today")
    df_live = live_to_df(raw) if raw else pd.DataFrame()
    if df_live.empty:
        st.warning("Live feed unavailable. Showing mock data.")
        df_live = generate_mock_data()
    st.metric("Total Runners", len(df_live))
    st.metric("Top BetEdge Win %", f"{df_live['BetEdge Win %'].max():.1f}%")
    st.dataframe(df_live.head(5), use_container_width=True)

# ─── PAGE: Horse Racing ─────────────────────────────────────────────────────────
elif tab == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    raw = fetch_live_data("today")
    df_live = live_to_df(raw) if raw else generate_mock_data()

    # FILTERS
    c1, c2, c3 = st.columns(3)
    with c1:
        bookie = st.selectbox("Bookmaker", ["All"] + sorted(df_live["Bookie"].unique().tolist()))
    with c2:
        courses = sorted(df_live["Course"].unique().tolist())
        course_filter = st.multiselect("Courses", ["All"]+courses, default=["All"])
    with c3:
        mn, mx = int(df_live["BetEdge Win %"].min()), int(df_live["BetEdge Win %"].max())
        edge_range = st.slider("BetEdge Win % range", mn, mx, (mn, mx))

    # APPLY
    filt = df_live.copy()
    if bookie!="All": filt = filt[filt["Bookie"]==bookie]
    if "All" not in course_filter: filt = filt[filt["Course"].isin(course_filter)]
    filt = filt[filt["BetEdge Win %"].between(*edge_range)]

    # VIEW MODE
    view = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)
    win_df   = filt.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    place_df = filt.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if filt.empty:
        st.warning("No runners match these filters.")
    elif view.startswith("📊"):
        st.subheader("Top 10 BetEdge Win %")
        st.bar_chart(win_df.head(10).set_index("Horse")["BetEdge Win %"])
        st.subheader("Top 10 BetEdge Place %")
        st.bar_chart(place_df.head(10).set_index("Horse")["BetEdge Place %"])
    else:
        st.subheader("🏆 Win Rankings")
        st.dataframe(win_df, use_container_width=True)
        st.subheader("🥈 Place Rankings")
        st.dataframe(place_df, use_container_width=True)

# ─── PAGE: Performance ──────────────────────────────────────────────────────────
elif tab == "Performance":
    st.title("📈 Historical Performance")
    uploaded = st.file_uploader("Upload your historical.csv", type="csv")
    hist = load_history(uploaded)
    if hist is None:
        st.info("Drop your historical.csv above to see ROI, P&L & strike rate.")
    else:
        total_stake  = hist["Stake"].sum()
        total_return = hist["Return"].sum()
        profit       = total_return - total_stake
        roi          = (profit / total_stake * 100) if total_stake else 0
        strike_rate  = hist["Result"].eq("Win").mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stake",    f"£{total_stake:.2f}")
        col2.metric("Total Return",   f"£{total_return:.2f}")
        col3.metric("Profit / Loss",  f"£{profit:.2f}")
        col4.metric("ROI",            f"{roi:.1f}%")
        st.metric("Strike Rate",      f"{strike_rate:.1f}%")
        st.subheader("Sample Bets")
        st.dataframe(hist.head(20), use_container_width=True)

# ─── PAGE: EdgeBrain ───────────────────────────────────────────────────────────
elif tab == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    raw = fetch_live_data("today")
    df_live = live_to_df(raw) if raw else generate_mock_data()
    st.dataframe(df_live[["Horse","Course","BetEdge Win %","Risk"]], use_container_width=True)

# ─── PAGE: How It Works ─────────────────────────────────────────────────────────
elif tab == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
- **Implied Probability** from Odds  
- **Win & Place Value** Metrics  
- **EdgeBrain Simulation** Logic  
- **Risk Indicators** & Smart Filtering  
""")
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # replace with your final intro

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
