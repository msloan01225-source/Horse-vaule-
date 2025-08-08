import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ─── CONFIG & THEME ──────────────────────────────────────────────────────────
st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide")
st.markdown("""
<style>
  body { background-color: #111111; color: #EEE; }
  h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
  .css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ─── MOCK LIVE DATA & EDGE BRAIN ─────────────────────────────────────────────
def generate_live_data(n=40):
    rng = np.random.RandomState(42)
    horses   = [f"Horse {i+1}" for i in range(n)]
    courses  = rng.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    countries= ["UK" if c in ("Ascot","York","Cheltenham") else "USA" for c in courses]
    bookies  = rng.choice(["SkyBet","Bet365","Betfair"], n)
    odds     = rng.uniform(2,10,n).round(2)
    win_val  = rng.uniform(5,30,n).round(1)

    df = pd.DataFrame({
      "Horse": horses,
      "Course": courses,
      "Country": countries,
      "Bookie": bookies,
      "Odds": odds,
      "Win_Value": win_val,
      "Place_Value": (win_val*0.6).round(1),
    })
    df["Pred Win %"]   = (1/df["Odds"]*100).round(1)
    df["Pred Place %"] = (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"]   = ((df["Pred Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"] = ((df["Pred Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"]>25, "✅",
                  np.where(df["BetEdge Win %"]>15, "⚠️","❌"))
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

live_df = generate_live_data()

# ─── SIDEBAR NAV ─────────────────────────────────────────────────────────────
with st.sidebar:
    page = option_menu(
      "🏠 Main Menu",
      ["Overview","Horse Racing","Performance","EdgeBrain","How It Works"],
      icons = ["house","activity","bar-chart-line","robot","book"],
      menu_icon="cast", default_index=0
    )

# ─── OVERVIEW ────────────────────────────────────────────────────────────────
if page=="Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Cutting-edge value & prediction engine for UK & USA racing.")
    st.metric("Active Sports", 2)
    st.metric("Live Runners", len(live_df))
    st.metric("Top EdgeWin %", f"{live_df['BetEdge Win %'].max()}%")

# ─── HORSE RACING ────────────────────────────────────────────────────────────
elif page=="Horse Racing":
    st.title("🏇 Horse Racing – Live Mock Data")
    c1,c2,c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with c2:
        bookie  = st.selectbox("Bookmaker", ["All"]+live_df["Bookie"].unique().tolist())
    with c3:
        courses = live_df["Course"].unique().tolist()
        course_filter = st.multiselect("Courses", courses, default=courses)
    # slider
    lo,hi = int(live_df["BetEdge Win %"].min()), int(live_df["BetEdge Win %"].max())
    thr = st.slider("🎯 BetEdge Win % ≥", lo, hi, lo)
    # view mode
    vm = st.radio("View", ["📊 Charts","📋 Tables"], horizontal=True)

    filt = live_df[
      ((live_df["Country"]==country)|(country=="All")) &
      ((live_df["Bookie"]==bookie)|(bookie=="All")) &
      (live_df["Course"].isin(course_filter)) &
      (live_df["BetEdge Win %"]>=thr)
    ]
    win = filt.sort_values("BetEdge Win %",ascending=False).reset_index(drop=True)
    plc = filt.sort_values("BetEdge Place %",ascending=False).reset_index(drop=True)

    if filt.empty:
        st.warning("No horses match those filters.")
    else:
        if vm=="📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(win.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(plc.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(plc, use_container_width=True)

# ─── PERFORMANCE (STEP 3) ─────────────────────────────────────────────────────
elif page=="Performance":
    st.title("📈 Historical Back-Test Performance")
    # try auto-load historical.csv
    if os.path.exists("historical.csv"):
        hist = pd.read_csv("historical.csv")
    else:
        hist = st.file_uploader("Upload your historical.csv", type="csv")
        if hist: hist = pd.read_csv(hist)

    if isinstance(hist, pd.DataFrame) and not hist.empty:
        # coerce numeric
        hist["Stake"]  = pd.to_numeric(hist.get("Stake",0), errors="coerce").fillna(0)
        hist["Return"] = pd.to_numeric(hist.get("Return",0), errors="coerce").fillna(0)
        # metrics
        B = len(hist)
        S = hist["Stake"].sum()
        R = hist["Return"].sum()
        P = R - S
        ROI = (P/S*100) if S else 0
        win_ct = hist[hist["Outcome"].str.lower()=="win"].shape[0]
        SR  = (win_ct/B*100) if B else 0

        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Total Bets",        f"{B}")
        col2.metric("Total Stake",       f"£{S:,.2f}")
        col3.metric("Total Return",      f"£{R:,.2f}")
        col4.metric("Profit/Loss",       f"£{P:,.2f}")
        col5.metric("ROI",               f"{ROI:.1f}%")
        st.metric("Strike Rate",         f"{SR:.1f}%")

        # threshold backtest
        if "BetEdge Win %" in hist.columns:
            th = st.slider("▶️ Back-test Thr (%)", 0, 100, 50)
            subset = hist[hist["BetEdge Win %"]>=th]
            sb = len(subset); ss=subset["Stake"].sum(); srn=subset["Return"].sum()
            sp = srn-ss; sroi=(sp/ss*100) if ss else 0; sw=(subset["Outcome"].str.lower()=="win").mean()*100
            st.write(f"## Bets ≥ {th}% EdgeWin")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Bets", sb)
            m2.metric("Stake", f"£{ss:,.2f}")
            m3.metric("Profit/Loss", f"£{sp:,.2f}")
            m4.metric("ROI", f"{sroi:.1f}%  |  SR {sw:.1f}%")

        # cumulative P&L chart
        hist["PnL"] = hist["Return"] - hist["Stake"]
        hist["CumPnL"] = hist["PnL"].cumsum()
        st.subheader("Cumulative P&L")
        st.line_chart(hist.set_index(pd.to_datetime(hist.get("Date",hist.index)))["CumPnL"])

        st.subheader("Sample Historical Bets")
        st.dataframe(hist.head(20), use_container_width=True)

    else:
        st.info("No historical data. Drop in your `historical.csv` to see performance.")

# ─── EDGEBRAIN ───────────────────────────────────────────────────────────────
elif page=="EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("Integrate your refined model here.")
    st.dataframe(live_df[["Horse","Course","Odds","BetEdge Win %","Risk"]], use_container_width=True)

# ─── HOW IT WORKS ────────────────────────────────────────────────────────────
else:
    st.title("📚 How EdgeBet Works")
    st.markdown("""
      EdgeBet combines:
      - ✅ Implied Probability  
      - 📊 Win & Place Value  
      - 🧠 Proprietary Simulation  
      - 🔎 Risk & Filtering  
      → **Your Edge % over the market**  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
