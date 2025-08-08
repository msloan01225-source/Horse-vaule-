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
st.set_page_config(page_title="EdgeBet", layout="wide")

# ——— DARK THEME STYLING ———
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)


# ——— Fetch Live Racecards ———
@st.cache_data(ttl=300)
def fetch_live_data(day: str = "today"):
    # compute target date
    d = datetime.utcnow().date()
    if day.lower() == "tomorrow":
        d += timedelta(days=1)
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"date": d.isoformat(), "region": "GB"}
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# ——— Normalize to DataFrame ———
def live_to_df(raw) -> pd.DataFrame:
    rows = []
    for meeting in raw.get("meetings", []):
        course = meeting.get("course", {}).get("name", "Unknown")
        for race in meeting.get("races", []):
            off = race.get("off", "")[:5]
            for runner in race.get("runners", []):
                # ensure dict-like access
                if isinstance(runner, pd.Series):
                    runner = runner.to_dict()
                odds = runner.get("sp_dec")
                if odds is None:
                    continue
                odds = float(odds)
                implied = float(runner.get("implied_prob", 0)) * 100
                win_val = max((1 / odds * 100) - implied, 0)
                rows.append({
                    "RaceTime": f"{course} {off}",
                    "Course":    course,
                    "Horse":     runner.get("horse", "Unknown"),
                    "Bookie":    runner.get("bookmaker", "Unknown"),
                    "Odds":      odds,
                    "Win_Value": round(win_val, 1),
                    "Place_Value": round(win_val * 0.6, 1)
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Pred Win %"]       = (1 / df["Odds"] * 100).round(1)
    df["Pred Place %"]     = (df["Pred Win %"] * 0.6).round(1)
    df["BetEdge Win %"]    = ((df["Pred Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"]  = ((df["Pred Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"] > 25, "✅",
        np.where(df["BetEdge Win %"] > 15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)


# ——— Load Historical Performance ———
@st.cache_data(ttl=600)
def load_history() -> pd.DataFrame:
    return pd.read_csv("historical.csv")

hist = load_history()
# coerce numeric
hist["Stake"]  = pd.to_numeric(hist.get("Stake", []), errors="coerce").fillna(0)
hist["Return"] = pd.to_numeric(hist.get("Return", []), errors="coerce").fillna(0)
hist["Outcome"]= hist.get("Outcome", "").astype(str)

total_bets   = len(hist)
total_stake  = hist["Stake"].sum()
total_return = hist["Return"].sum()
profit       = total_return - total_stake
roi          = (profit / total_stake * 100) if total_stake else 0
wins         = hist[hist["Outcome"].str.lower() == "win"]
strike_rate  = (len(wins) / total_bets * 100) if total_bets else 0


# ——— Sidebar Navigation ———
with st.sidebar:
    page = option_menu(
        "🏠 Main Menu",
        ["Overview", "Horse Racing", "Performance", "EdgeBrain", "How It Works"],
        icons=["house","activity","bar-chart","robot","book"],
        menu_icon="cast",
        default_index=0
    )

# ——— PAGES ———
if page == "Overview":
    st.title("📊 Welcome to EdgeBet")
    live = live_to_df(fetch_live_data())
    st.metric("Total Runners",       len(live))
    top_val = live["BetEdge Win %"].max() if not live.empty else 0
    st.metric("Top Edge Value %",    f"{top_val:.1f}%")

elif page == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    live = live_to_df(fetch_live_data())

    # filters
    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country", ["All", "UK", "USA"])
    with col2:
        bookie = st.selectbox("Bookmaker", ["All"] + live["Bookie"].unique().tolist())
    with col3:
        courses = live["Course"].unique().tolist()
        course_filter = st.multiselect("Courses", courses, default=courses)

    minv = int(live["BetEdge Win %"].min()) if not live.empty else 0
    maxv = int(live["BetEdge Win %"].max()) if not live.empty else 0
    edge_range = st.slider("🎯 BetEdge Win % range", minv, maxv, (minv, maxv))

    view = st.radio("View", ["Charts","Tables"], horizontal=True)

    # apply
    flt = live[
        ((live["Course"].isin(course_filter))) &
        ((live["Bookie"] == bookie) | (bookie == "All")) &
        ((live["BetEdge Win %"] >= edge_range[0]) & (live["BetEdge Win %"] <= edge_range[1])) &
        ((live["Course"].isin(courses))) &
        ((live["Course"].map(lambda c: "UK" if c in ["Ascot","York","Cheltenham"] else "USA") == country) | (country=="All"))
    ].copy()

    if flt.empty:
        st.warning("No horses match your filters.")
    else:
        dfw = flt.sort_values("BetEdge Win %", ascending=False)
        dfp = flt.sort_values("BetEdge Place %", ascending=False)
        if view=="Charts":
            st.subheader("Top 20 – Win %")
            st.bar_chart(dfw.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 – Place %")
            st.bar_chart(dfp.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(dfw, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(dfp, use_container_width=True)

elif page == "Performance":
    st.title("📈 Back-Test Performance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Bets",      total_bets)
    c2.metric("Total Stake","£{:,.2f}".format(total_stake))
    c3.metric("Profit/Loss","£{:,.2f}".format(profit))
    c4.metric("ROI",       f"{roi:.1f}%")
    st.metric("Strike Rate", f"{strike_rate:.1f}%")
    st.subheader("Sample Historical Bets")
    st.dataframe(hist.head(20), use_container_width=True)

elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    live = live_to_df(fetch_live_data())
    st.dataframe(live[["Horse","Course","Odds","BetEdge Win %","Risk"]], use_container_width=True)

elif page == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet combines:
    - ✅ Implied Probability from SP odds  
    - 📊 Win & Place value metrics  
    - 🧠 Proprietary EdgeBrain simulation  
    - 🔎 Risk indicators & filters  

    All rolled up into your **BetEdge %**.
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

# ——— Footer ———
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
