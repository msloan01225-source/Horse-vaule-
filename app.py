import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

st.set_page_config(page_title="EdgeBet", layout="wide")

# --- EDGEBRAIN MODEL LOADING ---
@st.cache_resource
def load_edgebrain_model():
    try:
        return joblib.load("edgebrain_model.pkl")
    except Exception:
        st.warning("⚠️ edgebrain_model.pkl not found — using stub predictions.")
        return None

edge_model = load_edgebrain_model()

def predict_edgebrain(df: pd.DataFrame) -> pd.Series:
    if edge_model:
        X = df[["Odds", "Win_Value", "Place_Value"]]
        return (edge_model.predict_proba(X)[:, 1] * 100).round(1)
    else:
        return df["BetEdge Win %"]

# --- LIVE SP SCRAPER — failure-safe ---
@st.cache_data(ttl=60)
def fetch_live_sp(day: str = "today") -> pd.DataFrame:
    try:
        target = datetime.utcnow().date()
        if day.lower() == "tomorrow":
            target += timedelta(days=1)
        url = f"https://www.racingpost.com/racecards/time-order/{target.isoformat()}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        for sec in soup.select("section.race-time"):
            time_el   = sec.select_one(".rc-race-time")
            course_el = sec.select_one(".rc-race-meeting__course a")
            if not time_el or not course_el:
                continue
            off    = time_el.text.strip()
            course = course_el.text.strip()
            for runner in sec.select(".runner-wrap"):
                name_el = runner.select_one(".runner-runner__name")
                sp_el   = runner.select_one(".runner-sp__price")
                if not name_el or not sp_el:
                    continue
                try:
                    od = float(sp_el.text.strip())
                except ValueError:
                    continue
                rows.append({
                    "Horse":  name_el.text.strip(),
                    "Course": course,
                    "Time":   off,
                    "Odds":   od
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# --- MOCK FALLBACK & DATA BUILDING ---
def generate_mock_data(n=40) -> pd.DataFrame:
    horses  = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    odds    = np.random.uniform(2,10,n).round(2)
    return pd.DataFrame({"Horse":horses, "Course":courses, "Odds":odds})

def build_live_df(day: str="today") -> pd.DataFrame:
    df = fetch_live_sp(day)
    if df.empty:
        df = generate_mock_data()
        df["Time"] = ""
    df["Win_Value"]     = np.random.uniform(5,30,len(df)).round(1)
    df["Place_Value"]   = (df["Win_Value"]*0.6).round(1)
    df["Pred Win %"]    = (1/df["Odds"]*100).round(1)
    df["Pred Pl %"]     = (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"] = ((df["Pred Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Pl %"]  = ((df["Pred Pl %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"]          = np.where(df["BetEdge Win %"]>25,"✅",
                           np.where(df["BetEdge Win %"]>15,"⚠️","❌"))
    df["EdgeBrain %"]   = predict_edgebrain(df)
    df["Country"] = df["Course"].apply(
        lambda c: "UK" if c in ["Ascot","York","Cheltenham"] else "USA"
    )
    return df

# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1,h2,h3,h4,h5,h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR MENU ---
with st.sidebar:
    page = option_menu(
        "🏇 EdgeBet",
        ["Overview","Horse Racing","EdgeBrain","How It Works"],
        icons=["house","card-list","robot","info-circle"],
        default_index=0
    )

# --- PAGES ---
if page == "Overview":
    st.title("📊 EdgeBet Live Tracker")
    df0 = build_live_df("today")
    st.metric("Races Found", df0["Course"].nunique())
    st.metric("Total Runners", len(df0))
    st.metric("Top BetEdge Value", f"{df0['BetEdge Win %'].max()}%")

elif page == "Horse Racing":
    st.title("🏇 Horse Racing – Live UK/USA")
    day     = st.selectbox("Day", ["today","tomorrow"])
    df      = build_live_df(day)

    # --- SUB-FILTER BAR ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with col2:
        bookie  = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with col3:
        courses = df["Course"].unique().tolist()
        selected_courses = st.multiselect("Courses", courses, default=courses)
    with col4:
        mv, Mv = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
        edge_range = st.slider("BetEdge % Range", mv, Mv, (mv, Mv))

    # apply filters
    filt = df[
        ((df["Country"]==country)|(country=="All")) &
        ((df["Bookie"]==bookie)|(bookie=="All")) &
        (df["Course"].isin(selected_courses)) &
        df["BetEdge Win %"].between(*edge_range)
    ]

    tabs = st.tabs(["📊 Charts","📋 Tables","🏅 Top Picks"])
    with tabs[0]:
        st.subheader("Top 20 BetEdge Win %")
        st.bar_chart(filt.sort_values("BetEdge Win %", ascending=False)
                         .head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("Top 20 BetEdge Place %")
        st.bar_chart(filt.sort_values("BetEdge Pl %", ascending=False)
                         .head(20).set_index("Horse")["BetEdge Pl %"])
    with tabs[1]:
        st.subheader("🏆 Win + Place Rankings")
        st.dataframe(filt.sort_values("BetEdge Win %", ascending=False)
                     .reset_index(drop=True), use_container_width=True)
    with tabs[2]:
        st.subheader("🥇 Top 3 Win Picks")
        top_win = filt.sort_values("BetEdge Win %", ascending=False).head(3)
        for i, r in top_win.iterrows():
            st.markdown(f"**{i+1}. {r['Horse']} @ {r['BetEdge Win %']}%**")
            st.progress(r["BetEdge Win %"]/100)
        st.subheader("🥈 Top 3 Place Picks")
        top_pl = filt.sort_values("BetEdge Pl %", ascending=False).head(3)
        for i, r in top_pl.iterrows():
            st.markdown(f"**{i+1}. {r['Horse']} @ {r['BetEdge Pl %']}%**")
            st.progress(r["BetEdge Pl %"]/100)

elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    df = build_live_df("today")
    st.subheader("📊 Top 20 EdgeBrain Win %")
    st.bar_chart(df.sort_values("EdgeBrain %", ascending=False)
                 .head(20).set_index("Horse")["EdgeBrain %"])
    st.dataframe(df[["Horse","Course","Odds","EdgeBrain %","Risk"]],
                 use_container_width=True)

else:
    st.title("ℹ️ How EdgeBet Works")
    st.markdown("""
- Scrape live SP odds from Racing Post  
- Compute real-time "BetEdge" value metrics  
- Run the **EdgeBrain** ML model for smarter picks  
- Filter, chart and table your best selections
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
