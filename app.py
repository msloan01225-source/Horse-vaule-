import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu

st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")

# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- LOAD & AGGREGATE HISTORICAL DATA ----
@st.cache_data(ttl=600)
def load_historical(path="historical.csv"):
    # historical.csv should have columns: Horse, Date, Odds, Result (Win/Place/Show/None), Payout
    hist = pd.read_csv(path)
    # Compute win rate and ROI per horse
    summary = hist.groupby("Horse").agg(
        Total_Runs = ("Result", "count"),
        Wins        = ("Result", lambda x: (x=="Win").sum()),
        Total_Payout= ("Payout", "sum")
    ).reset_index()
    summary["Win_Rate %"] = (summary["Wins"] / summary["Total_Runs"] * 100).round(1)
    # ROI = (Total Payout - Total Stake) / Total Stake. Assume stake=1 per run:
    summary["ROI %"] = ((summary["Total_Payout"] - summary["Total_Runs"]) / summary["Total_Runs"] * 100).round(1)
    return summary

hist_summary = load_historical()

# ---- MOCK EDGE BRAIN (OR REPLACE WITH LIVE FETCH) ----
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(["Ascot", "York", "Cheltenham", "Churchill Downs", "Santa Anita"], n)
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" for c in courses]
    bookies = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds = np.random.uniform(2, 10, n).round(2)
    win_val = np.random.uniform(5, 30, n).round(1)
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
    df["Risk"]              = np.where(df["BetEdge Win %"]>25, "✅",
                                np.where(df["BetEdge Win %"]>15, "⚠️","❌"))
    # Merge in historical metrics
    df = df.merge(hist_summary[["Horse","Win_Rate %","ROI %"]], on="Horse", how="left")
    df[["Win_Rate %","ROI %"]] = df[["Win_Rate %","ROI %"]].fillna(0)  # defaults for new horses
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ---- MAIN MENU ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","EdgeBrain","How It Works"],
        icons=['house','horse','robot','book'],
        menu_icon="cast", default_index=0
    )

# ---- OVERVIEW ----
if selected=="Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Cutting-edge value models + historical performance.")
    st.metric("Active Sports","1 (Horse Racing)")
    st.metric("Total Runners", len(df))
    st.metric("Top Edge Win %", f"{df['BetEdge Win %'].max()}%")
    st.metric("Top Historical Win Rate", f"{df['Win_Rate %'].max()}%")

# ---- HORSE RACING ----
elif selected=="Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")

    # Filters
    c1,c2,c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with c2:
        bookie  = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with c3:
        course_filter = st.multiselect(
            "Courses", df["Course"].unique(), default=df["Course"].unique()
        )

    # BetEdge slider
    lo,hi = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
    betedge_range = st.slider("BetEdge Win % range", lo,hi,(lo,hi))

    view = st.radio("View", ["📊 Charts","📋 Tables"], horizontal=True)

    # Apply
    filt = df[
        ((df.Country==country)|(country=="All")) &
        ((df.Bookie==bookie)|(bookie=="All")) &
        (df.Course.isin(course_filter)) &
        df["BetEdge Win %"].between(*betedge_range)
    ]
    win_df   = filt.sort_values("BetEdge Win %", ascending=False)
    place_df = filt.sort_values("BetEdge Place %", ascending=False)

    if filt.empty:
        st.warning("No horses match filters.")
    else:
        if view.startswith("📊"):
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(win_df.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Historical Win Rate (%)")
            st.bar_chart(win_df.head(20).set_index("Horse")["Win_Rate %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win_df, use_container_width=True)
            st.subheader("🥇 Historical Metrics")
            st.dataframe(win_df[["Horse","Win_Rate %","ROI %"]], use_container_width=True)

# ---- EDGEBRAIN ----
elif selected=="EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("Overlay of live value + past performance.")
    st.dataframe(df[["Horse","Odds","BetEdge Win %","Win_Rate %","ROI %","Risk"]], use_container_width=True)

# ---- HOW IT WORKS ----
elif selected=="How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
- **Live Value Model**: Combines implied probability + value metrics  
- **Historical Engine**: Win rate & ROI from past data  
- **EdgeBrain**: Hybrid score & risk flags  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

# ---- FOOTER ----
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
