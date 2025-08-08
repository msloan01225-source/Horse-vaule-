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

# ---- HISTORICAL CSV UPLOADER & METRICS ----
st.sidebar.markdown("### 📊 Backtest / Historical Data")
upload = st.sidebar.file_uploader("Upload historical.csv", type=["csv"])
if upload:
    hist = pd.read_csv(upload)
    st.sidebar.write("Preview:")
    st.sidebar.dataframe(hist.head(10))
    cols = hist.columns.tolist()
    st.sidebar.write("Detected columns:", cols)

    stake_col  = st.sidebar.selectbox("Stake column",  cols, index=cols.index("Stake") if "Stake" in cols else 0)
    return_col = st.sidebar.selectbox("Return/Profit col", cols, index=cols.index("Return") if "Return" in cols else 1)
    result_col = st.sidebar.selectbox("Result column",      cols, index=cols.index("Result") if "Result" in cols else 2)

    total_stake  = hist[stake_col].sum()
    total_return = hist[return_col].sum()
    profit       = total_return - total_stake
    roi          = (profit/total_stake*100) if total_stake else 0
    wins         = hist[hist[result_col].str.lower()=="win"].shape[0]
    strike_rate  = (wins/len(hist)*100) if len(hist) else 0

    # show on Overview
    st.sidebar.metric("🏦 Total Stake",    f"£{total_stake:,.2f}")
    st.sidebar.metric("💰 Total Return",   f"£{total_return:,.2f}")
    st.sidebar.metric("📈 ROI",            f"{roi:.1f}%")
    st.sidebar.metric("🎯 Strike Rate",    f"{strike_rate:.1f}%")
else:
    st.sidebar.info("Upload your historical.csv to enable backtest metrics.")

# ---- MOCK EDGE BRAIN DATA ----
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n)
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" 
                 for c in courses]
    bookies = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds = np.random.uniform(2,10,n).round(2)
    win_val = np.random.uniform(5,30,n).round(1)

    df = pd.DataFrame({
        "Horse": horses,
        "Course": courses,
        "Country": countries,
        "Bookie": bookies,
        "Odds": odds,
        "Win_Value": win_val,
        "Place_Value": (win_val*0.6).round(1)
    })
    df["Predicted Win %"]   = (1/df["Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"]>25, "✅",
                  np.where(df["BetEdge Win %"]>15, "⚠️", "❌"))
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ---- MAIN NAVIGATION ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# ---- OVERVIEW ----
if selected=="Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Cutting-edge value models for smarter betting.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners", len(df))
    st.metric("Top Edge Value", f"{df['BetEdge Win %'].max()}%")

# ---- HORSE RACING ----
elif selected=="Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    col1,col2,col3 = st.columns(3)
    with col1:
        country     = st.selectbox("Country",    ["All","UK","USA"])
    with col2:
        bookie      = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with col3:
        courses_all = df["Course"].unique().tolist()
        course_sel  = st.multiselect("Courses", courses_all, default=courses_all)

    min_val, max_val = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
    edge_range = st.slider("🎯 BetEdge Win % range", min_val, max_val, (min_val, max_val))
    view_mode  = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    filt = df[
        ((df["Country"]==country)|(country=="All")) &
        ((df["Bookie"]==bookie)|(bookie=="All")) &
        df["Course"].isin(course_sel) &
        df["BetEdge Win %"].between(*edge_range)
    ]

    win_df   = filt.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    place_df = filt.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if filt.empty:
        st.warning("No horses match filters.")
    else:
        if view_mode=="📊 Charts":
            st.subheader("Top 20 – BetEdge Win %")
            st.bar_chart(win_df.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 – BetEdge Place %")
            st.bar_chart(place_df.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win_df, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(place_df, use_container_width=True)

# ---- FOOTBALL ----
elif selected=="Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data—coming soon with EdgeBrain.")

# ---- EDGEBRAIN ----
elif selected=="EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("AI-enhanced simulated predictions.")
    st.dataframe(df[["Horse","Course","Odds","BetEdge Win %","Risk"]], use_container_width=True)

# ---- HOW IT WORKS ----
elif selected=="How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet combines:
    - ✅ Implied probability  
    - 📊 Win & place value  
    - 🧠 Proprietary EdgeBrain logic  
    - 🔎 Risk indicators  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # placeholder

st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
