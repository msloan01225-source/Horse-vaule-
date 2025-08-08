import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="EdgeBet – Phase 3",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- MOCK EDGE BRAIN FOR HORSE RACING ----
def generate_horse_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
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
    df["Predicted Win %"] = (1/df["Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"]   = (df["Predicted Win %"]*0.6 + df["Win_Value"]*0.4).round(1)
    df["BetEdge Place %"] = (df["Predicted Place %"]*0.6 + df["Place_Value"]*0.4).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"]>25, "✅",
        np.where(df["BetEdge Win %"]>15, "⚠️","❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ---- MOCK EDGE BRAIN FOR FOOTBALL ----
def generate_football_data(n=30):
    teams = [f"Team {i+1}" for i in range(2*n)]
    fixtures = [(teams[i], teams[i+1]) for i in range(0,2*n,2)]
    leagues = np.random.choice(["Premier League","Championship","La Liga","Serie A"], n)
    odds_home = np.random.uniform(1.5,4,n).round(2)
    odds_draw = np.random.uniform(2.5,5,n).round(2)
    odds_away = np.random.uniform(1.5,4,n).round(2)
    # Implied probabilities
    implied_win = (1/odds_home*100).round(1)
    implied_draw= (1/odds_draw*100).round(1)
    implied_away= (1/odds_away*100).round(1)
    # EdgeBrain score = weighted sum of best implied + league factor
    edge_score = np.maximum.reduce([implied_win, implied_draw, implied_away]) * np.random.uniform(0.6,0.8,n)
    df = pd.DataFrame({
        "Fixture": [f"{h} vs {a}" for h,a in fixtures],
        "League": leagues,
        "Home Odds": odds_home,
        "Draw Odds": odds_draw,
        "Away Odds": odds_away,
        "Edge Score %": edge_score.round(1)
    })
    return df.sort_values("Edge Score %", ascending=False).reset_index(drop=True)

# generate once
horse_df = generate_horse_data()
foot_df  = generate_football_data()

# ---- MAIN MENU ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=["house","activity","soccer","robot","book"],
        menu_icon="cast",
        default_index=0
    )

# ---- OVERVIEW ----
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge with our cutting-edge value models.")
    st.metric("Sports Available", 2)
    st.metric("Horse Runners", len(horse_df))
    st.metric("Football Matches", len(foot_df))
    st.metric("Top Horse Edge", f"{horse_df['BetEdge Win %'].max()}%")
    st.metric("Top Foot Edge", f"{foot_df['Edge Score %'].max()}%")

# ---- HORSE RACING ----
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    c1,c2,c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with c2:
        bookie = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with c3:
        courses = horse_df["Course"].unique().tolist()
        course_filter = st.multiselect("Courses", courses, default=courses)
    e_min,e_max = int(horse_df["BetEdge Win %"].min()), int(horse_df["BetEdge Win %"].max())
    edge_range = st.slider("BetEdge Win %", e_min,e_max,(e_min,e_max))
    view = st.radio("View As", ["Charts","Tables"], horizontal=True)

    filt = horse_df[
        ((horse_df["Country"]==country)|(country=="All")) &
        ((horse_df["Bookie"]==bookie)|(bookie=="All")) &
        horse_df["Course"].isin(course_filter) &
        horse_df["BetEdge Win %"].between(*edge_range)
    ]
    if filt.empty:
        st.warning("No horses match filters.")
    else:
        if view=="Charts":
            st.subheader("Top 20 Horse Edge %")
            st.bar_chart(filt.head(20).set_index("Horse")["BetEdge Win %"])
        else:
            st.subheader("🏆 Horse Win Rankings")
            st.dataframe(filt, use_container_width=True)

# ---- FOOTBALL ----
elif selected == "Football":
    st.title("⚽ Football Value Picks")
    league = st.selectbox("League", ["All"]+foot_df["League"].unique().tolist())
    score_min = float(foot_df["Edge Score %"].min())
    score_max = float(foot_df["Edge Score %"].max())
    score_range = st.slider("Edge Score %", score_min, score_max, (score_min,score_max))
    view = st.radio("View As", ["Charts","Tables"], horizontal=True)

    fdf = foot_df[
        ((foot_df["League"]==league)|(league=="All")) &
        foot_df["Edge Score %"].between(*score_range)
    ]
    if fdf.empty:
        st.warning("No matches match filters.")
    else:
        if view=="Charts":
            st.subheader("Top 20 Football Edge %")
            st.bar_chart(fdf.head(20).set_index("Fixture")["Edge Score %"])
        else:
            st.subheader("⚽ Football Value Table")
            st.dataframe(fdf, use_container_width=True)

# ---- EDGEBRAIN ----
elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("All sports combined EdgeBrain insights.")
    combined = (
        horse_df[["Horse","Course","BetEdge Win %","Risk"]]
        .rename(columns={"Horse":"Name","Course":"Context","BetEdge Win %":"Score"})
        .assign(Sport="Horse")
        .append(
            foot_df[["Fixture","League","Edge Score %"]]
            .rename(columns={"Fixture":"Name","League":"Context","Edge Score %":"Score"})
            .assign(Sport="Football"),
            ignore_index=True
        )
    ).sort_values("Score",ascending=False).reset_index(drop=True)
    st.dataframe(combined, use_container_width=True)

# ---- HOW IT WORKS ----
elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
**EdgeBet** combines:  
- ✅ Implied Probability (Odds)  
- 📊 Value Metrics (Win & Place or Draw)  
- 🧠 Proprietary EdgeBrain Simulations  
- 🔎 Risk Indicators & Smart Filters  
""")
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
