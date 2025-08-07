import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu
import joblib

# ---- LOAD YOUR TRAINED EdgeBrain MODEL (if you have one) ----
# If you don’t yet have a real model file, you can omit these two lines
# and the “EdgeBrain Win %” column will simply fall back to the mock generator.
# model = joblib.load("edgebrain_model.pkl")

st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")

# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- MOCK EDGE BRAIN DATA GENERATOR ----
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot", "York", "Cheltenham", "Churchill Downs", "Santa Anita"], n
    )
    countries = ["UK" if c in ["Ascot", "York", "Cheltenham"] else "USA" for c in courses]
    bookies = np.random.choice(["SkyBet", "Bet365", "Betfair"], n)
    odds = np.random.uniform(2, 10, n).round(2)
    win_val = np.random.uniform(5, 30, n).round(1)

    df = pd.DataFrame({
        "Horse": horses,
        "Course": courses,
        "Country": countries,
        "Bookie": bookies,
        "Odds": odds,
        "Win_Value": win_val,
        "Place_Value": (win_val * 0.6).round(1)
    })

    # BetEdge calculations
    df["Predicted Win %"]   = (1 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)

    # Uncomment below if you have a real model
    # features = df[["Odds","Win_Value","Place_Value"]]
    # df["EdgeBrain Win %"] = (model.predict_proba(features)[:,1]*100).round(1)
    # df["Risk"] = np.where(df["EdgeBrain Win %"]>60,"✅",
    #                np.where(df["EdgeBrain Win %"]>40,"⚠️","❌"))

    # For now use BetEdge Win % as our Risk proxy:
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅",
                   np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))

    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ---- MAIN MENU ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","How It Works"],
        icons=['house','activity','soccer','robot','book'],
        menu_icon="cast", default_index=0
    )

# ---- OVERVIEW ----
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge value models.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners",  len(df))
    st.metric("Top BetEdge Win %", f"{df['BetEdge Win %'].max()}%")

# ---- HORSE RACING ----
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")

    col1, col2, col3 = st.columns(3)
    with col1:
        country       = st.selectbox("Country", ["All","UK","USA"])
    with col2:
        bookie        = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with col3:
        course_list   = df["Course"].unique().tolist()
        course_filter = st.multiselect("Courses", course_list, default=course_list)

    min_val, max_val = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
    edge_range = st.slider("BetEdge Win % range", min_val, max_val, (min_val, max_val))

    mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    filtered = df[
        ((df["Country"] == country) | (country=="All")) &
        ((df["Bookie"]  == bookie)  | (bookie =="All")) &
        (df["Course"].isin(course_filter)) &
        (df["BetEdge Win %"].between(*edge_range))
    ]

    win_df   = filtered.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    place_df = filtered.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if filtered.empty:
        st.warning("No runners match those filters.")
    else:
        if mode.startswith("📊"):
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(win_df.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(place_df.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win_df, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(place_df, use_container_width=True)

# ---- FOOTBALL ----
elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data – EdgeBrain coming soon.")

# ---- EDGEBRAIN ----
elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("Model probabilities (mock) + risk flags")
    st.dataframe(df[["Horse","Course","BetEdge Win %","Risk"]], use_container_width=True)

# ---- HOW IT WORKS ----
elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    - ✅ Implied Probability from Odds  
    - 📊 Win & Place Value Metrics  
    - 🔎 Proprietary BetEdge Calculation  
    - 🧠 (soon) EdgeBrain ML Model  

    Gives you a **BetEdge %** and risk flags to inform your staking.
    """)
    # If you have a hosted intro video, swap in your URL:
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

# ---- FOOTER ----
st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
