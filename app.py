import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random

# ------------- Login Section ----------------
st.set_page_config(page_title="BetEdge Pro", layout="wide")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login to BetEdge")
    password = st.text_input("Enter password", type="password")
    if password == "edge2025":
        st.session_state.authenticated = True
        st.experimental_rerun()
    else:
        st.stop()

# ------------- Sidebar Navigation ----------------
st.sidebar.title("🔍 Navigation")
sport = st.sidebar.selectbox("Select Sport", ["🏇 Horse Racing", "⚽ Football"])

if sport == "🏇 Horse Racing":
    country = st.sidebar.radio("Select Region", ["🇬🇧 UK", "🇺🇸 USA"])
    bookie_filter = st.sidebar.multiselect(
        "Bookmaker Filter",
        ["Bet365", "William Hill", "Paddy Power", "Betfair", "Coral"],
        default=["Bet365", "William Hill"]
    )
    min_value = st.sidebar.slider("Minimum BetEdge Win %", 0, 100, 10)
    view_mode = st.sidebar.radio("View Mode", ["📊 Charts", "📋 Tables"])
    date_option = st.sidebar.radio("Day", ["Today", "Tomorrow"])

    # ------------- Mock Data ----------------
    def generate_mock_data(region="UK"):
        courses = ["Ascot", "Newbury", "Aintree", "York", "Kempton"] if region == "UK" else ["Gulfstream", "Belmont", "Santa Anita", "Del Mar"]
        horses = [f"Horse {i}" for i in range(1, 31)]
        data = []
        for h in horses:
            course = random.choice(courses)
            odds = round(random.uniform(2, 8), 2)
            win_val = round(random.uniform(5, 30), 1)
            place_val = round(win_val * 0.6, 1)
            bookie = random.choice(["Bet365", "William Hill", "Paddy Power", "Betfair", "Coral"])
            bet_win = round((1 / odds * 100 * 0.6) + (win_val * 0.4), 1)
            bet_place = round((1 / odds * 100 * 0.6 * 0.6) + (place_val * 0.4), 1)
            data.append({
                "Horse": h,
                "Course": course,
                "Bookie": bookie,
                "Best Odds": odds,
                "Win_Value": win_val,
                "Place_Value": place_val,
                "BetEdge Win %": bet_win,
                "BetEdge Place %": bet_place
            })
        df = pd.DataFrame(data)
        return df

    df = generate_mock_data("UK" if country == "🇬🇧 UK" else "USA")
    df = df[df["Bookie"].isin(bookie_filter)]
    df = df[df["BetEdge Win %"] >= min_value]

    # ------------- Display Content ----------------
    st.title(f"{country} Horse Racing – BetEdge Value Tracker")
    st.markdown(f"#### 📅 {date_option} | Bookies: {', '.join(bookie_filter)} | Min Win %: {min_value}+")
    st.markdown(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    def color_val(v):
        if v > 20: return 'background-color:#58D68D;color:black'
        if v > 10: return 'background-color:#F9E79F;color:black'
        return 'background-color:#F5B7B1;color:black'

    if view_mode == "📋 Tables":
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win.style.applymap(color_val, subset=["BetEdge Win %"]))
        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place.style.applymap(color_val, subset=["BetEdge Place %"]))
    else:
        st.subheader("📊 BetEdge Win % Chart (Top 20)")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("📊 BetEdge Place % Chart (Top 20)")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])

elif sport == "⚽ Football":
    st.title("⚽ Football Betting Dashboard (Coming Soon)")
    st.info("Football mock market will be added here with value indicators, odds comparison, and bookmaker filters.")
