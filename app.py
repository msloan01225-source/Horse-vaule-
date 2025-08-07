import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ----------- App Config -----------
st.set_page_config(page_title="BetEdge Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

# ----------- Sidebar Navigation -----------
menu = st.sidebar.radio("🔍 Navigate", ["🏠 Home", "🐎 Horse Racing", "🌍 All Sports"])

# ----------- Mock Data Generator -----------
def generate_mock_data(n=100):
    np.random.seed(42)
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(["Ascot", "Newmarket", "Cheltenham", "Aintree", "York"], n)
    times = np.random.choice(["13:00", "14:30", "15:15", "16:45", "17:30"], n)
    odds = np.round(np.random.uniform(2, 10, n), 2)
    win_val = np.round(np.random.uniform(5, 25, n), 1)
    place_val = np.round(win_val * 0.6, 1)

    df = pd.DataFrame({
        "Time": times,
        "Course": courses,
        "Horse": horses,
        "Best Odds": odds,
        "Win_Value": win_val,
        "Place_Value": place_val
    })

    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)

    return df

# ----------- Filters -----------
def apply_filters(df):
    with st.sidebar.expander("🔧 Filters", expanded=True):
        selected_course = st.multiselect("🏟️ Filter by Course", sorted(df["Course"].unique()))
        min_odds, max_odds = st.slider("💰 Filter by Odds", 1.0, 15.0, (2.0, 10.0), step=0.5)
        df_filtered = df.copy()
        if selected_course:
            df_filtered = df_filtered[df_filtered["Course"].isin(selected_course)]
        df_filtered = df_filtered[(df_filtered["Best Odds"] >= min_odds) & (df_filtered["Best Odds"] <= max_odds)]
        return df_filtered

# ----------- Table Colouring -----------
def color_val(val):
    if val > 20:
        return 'background-color:#58D68D;color:black'
    elif val > 10:
        return 'background-color:#F9E79F;color:black'
    else:
        return 'background-color:#F5B7B1;color:black'

# ----------- Horse Racing View -----------
def horse_racing_view():
    df = generate_mock_data()
    df = apply_filters(df)

    if df.empty:
        st.warning("No races match the selected filters.")
        return

    st.markdown(f"**Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    view = st.radio("📊 View Mode", ["Charts", "Tables"], horizontal=True)

    df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if view == "Charts":
        st.subheader("📈 Top 20 BetEdge Win %")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])

        st.subheader("📈 Top 20 BetEdge Place %")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])

    else:
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win[["Time", "Course", "Horse", "Best Odds", "Win_Value", "Predicted Win %", "BetEdge Win %"]]
                     .style.applymap(color_val, subset=["BetEdge Win %"]))

        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place[["Time", "Course", "Horse", "Best Odds", "Place_Value", "Predicted Place %", "BetEdge Place %"]]
                     .style.applymap(color_val, subset=["BetEdge Place %"]))

# ----------- All Sports View (Placeholder) -----------
def all_sports_view():
    st.markdown("### 🌍 All Sports Search (Coming Soon)")
    st.info("You’ll be able to explore Football, Tennis, and more. Stay tuned!")

# ----------- Home Page -----------
def home_view():
    st.markdown("""
    ## 👋 Welcome to BetEdge
    **Beat the bookies using data-driven betting value.**

    - 🧠 Smart probability models  
    - 🔎 Real-time filtering by odds & course  
    - 📊 Visual ranking of best bets  
    - 📍 Works better than BookieBashing 😉

    👉 Select a sport from the sidebar to get started.
    """)

# ----------- View Routing -----------
if menu == "🏠 Home":
    home_view()
elif menu == "🐎 Horse Racing":
    horse_racing_view()
elif menu == "🌍 All Sports":
    all_sports_view()
