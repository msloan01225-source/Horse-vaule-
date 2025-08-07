import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="BetEdge - Horse Racing", layout="wide")
st.title("🏇 BetEdge – Horse Racing Value Tracker")

# --- Generate Mock Data ---
@st.cache_data(ttl=600)
def generate_mock_data(n=50):
    np.random.seed(42)
    courses = ["Ascot", "York", "Newmarket", "Cheltenham", "Aintree", "Sandown"]
    bookies = ["Bet365", "SkyBet", "William Hill", "PaddyPower"]
    track_types = ["Flat", "Jump"]
    race_classes = ["Class 1", "Class 2", "Class 3", "Class 4"]

    data = []
    for i in range(n):
        course = np.random.choice(courses)
        horse = f"Horse {i+1}"
        bookie = np.random.choice(bookies)
        track = np.random.choice(track_types)
        race_class = np.random.choice(race_classes)
        form = "".join(np.random.choice(list("1234567890-"), size=5))
        best_odds = round(np.random.uniform(2, 15), 2)
        win_val = round(np.random.uniform(0, 25), 1)
        place_val = round(win_val * np.random.uniform(0.5, 0.8), 1)
        pred_win = round(100 / best_odds, 1)
        pred_place = round(pred_win * 0.6, 1)
        betedge_win = round((pred_win * 0.6 + win_val * 0.4), 1)
        betedge_place = round((pred_place * 0.6 + place_val * 0.4), 1)

        data.append({
            "Time": (datetime.now() + timedelta(minutes=30 * i)).strftime("%H:%M"),
            "Course": course,
            "Horse": horse,
            "Form": form,
            "Bookie": bookie,
            "Track": track,
            "Race Class": race_class,
            "Best Odds": best_odds,
            "Win_Value": win_val,
            "Place_Value": place_val,
            "Predicted Win %": pred_win,
            "Predicted Place %": pred_place,
            "BetEdge Win %": betedge_win,
            "BetEdge Place %": betedge_place
        })

    return pd.DataFrame(data)

df = generate_mock_data()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("📊 Filters")
    selected_course = st.multiselect("Course", sorted(df["Course"].unique()), default=df["Course"].unique())
    selected_bookie = st.multiselect("Bookie", sorted(df["Bookie"].unique()), default=df["Bookie"].unique())
    selected_track = st.multiselect("Track Type", sorted(df["Track"].unique()), default=df["Track"].unique())
    selected_class = st.multiselect("Race Class", sorted(df["Race Class"].unique()), default=df["Race Class"].unique())

# --- Apply Filters ---
filtered_df = df[
    df["Course"].isin(selected_course) &
    df["Bookie"].isin(selected_bookie) &
    df["Track"].isin(selected_track) &
    df["Race Class"].isin(selected_class)
]

# --- Display Interface ---
if filtered_df.empty:
    st.warning("No races match your filters.")
else:
    view = st.radio("Select View Mode", ["📈 Charts", "📋 Tables"])
    st.caption(f"Showing **{len(filtered_df)}** horses | Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    if view == "📋 Tables":
        st.subheader("🏆 Win Rankings")
        st.dataframe(filtered_df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True))

        st.subheader("🥈 Place Rankings")
        st.dataframe(filtered_df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True))

    else:
        st.subheader("📊 Top 20 by BetEdge Win %")
        win_top = filtered_df.sort_values("BetEdge Win %", ascending=False).head(20)
        st.bar_chart(win_top.set_index("Horse")["BetEdge Win %"])

        st.subheader("📊 Top 20 by BetEdge Place %")
        place_top = filtered_df.sort_values("BetEdge Place %", ascending=False).head(20)
        st.bar_chart(place_top.set_index("Horse")["BetEdge Place %"])
