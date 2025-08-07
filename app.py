import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="BetEdge Pro", layout="wide", initial_sidebar_state="expanded")
st.title("🏇 BetEdge Pro – UK Racing Value Tracker")

# --- Mock Data Generator ---
@st.cache_data(ttl=600)
def generate_mock_data(num=100):
    courses = ["Ascot", "York", "Newmarket", "Cheltenham", "Goodwood"]
    horses = [f"Horse {i}" for i in range(1, num + 1)]
    times = [f"{hour}:{minute:02d}" for hour in range(13, 18) for minute in (0, 15, 30, 45)]
    data = []

    for horse in horses:
        data.append({
            "Course": np.random.choice(courses),
            "Time": np.random.choice(times),
            "Horse": horse,
            "Best Odds": round(np.random.uniform(2, 8), 2),
            "Win_Value": round(np.random.uniform(5, 25), 1),
        })

    df = pd.DataFrame(data)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)

    return df

df = generate_mock_data()

# --- Sidebar Navigation and Filters ---
with st.sidebar:
    st.header(" Menu ")
    section = st.radio("", ["Overview", "Charts", "Settings"])

    st.markdown("---")
    st.header(" Filters ")
    selected_courses = st.multiselect("Course(s):", sorted(df["Course"].unique()), default=sorted(df["Course"].unique()))
    odds_min, odds_max = st.slider("Best Odds Range:", 2.0, 10.0, (2.0, 8.0), step=0.1)
    min_betedge = st.slider("Min BetEdge Win %:", 0, 100, 0)

    st.text_input("Horse Search:", key="horse_search")

# --- Apply Filters ---
filtered = df[
    df["Course"].isin(selected_courses) &
    df["Best Odds"].between(odds_min, odds_max)
]
if st.session_state.horse_search:
    filtered = filtered[filtered["Horse"].str.contains(st.session_state.horse_search, case=False)]

filtered = filtered[filtered["BetEdge Win %"] >= min_betedge]

# --- Bookmaker Odds (Mock) ---
def bookmaker_odds_top(horses):
    bookmakers = ["Bet365", "SkyBet", "Betfair"]
    odds_data = {bk: [round(np.random.uniform(1.5, 15), 2) for _ in horses] for bk in bookmakers}
    odds_data["Horse"] = horses
    return pd.DataFrame(odds_data)

# --- Show Section ---
if section == "Overview":
    st.subheader(" ⭐ Top BetEdge Value Horse")
    best = filtered.sort_values("BetEdge Win %", ascending=False).head(1).iloc[0]
    st.markdown(f"""
    **{best['Horse']}** @ {best['Course']} • {best['Time']}  
    **BetEdge Win %**: {best['BetEdge Win %']}%  •  **Best Odds**: {best['Best Odds']}
    """)

    # Display mock bookmaker odds for top 5
    top5 = filtered.sort_values("BetEdge Win %", ascending=False).head(5)
    if not top5.empty:
        st.subheader(" Bookmaker Odds (Mock)")
        st.dataframe(bookmaker_odds_top(top5["Horse"].tolist()), width=400)

elif section == "Charts":
    st.subheader(" BetEdge Charts")
    df_win = filtered.sort_values("BetEdge Win %", ascending=False)
    top10 = df_win.head(10)
    if not top10.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(top10))
        width = 0.35
        ax.bar(x - width/2, top10["BetEdge Win %"], width, label="Win %")
        ax.bar(x + width/2, top10["BetEdge Place %"], width, label="Place %")
        ax.set_xticks(x)
        ax.set_xticklabels(top10["Horse"], rotation=45, ha='right')
        ax.set_ylabel("%")
        ax.set_title("Top 10 BetEdge – Win vs Place")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

elif section == "Settings":
    st.subheader("⚙ App Settings")
    st.caption("Here you can adjust mock data parameters or integration configurations.")

# --- Shared Rankings View ---
st.markdown("---")
st.subheader(" Rankings Table")
df_rank = filtered.sort_values("BetEdge Win %", ascending=False)
st.dataframe(
    df_rank[["Course", "Time", "Horse", "Best Odds", "BetEdge Win %", "BetEdge Place %"]],
    use_container_width=True
)

# --- Footer ---
st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M UTC}  |  Mock data only.")
