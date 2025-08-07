import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="BetEdge Pro Tracker", layout="wide")
st.title("🏇 BetEdge Pro – UK Racing Value Tracker")

# ------------------- MOCK DATA -------------------
@st.cache_data(ttl=600)
def generate_mock_data():
    horses = [f"Horse {i}" for i in range(1, 31)]
    times = pd.date_range("14:00", periods=30, freq="10min").time
    courses = [f"Course {i%5+1}" for i in range(30)]

    df = pd.DataFrame({
        "Time": times,
        "Course": courses,
        "Horse": horses,
        "Best Odds": np.round(np.random.uniform(2.0, 8.0, size=30), 2),
        "Win_Value": np.round(np.random.uniform(5, 25, size=30), 1),
    })
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

df = generate_mock_data()

# ------------------- FILTERS -------------------
with st.sidebar:
    st.header("🔍 Filters")
    selected_courses = st.multiselect("Select Courses", df["Course"].unique(), default=df["Course"].unique())
    min_odds, max_odds = st.slider("Best Odds Range", 2.0, 10.0, (2.0, 8.0), step=0.1)
    min_betedge = st.slider("Minimum BetEdge Win %", 0, 100, 0)

filtered_df = df[
    (df["Course"].isin(selected_courses)) &
    (df["Best Odds"] >= min_odds) &
    (df["Best Odds"] <= max_odds) &
    (df["BetEdge Win %"] >= min_betedge)
].reset_index(drop=True)

# ------------------- TABLES -------------------
st.subheader("📋 Sorted BetEdge Rankings (Mock Data)")
tab1, tab2 = st.tabs(["🏆 Win Rankings", "🥈 Place Rankings"])

def color_val(v):
    if v > 20: return 'background-color:#58D68D;color:black'
    if v > 10: return 'background-color:#F9E79F;color:black'
    return 'background-color:#F5B7B1;color:black'

with tab1:
    st.dataframe(
        filtered_df.sort_values("BetEdge Win %", ascending=False)
        .style.applymap(color_val, subset=["BetEdge Win %"]),
        use_container_width=True
    )

with tab2:
    st.dataframe(
        filtered_df.sort_values("BetEdge Place %", ascending=False)
        .style.applymap(color_val, subset=["BetEdge Place %"]),
        use_container_width=True
    )

# ------------------- VISUAL -------------------
st.markdown("---")
st.subheader("📊 BetEdge Win vs Place Comparison (Top 10)")
top10 = filtered_df.sort_values("BetEdge Win %", ascending=False).head(10)
x = np.arange(len(top10["Horse"]))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, top10["BetEdge Win %"], width, label='Win %')
bars2 = ax.bar(x + width/2, top10["BetEdge Place %"], width, label='Place %')

ax.set_ylabel('%')
ax.set_title('Top 10 BetEdge: Win vs Place')
ax.set_xticks(x)
ax.set_xticklabels(top10["Horse"], rotation=45, ha="right")
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
st.caption("🔒 All data is mock data for demo purposes. Real-time integration pending.")
