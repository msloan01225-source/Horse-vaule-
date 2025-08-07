import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------- Mock Data Generator --------
def generate_mock_data(day="today"):
    np.random.seed(42)
    courses = ["Ascot", "Chelmsford", "Lingfield", "York", "Newbury", "Kempton", "Ayr"]
    horses = [f"Horse {i}" for i in range(1, 51)]
    rows = []
    for horse in horses:
        course = np.random.choice(courses)
        odds = np.round(np.random.uniform(2, 10), 2)
        win_val = np.round(np.random.uniform(5, 25), 1)
        place_val = np.round(win_val * 0.6, 1)
        time = f"{np.random.randint(12, 18)}:{np.random.choice(['00', '15', '30', '45'])}"
        rows.append({
            "Time": time,
            "Course": course,
            "Horse": horse,
            "Best Odds": odds,
            "Win_Value": win_val,
            "Place_Value": place_val
        })
    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

# -------- Streamlit UI --------
st.set_page_config(page_title="BetEdge Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

# --- Sidebar Controls ---
st.sidebar.title("🎛️ BetEdge Controls")
sport = st.sidebar.radio("Select Sport", ["Horse Racing", "Greyhounds", "Football"])
country = st.sidebar.radio("Country", ["UK", "Ireland", "South Africa"])
day_option = st.sidebar.radio("Select Day", ["Today", "Tomorrow"])
view = st.sidebar.radio("View Mode", ["Tables", "Charts"])

# --- Load Data ---
df = generate_mock_data(day_option.lower())

# --- Filters ---
course_options = ["All"] + sorted(df["Course"].unique().tolist())
selected_course = st.sidebar.selectbox("Filter by Course", course_options)
min_val, max_val = st.sidebar.slider("Filter by BetEdge Win %", 0, 100, (0, 100))

# --- Apply Filters ---
if selected_course != "All":
    df = df[df["Course"] == selected_course]
df = df[(df["BetEdge Win %"] >= min_val) & (df["BetEdge Win %"] <= max_val)]

df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

# --- Output Section ---
if df.empty:
    st.warning("No races match the selected filters.")
else:
    st.success(f"Showing {len(df)} runners for **{day_option.title()}**")
    st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    def color_val(v):
        if v > 20: return 'background-color:#58D68D;color:black'
        elif v > 10: return 'background-color:#F9E79F;color:black'
        else: return 'background-color:#F5B7B1;color:black'

    if view == "Tables":
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win.style.applymap(color_val, subset=["BetEdge Win %"]))
        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place.style.applymap(color_val, subset=["BetEdge Place %"]))
    else:
        st.subheader("📊 Top 20 BetEdge Win %")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("📊 Top 20 BetEdge Place %")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
