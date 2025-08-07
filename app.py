import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="🏇 BetEdge Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

# --- Generate Mock Data ---
@st.cache_data(ttl=600)
def generate_mock_data():
    courses = ["Ascot", "Newmarket", "York", "Goodwood", "Cheltenham"]
    horses = [f"Horse {i}" for i in range(1, 51)]
    rows = []
    for horse in horses:
        course = np.random.choice(courses)
        time = (datetime.now() + timedelta(minutes=np.random.randint(30, 600))).strftime("%H:%M")
        odds = round(np.random.uniform(2, 10), 2)
        win_val = round(np.random.uniform(5, 25), 1)
        place_val = round(win_val * np.random.uniform(0.5, 0.7), 1)
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
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# --- Load Data ---
df = generate_mock_data()

# --- Layout Options ---
view_mode = st.radio("🔎 View Mode", ["📈 Charts", "📋 Tables"])
st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# --- Styling Function ---
def color_val(v):
    if v > 20: return 'background-color:#58D68D;color:black'
    elif v > 10: return 'background-color:#F9E79F;color:black'
    else: return 'background-color:#F5B7B1;color:black'

# --- Split & Sort ---
df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

# --- Display ---
if view_mode == "📋 Tables":
    st.subheader("🏆 Win Rankings")
    st.dataframe(df_win.style.applymap(color_val, subset=["BetEdge Win %"]))

    st.subheader("🥈 Place Rankings")
    st.dataframe(df_place.style.applymap(color_val, subset=["BetEdge Place %"]))

    st.subheader("💡 Top 3 BetEdge Picks (Win)")
    for idx, row in df_win.head(3).iterrows():
        st.markdown(f"**{idx+1}. {row['Horse']} at {row['Course']} – {row['Time']}**")
        st.progress(row["BetEdge Win %"] / 100)
        st.write(f"📊 BetEdge Win: {row['BetEdge Win %']}%")

    st.subheader("📌 Top 3 BetEdge Picks (Place)")
    for idx, row in df_place.head(3).iterrows():
        st.markdown(f"**{idx+1}. {row['Horse']} at {row['Course']} – {row['Time']}**")
        st.progress(row["BetEdge Place %"] / 100)
        st.write(f"📊 BetEdge Place: {row['BetEdge Place %']}%")

else:
    st.subheader("📊 Top 20 BetEdge Win %")
    st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])

    st.subheader("📊 Top 20 BetEdge Place %")
    st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
