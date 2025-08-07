import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import numpy as np

# --- Auth credentials ---
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(page_title="BetEdge Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

# --- Fetch racecards from API ---
@st.cache_data(ttl=600)
def fetch_racecards(day="today"):
    url = f"https://api.theracingapi.com/v1/racecards/{day}"
    try:
        r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"API Error {r.status_code}: {r.text}")
        return None

# --- Process API JSON into DataFrame ---
def process_data(raw):
    if not raw or "meetings" not in raw:
        return pd.DataFrame()

    rows = []
    for meeting in raw["meetings"]:
        course = meeting.get("course", {}).get("name", "Unknown")
        for race in meeting.get("races", []):
            race_time = race.get("off", "")[:5]
            for runner in race.get("runners", []):
                horse = runner.get("horse", runner.get("horse_id", "Unknown"))
                odds = float(runner.get("sp_dec", np.random.uniform(2, 6)))
                win_val = np.random.uniform(5, 25)
                place_val = win_val * 0.6
                rows.append({
                    "Time": race_time,
                    "Course": course,
                    "Horse": horse,
                    "Best Odds": round(odds, 2),
                    "Win_Value": round(win_val, 1),
                    "Place_Value": round(place_val, 1)
                })

    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

# --- Try today, then fallback to tomorrow ---
def fetch_with_fallback():
    for day in ["today", "tomorrow"]:
        raw = fetch_racecards(day)
        df = process_data(raw)
        if not df.empty:
            return df, day
    return pd.DataFrame(), "none"

# --- Load data ---
df, active_day = fetch_with_fallback()

if df.empty:
    st.warning("No races found for today or tomorrow.")
else:
    st.success(f"Showing data for **{active_day.title()}**")
    st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    view = st.radio("Select View Mode", ["Tables", "Charts"])

    def color_cell(val):
        if val > 20: return 'background-color: #58D68D; color: black'
        if val > 10: return 'background-color: #F9E79F; color: black'
        return 'background-color: #F5B7B1; color: black'

    if view == "Tables":
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win[["Time", "Course", "Horse", "Best Odds", "Win_Value", "BetEdge Win %"]]
                     .style.applymap(color_cell, subset=["BetEdge Win %"]))

        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place[["Time", "Course", "Horse", "Best Odds", "Place_Value", "BetEdge Place %"]]
                     .style.applymap(color_cell, subset=["BetEdge Place %"]))
    else:
        st.subheader("📊 BetEdge Win % (Top 20)")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])

        st.subheader("📊 BetEdge Place % (Top 20)")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
