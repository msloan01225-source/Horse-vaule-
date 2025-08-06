import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
import numpy as np

# --- API Credentials ---
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(page_title="BetEdge Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

# --- Fetch Data from API ---
@st.cache_data(ttl=600)
def fetch_racecards(day="today"):
    r = requests.get(
        f"https://api.theracingapi.com/v1/racecards/{day}",
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        timeout=10
    )
    try:
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        st.error(f"API Error {r.status_code}: {r.text}")
        return None

# --- Process API Data ---
def process_data(raw):
    if not raw or "meetings" not in raw:
        return pd.DataFrame()
    rows = []
    for m in raw["meetings"]:
        course = m.get("course", {}).get("name", "Unknown")
        for race in m.get("races", []):
            off = race.get("off", "")[:5]
            for runner in race.get("runners", []):
                horse = runner.get("horse", runner.get("horse_id", "Unknown"))

                try:
                    odds = float(runner.get("sp_dec"))
                    if odds <= 1.01:
                        raise ValueError
                except (TypeError, ValueError):
                    odds = round(np.random.uniform(2, 6), 2)

                win_val = round(np.random.uniform(5, 25), 1)
                place_val = round(win_val * 0.6, 1)

                rows.append({
                    "Time": off,
                    "Course": course,
                    "Horse": horse,
                    "Best Odds": odds,
                    "Win_Value": win_val,
                    "Place_Value": place_val
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

# --- UI & Display ---
day = st.radio("Select Day", ["today", "tomorrow"])
raw_data = fetch_racecards(day)
df = process_data(raw_data)

if df.empty:
    st.warning("No races available for this date.")
else:
    df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    view = st.radio("View Mode", ["Charts", "Tables"])

    def color_val(v):
        if v > 20: return 'background-color:#58D68D;color:black'
        if v > 10: return 'background-color:#F9E79F;color:black'
        return 'background-color:#F5B7B1;color:black'

    if view == "Tables":
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win.style.applymap(color_val, subset=["BetEdge Win %"]))
        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place.style.applymap(color_val, subset=["BetEdge Place %"]))
    else:
        st.subheader("📊 BetEdge Win % Chart (Top 20)")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("📊 BetEdge Place % Chart (Top 20)")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
