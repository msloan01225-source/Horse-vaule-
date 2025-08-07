import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import numpy as np

# --- Credentials (update as needed) ---
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(page_title="BetEdge Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

@st.cache_data(ttl=600)
def fetch_racecards(day="today"):
    url = f"https://api.theracingapi.com/v1/racecards/{day}"
    try:
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"API Error: {e}")
        return None

def fetch_mock_data():
    return {
        "meetings": [{
            "course": {"name": "Mock Course"},
            "races": [{
                "off": "14:00",
                "runners": [
                    {"horse": "Mock Horse A", "sp_dec": 3.5},
                    {"horse": "Mock Horse B", "sp_dec": 5.0},
                    {"horse": "Mock Horse C", "sp_dec": 4.2}
                ]
            }]
        }]
    }

def process_data(raw):
    if not raw or "meetings" not in raw:
        return pd.DataFrame()
    rows = []
    for m in raw["meetings"]:
        course = m.get("course", {}).get("name", "Unknown")
        for race in m.get("races", []):
            off = race.get("off","")[:5]
            for runner in race.get("runners", []):
                horse = runner.get("horse", runner.get("horse_id", "Unknown"))
                odds = float(runner.get("sp_dec", np.random.uniform(2,6)))
                win_val = np.random.uniform(5,25)
                rows.append({
                    "Time": off,
                    "Course": course,
                    "Horse": horse,
                    "Best Odds": round(odds,2),
                    "Win_Value": round(win_val,1),
                    "Place_Value": round(win_val * 0.6,1)
                })
    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

def fetch_racecards_with_fallback():
    for day in ["today", "tomorrow"]:
        raw = fetch_racecards(day)
        df = process_data(raw)
        if not df.empty:
            return df, day
    mock = fetch_mock_data()
    return process_data(mock), "mock"

df, source = fetch_racecards_with_fallback()

if df.empty:
    st.warning("No race data available.")
else:
    label = "Mock Data" if source == "mock" else f"**{source.title()}**"
    st.success(f"Showing races for {label}")
    st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

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
