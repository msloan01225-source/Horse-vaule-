import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
import numpy as np

USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(page_title="BetEdge Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

@st.cache_data(ttl=600)
def fetch_racecards(day="today"):
    date = datetime.utcnow().date()
    if day == "tomorrow":
        date += timedelta(days=1)
    params = {"date": date.strftime("%Y-%m-%d"), "region_codes": ["GB"]}
    r = requests.get("https://api.theracingapi.com/v1/racecards",
                     auth=HTTPBasicAuth(USERNAME, PASSWORD),
                     params=params, timeout=10)
    try:
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        st.error(f"API Error {r.status_code}: {r.text}")
        return None

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
                odds = float(runner.get("sp_dec", np.random.uniform(2,6)))
                win_val = np.random.uniform(5, 25)
                rows.append({"Time": off, "Course": course, "Horse": horse,
                             "Best Odds": round(odds,2),
                             "Win_Value": round(win_val,1),
                             "Place_Value": round(win_val*0.6,1)})
    df = pd.DataFrame(rows)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

day = st.radio("Select Day", ["today", "tomorrow"])
raw = fetch_racecards(day)
df = process_data(raw)

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
