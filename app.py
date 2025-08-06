import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

### API credentials (provided)
USERNAME = "edypVknQtk8n3artYstntbPu"
PASSWORD = "DIDUKRnNjVtP1tvQOpbcCGC7"
API_URL = "https://api.theracingapi.com/racecards"

st.set_page_config(page_title="BetEdge – Racing API Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker (via Racing API)")

def fetch_racecards(region="GB", date=None):
    params = {"region": region}
    if date:
        params["date"] = date
    try:
        r = requests.get(API_URL, params=params, auth=(USERNAME, PASSWORD), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return []

def build_dataframe(api_data):
    rows = []
    for race in api_data:
        race_time = race.get("off", "")
        course     = race.get("course", "")
        for runner in race.get("runners", []):
            horse = runner.get("horse") or "Unknown"
            odds  = float(runner.get("sp_dec") or 0)
            rows.append({
                "Race":   f"{course} {race_time}",
                "Time":    race_time,
                "Course":  course,
                "Horse":   horse,
                "Best Odds": odds
            })
    return pd.DataFrame(rows)

# UI: Today / Tomorrow switch
day_choice = st.selectbox("Select Day:", ["Today", "Tomorrow"])
target_date = (datetime.utcnow() if day_choice=="Today"
               else datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

raw = fetch_racecards(date=target_date)
df = build_dataframe(raw)

if df.empty:
    st.warning("No data available — either no races scheduled or API returned none.")
else:
    # Filter out zero odds
    df = df[df["Best Odds"] > 0].copy()
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["BetEdge Win %"] = (df["Predicted Win %"] * 0.6).round(1)  # Extend algorithm with win_value etc.

    # Sort top 20 by BetEdge Win %
    df_win = df.sort_values(by="BetEdge Win %", ascending=False).head(20)

    st.subheader(f"📊 Top 20 Bets by BetEdge Win % — {day_choice}")
    st.dataframe(df_win[["Race", "Horse", "Best Odds", "Predicted Win %", "BetEdge Win %"]])

    st.subheader("📈 BetEdge Win % Chart (Top 20)")
    chart_data = df_win.set_index("Horse")["BetEdge Win %"]
    st.bar_chart(chart_data)

    st.subheader("🏅 Your Top 3 Picks")
    for i, (_, row) in enumerate(df_win.head(3).iterrows(), start=1):
        st.markdown(f"**{i}. {row['Race']} — {row['Horse']}**")
        st.progress(row["BetEdge Win %"] / 100)
        st.caption(f"Odds: {row['Best Odds']} | BetEdge Win %: {row['BetEdge Win %']}%")
