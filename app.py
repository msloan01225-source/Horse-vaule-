import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# === API credentials ===
USERNAME = "edypVknQtk8n3artYstntbPu"
PASSWORD = "DIDUKRnNjVtP1tvQOpbcCGC7"
API_URL = "https://api.theracingapi.com/v1/racecards"

st.set_page_config(page_title="BetEdge Value Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

def fetch_racecards(region="GB", date=None):
    params = {"region": region}
    if date:
        params["date"] = date
    r = requests.get(API_URL, params=params, auth=(USERNAME, PASSWORD), timeout=10)
    r.raise_for_status()
    return r.json()

def build_df(api_data):
    rows = []
    for race in api_data:
        off = race.get("off", "")
        course = race.get("course", "")
        for runner in race.get("runners", []):
            horse = runner.get("horse", "Unknown")
            sp_dec = runner.get("sp_dec")
            odds = float(sp_dec) if sp_dec else 0.0
            win_val = round(max(0, (10 - odds) * 2), 1)
            rows.append({
                "Race":      f"{course} {off}",
                "Time":      off,
                "Course":    course,
                "Horse":     horse,
                "Best Odds": odds,
                "Win_Value": win_val
            })
    return pd.DataFrame(rows)

day = st.selectbox("Select Day:", ["Today", "Tomorrow"])
target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y‑%m‑%d") if day == "Tomorrow" else datetime.utcnow().strftime("%Y‑%m‑%d")

raw = fetch_racecards(date=target_date)
df = build_df(raw)

if df.empty:
    st.warning("No races found or API returned empty.")
else:
    df = df[df["Best Odds"] > 0].copy()
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)

    top = df.sort_values("BetEdge Win %", ascending=False).head(20).reset_index(drop=True)

    st.subheader("📊 Top 20 by BetEdge Win %")
    st.dataframe(top[["Race", "Horse", "Best Odds", "Win_Value", "Predicted Win %", "BetEdge Win %"]])

    st.subheader("📈 Bar Chart – BetEdge Win %")
    st.bar_chart(top.set_index("Horse")["BetEdge Win %"])

    st.subheader("🏅 Top 3 Picks")
    for i, row in top.head(3).iterrows():
        st.markdown(f"**{i+1}. {row['Horse']} – {row['Race']}**")
        st.progress(row["BetEdge Win %"] / 100)
        st.caption(f"Odds: {row['Best Odds']} | Win_Value: {row['Win_Value']} | BetEdge%: {row['BetEdge Win %']}%")
