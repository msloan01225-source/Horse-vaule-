import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# === Credentials ===
USERNAME = "edypVknQtk8n3artYstntbPu"
PASSWORD = "DIDUKRnNjVtP1tvQOpbcCGC7"
API_URL = "https://api.theracingapi.com/v1/racecards"

st.set_page_config(page_title="BetEdge Value Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker (via Racing API)")

# === Get Data from Racing API ===
def fetch_racecards(region="GB", date=None):
    params = {"region": region}
    if date:
        params["date"] = date
    try:
        resp = requests.get(API_URL, params=params, auth=(USERNAME, PASSWORD), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return []

# === Parse API JSON into DataFrame ===
def build_df(api_data):
    rows = []
    for race in api_data:
        off = race.get("off", "")
        course = race.get("course", "")
        for runner in race.get("runners", []):
            horse = runner.get("horse") or "Unknown"
            sp_dec = runner.get("sp_dec")
            try:
                odds = float(sp_dec) if sp_dec else 0.0
            except:
                odds = 0.0
            win_val = round(max(0, (10 - odds) * 2), 1)  # placeholder logic
            rows.append({
                "Race":    f"{course} {off}",
                "Time":    off,
                "Course":  course,
                "Horse":   horse,
                "Best Odds": odds,
                "Win_Value": win_val
            })
    return pd.DataFrame(rows)

# === App Logic ===
day = st.selectbox("Select Day:", ["Today", "Tomorrow"])
target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d") if day == "Tomorrow" else datetime.utcnow().strftime("%Y-%m-%d")

api_data = fetch_racecards(date=target_date)
df = build_df(api_data)

if df.empty:
    st.warning("No races found. Please try again later or select another day.")
else:
    df = df[df["Best Odds"] > 0].copy()
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)

    df_sorted = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

    # Display Top 20 Table
    st.subheader("📊 Top 20 Horses by BetEdge Win %")
    st.dataframe(df_sorted.head(20)[["Race", "Horse", "Best Odds", "Win_Value", "Predicted Win %", "BetEdge Win %"]])

    # Chart
    st.subheader("📈 Bar Chart – BetEdge Win % (Top 20)")
    chart_df = df_sorted.head(20).set_index("Horse")["BetEdge Win %"]
    st.bar_chart(chart_df)

    # Top 3 Picks
    st.subheader("🏅 Top 3 BetEdge Picks")
    for idx, row in df_sorted.head(3).iterrows():
        st.markdown(f"**{idx+1}. {row['Horse']} – {row['Race']}**")
        st.progress(row["BetEdge Win %"] / 100)
        st.caption(f"Odds: {row['Best Odds']} | BetEdge: {row['BetEdge Win %']}%")
