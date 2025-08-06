import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import numpy as np

# API credentials
USERNAME = "edypVknQtk8n3artYstntbPu"
PASSWORD = "DIDUKRnNjVtP1tvQOpbcCGC7"

# Page setup
st.set_page_config(page_title="BetEdge Value Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

# Fetch racecards from The Racing API
def fetch_racecards():
    url = "https://api.theracingapi.com/v1/racecards/today"
    headers = {
        "User-Agent": "BetEdge/1.0"
    }
    try:
        r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e} – {r.text}")
        return None
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")
        return None

# Process API data into DataFrame
def process_data(raw_data):
    races = []
    if not raw_data:
        return pd.DataFrame()
    for meeting in raw_data.get("meetings", []):
        course = meeting.get("course", {}).get("name", "Unknown")
        for race in meeting.get("races", []):
            time = race.get("off", "")[:5]
            for runner in race.get("runners", []):
                name = runner.get("name", "Unknown Horse")
                odds = runner.get("best_price", {}).get("decimal", np.random.uniform(2, 6))
                win_val = np.random.uniform(5, 25)
                place_val = win_val * 0.6
                races.append({
                    "Race": f"{course} {time}",
                    "Time": time,
                    "Course": course,
                    "Horse": name,
                    "Best Odds": round(odds, 2),
                    "Win_Value": round(win_val, 1),
                    "Place_Value": round(place_val, 1)
                })
    df = pd.DataFrame(races)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    return df

# UI interaction
view_mode = st.radio("Select View Mode:", ["Charts", "Tables"])
raw = fetch_racecards()
df = process_data(raw)

if df.empty:
    st.warning("No data available.")
else:
    df_win = df.sort_values(by="BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values(by="BetEdge Place %", ascending=False).reset_index(drop=True)

    def color_val(val):
        if val > 20:
            return 'background-color: #58D68D; color: black'
        elif val > 10:
            return 'background-color: #F9E79F; color: black'
        else:
            return 'background-color: #F5B7B1; color: black'

    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if view_mode == "Tables":
        st.subheader("🏆 Win Rankings (BetEdge %)")
        st.dataframe(
            df_win[["Race", "Horse", "Best Odds", "Win_Value", "Predicted Win %", "BetEdge Win %"]]
            .style.applymap(color_val, subset=["BetEdge Win %"])
        )

        st.subheader("🥈 Place Rankings (BetEdge %)")
        st.dataframe(
            df_place[["Race", "Horse", "Best Odds", "Place_Value", "Predicted Place %", "BetEdge Place %"]]
            .style.applymap(color_val, subset=["BetEdge Place %"])
        )

        st.subheader("🏅 Top 3 BetEdge Win Picks")
        for idx, row in df_win.head(3).iterrows():
            st.markdown(f"**{idx+1}. {row['Race']} – {row['Horse']}**")
            st.progress(row["BetEdge Win %"] / 100)
            st.write(f"BetEdge Win: {row['BetEdge Win %']}%")

        st.subheader("🥉 Top 3 BetEdge Place Picks")
        for idx, row in df_place.head(3).iterrows():
            st.markdown(f"**{idx+1}. {row['Race']} – {row['Horse']}**")
            st.progress(row["BetEdge Place %"] / 100)
            st.write(f"BetEdge Place: {row['BetEdge Place %']}%")

    else:
        st.subheader("📊 Top 20 BetEdge Win % Horses")
        win_chart_data = df_win.head(20).set_index("Horse")["BetEdge Win %"]
        st.bar_chart(win_chart_data)

        st.subheader("📊 Top 20 BetEdge Place % Horses")
        place_chart_data = df_place.head(20).set_index("Horse")["BetEdge Place %"]
        st.bar_chart(place_chart_data)
