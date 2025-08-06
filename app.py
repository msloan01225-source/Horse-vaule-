import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

# API credentials (updated)
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

# Streamlit setup
st.set_page_config(page_title="BetEdge Value Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

# -------- Fetch racecards from API --------
@st.cache_data(ttl=600)
def fetch_racecards(date="today"):
    url = f"https://api.theracingapi.com/v1/racecards/{date}"
    try:
        r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# -------- Process and score horses --------
def process_racecards(data):
    if not data or "racecards" not in data:
        return pd.DataFrame()
    
    races = []
    for card in data["racecards"]:
        course = card.get("course", {}).get("name", "Unknown")
        time = card.get("advertised_start_time", "Unknown")
        for runner in card.get("runners", []):
            horse = runner.get("horse", {}).get("name", "Unknown")
            races.append({
                "Time": time,
                "Course": course,
                "Horse": horse
            })

    df = pd.DataFrame(races)
    df["Best Odds"] = pd.Series(pd.np.random.uniform(2, 6, len(df))).round(2)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["BetEdge Score"] = (df["Predicted Win %"] * 0.7).round(1)
    return df

# -------- UI: Select Date --------
day = st.radio("Select Day", ["today", "tomorrow"])
data = fetch_racecards(day)
df = process_racecards(data)

# -------- UI: Output --------
if df.empty:
    st.warning("No race data found.")
else:
    st.subheader("📊 Top 20 Horses by BetEdge Score")
    df_sorted = df.sort_values(by="BetEdge Score", ascending=False).head(20)
    
    st.dataframe(df_sorted.reset_index(drop=True))
    
    st.bar_chart(df_sorted.set_index("Horse")["BetEdge Score"])
