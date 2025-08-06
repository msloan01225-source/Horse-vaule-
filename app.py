import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="BetEdge Value Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

# --------- Scrapers (Non-Selenium) ---------
def get_racingpost_data(day="Today"):
    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"
    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        races = []
        for tag in soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$")):
            race_time = tag.strip()
            course_tag = tag.find_next(string=True)
            course = course_tag.strip() if course_tag else "Unknown"
            race_name_tag = tag.find_next("a")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Race"
            races.append({
                "Race": f"{course} {race_time} – {race_name}",
                "Time": race_time,
                "Course": course,
                "Horse": race_name,
                "Win_Value": np.random.uniform(5, 25),  # Placeholder
                "Place_Value": np.random.uniform(5, 25)  # Placeholder
            })
        return pd.DataFrame(races)
    except Exception:
        return pd.DataFrame(columns=["Race", "Time", "Course", "Horse", "Win_Value", "Place_Value"])

def get_timeform_data(day="Today"):
    base_url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"
    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        races = []
        for tag in soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$")):
            race_time = tag.strip()
            course_tag = tag.find_next(string=True)
            course = course_tag.strip() if course_tag else "Unknown"
            race_name_tag = tag.find_next("a")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Race"
            races.append({
                "Race": f"{course} {race_time} – {race_name}",
                "Time": race_time,
                "Course": course,
                "Horse": race_name,
                "Win_Value": np.random.uniform(5, 25),  # Placeholder
                "Place_Value": np.random.uniform(5, 25)  # Placeholder
            })
        return pd.DataFrame(races)
    except Exception:
        return pd.DataFrame(columns=["Race", "Time", "Course", "Horse", "Win_Value", "Place_Value"])

# --------- Data Loading ---------
@st.cache_data(ttl=300)
def load_data(day):
    rp = get_racingpost_data(day)
    tf = get_timeform_data(day)

    # Add placeholder odds and predictions
    for df in [rp, tf]:
        if not df.empty:
            df["Best Odds"] = np.random.uniform(2, 6, len(df)).round(2)
            df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
            df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
            df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
            df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)

    return rp, tf

# --------- UI ---------
day_option = st.selectbox("Select Day:", ["Today", "Tomorrow"])
rp_df, tf_df = load_data(day_option)

def color_val(val):
    if val > 20:
        return 'background-color: #58D68D; color: black'
    elif val > 10:
        return 'background-color: #F9E79F; color: black'
    else:
        return 'background-color: #F5B7B1; color: black'

st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Display Racing Post Table
if not rp_df.empty:
    st.subheader("📊 Racing Post Rankings (BetEdge %)")
    st.dataframe(
        rp_df.sort_values(by="BetEdge Win %", ascending=False)[
            ["Race", "Horse", "Best Odds", "Win_Value", "Predicted Win %", "BetEdge Win %"]
        ].style.applymap(color_val, subset=["BetEdge Win %"])
    )
else:
    st.info("No Racing Post data available.")

# Display Timeform Table
if not tf_df.empty:
    st.subheader("📊 Timeform Rankings (BetEdge %)")
    st.dataframe(
        tf_df.sort_values(by="BetEdge Win %", ascending=False)[
            ["Race", "Horse", "Best Odds", "Win_Value", "Predicted Win %", "BetEdge Win %"]
        ].style.applymap(color_val, subset=["BetEdge Win %"])
    )
else:
    st.info("No Timeform data available.")
