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
                "Win_Value": 0.0,
                "Place_Value": 0.0
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
                "Win_Value": 0.0,
                "Place_Value": 0.0
            })
        return pd.DataFrame(races)
    except Exception:
        return pd.DataFrame(columns=["Race", "Time", "Course", "Horse", "Win_Value", "Place_Value"])

# --------- Data Loading & Merging ---------

@st.cache_data(ttl=300)
def load_data(day):
    rp = get_racingpost_data(day)
    tf = get_timeform_data(day)
    if rp.empty and tf.empty:
        return pd.DataFrame()
    try:
        df = pd.merge(rp, tf, on=["Race", "Horse"], how="inner", suffixes=("_RP", "_TF"))
    except Exception:
        df = pd.concat([rp, tf], ignore_index=True)
    if "Win_Value_RP" in df.columns and "Win_Value_TF" in df.columns:
        df["Win_Value"] = df[["Win_Value_RP", "Win_Value_TF"]].mean(axis=1)
    elif "Win_Value" not in df.columns:
        df["Win_Value"] = 0.0
    if "Place_Value_RP" in df.columns and "Place_Value_TF" in df.columns:
        df["Place_Value"] = df[["Place_Value_RP", "Place_Value_TF"]].mean(axis=1)
    elif "Place_Value" not in df.columns:
        df["Place_Value"] = 0.0
    # Placeholder "Best Odds" and probabilities
    df["Best Odds"] = np.random.uniform(2, 6, len(df)).round(2)
    df["Predicted Win %"] = (1 / df["Best Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    return df

# --------- UI ---------

day_option = st.selectbox("Select Day:", ["Today", "Tomorrow"])
df = load_data(day_option)

if df.empty:
    st.warning("No data available. Please check sources.")
else:
    # Colour coding functions
    def color_val(val):
        if val > 20:
            return 'background-color: #58D68D; color: black'  # bright green
        elif val > 10:
            return 'background-color: #F9E79F; color: black'  # yellow
        else:
            return 'background-color: #F5B7B1; color: black'  # red

    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Win table
    st.subheader("🏆 Win Rankings (Highest Value %)")
    df_win = df.sort_values(by="Win_Value", ascending=False).reset_index(drop=True)
    st.dataframe(
        df_win[["Race", "Horse", "Best Odds", "Win_Value", "Predicted Win %"]]
        .style.applymap(color_val, subset=["Win_Value"])
    )

    # Place table
    st.subheader("🥈 Place Rankings (Highest Value %)")
    df_place = df.sort_values(by="Place_Value", ascending=False).reset_index(drop=True)
    st.dataframe(
        df_place[["Race", "Horse", "Best Odds", "Place_Value", "Predicted Place %"]]
        .style.applymap(color_val, subset=["Place_Value"])
    )

    # Top 3 predicted wins & places
    st.subheader("🏅 Top 3 Predicted Win Picks")
    top3_win = df_win.head(3)
    for idx, row in top3_win.iterrows():
        st.markdown(f"**{idx+1}. {row['Race']} – {row['Horse']}**")
        st.progress(row["Predicted Win %"] / 100)
        st.write(f"Predicted Win: {row['Predicted Win %']}%")

    st.subheader("🥉 Top 3 Predicted Place Picks")
    top3_place = df_place.head(3)
    for idx, row in top3_place.iterrows():
        st.markdown(f"**{idx+1}. {row['Race']} – {row['Horse']}**")
        st.progress(row["Predicted Place %"] / 100)
        st.write(f"Predicted Place: {row['Predicted Place %']}%")
