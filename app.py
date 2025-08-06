import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="BetEdge – UK Racing Value Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker (via The Racing API)")

# ----------- API Credentials -----------
USERNAME = "edypVknQtk8n3artYstntbPu"
PASSWORD = "DIDUKRnNjVtP1tvQOpbcCGC7"

# ----------- Fetch Racecards Function -----------
@st.cache_data(ttl=300)
def fetch_racecards():
    url = "https://api.theracingapi.com/v1/racecards/today"
    try:
        r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=10)
        r.raise_for_status()
        data = r.json()
        return data
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e}")
        return None
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")
        return None

# ----------- Process & Display Data -----------
data = fetch_racecards()

if data and isinstance(data, list) and len(data) > 0:
    rows = []
    for meeting in data:
        course = meeting.get("course", {}).get("name", "Unknown")
        for race in meeting.get("races", []):
            race_time = race.get("offTime", "Unknown")
            race_title = race.get("raceTitle", "")
            runners = race.get("runners", [])
            for horse in runners:
                horse_name = horse.get("horse", {}).get("name", "Unknown")
                rows.append({
                    "Time": race_time,
                    "Course": course,
                    "Race Title": race_title,
                    "Horse": horse_name
                })
    df = pd.DataFrame(rows)
    df = df.sort_values(by="Time").reset_index(drop=True)

    st.success(f"Showing {len(df)} runners from today's UK racecards")
    st.dataframe(df)

else:
    st.warning("No race data found or API request failed.")
