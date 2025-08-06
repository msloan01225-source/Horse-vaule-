import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import numpy as np

USERNAME = "edypVknQtk8n3artYstntbPu"
PASSWORD = "DIDUKRnNjVtP1tvQOpbcCGC7"

st.set_page_config(page_title="BetEdge Value Tracker", layout="wide")
st.title("🏇 BetEdge – UK Racing Value Tracker")

def fetch_racecards(day="today"):
    date_str = datetime.utcnow().strftime("%Y-%m-%d") if day=="today" else (datetime.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"date": date_str, "region": "GB"}
    headers = {"User-Agent": "BetEdge/1.0"}
    try:
        r = requests.get(url, params=params, auth=HTTPBasicAuth(USERNAME, PASSWORD), headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")
        return None

def process_data(raw):
    if not raw:
        return pd.DataFrame()
    rows=[]
    for meeting in raw.get("meetings", []):
        course = meeting.get("course","") or ""
        for race in meeting.get("races", []):
            time = race.get("off","")[:5]
            for runner in race.get("runners", []):
                name = runner.get("horse", runner.get("horse_id","Unknown"))
                odds = float(runner.get("sp_dec", np.random.uniform(2,6)))
                win_val = np.random.uniform(5,25)
                rows.append({
                    "Race": f"{course} {time}",
                    "Time": time,
                    "Course": course,
                    "Horse": name,
                    "Best Odds": round(odds,2),
                    "Win_Value": round(win_val,1),
                    "Place_Value": round(win_val*0.6,1)
                })
    df=pd.DataFrame(rows)
    df["Predicted Win %"] = (1/df["Best Odds"]*100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"]*0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    return df

view = st.radio("Mode:", ["Charts","Tables"])
raw = fetch_racecards("today")
df = process_data(raw)

if df.empty:
    st.warning("No data to display.")
else:
    df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    def color_val(v):
        if v>20: return 'background-color:#58D68D;color:black'
        if v>10: return 'background-color:#F9E79F;color:black'
        return 'background-color:#F5B7B1;color:black'

    st.write(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    if view=="Tables":
        st.subheader("🏆 Win Rankings")
        st.dataframe(df_win.style.applymap(color_val, subset=["BetEdge Win %"]))
        st.subheader("🥈 Place Rankings")
        st.dataframe(df_place.style.applymap(color_val, subset=["BetEdge Place %"]))
    else:
        st.subheader("📊 Top 20 BetEdge Win %")
        st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
        st.subheader("📊 Top 20 BetEdge Place %")
        st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
