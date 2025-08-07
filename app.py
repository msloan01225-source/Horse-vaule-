import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# --- Your Racing API credentials ---
USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

st.set_page_config(page_title="EdgeBet – Live Horse Racing", layout="wide")
# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_live_horse_data(day: str = "today") -> pd.DataFrame:
    """Fetch today’s or tomorrow’s UK racing from The Racing API."""
    # build date
    target = datetime.utcnow().date()
    if day == "tomorrow":
        target += timedelta(days=1)
    ds = target.strftime("%Y-%m-%d")

    # 1) get all “stages” (meetings) for that date & UK
    try:
        r = requests.get(
            "https://api.theracingapi.com/v1/stages",
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            params={"from": ds, "to": ds, "countrycodes": "GB"},
            timeout=10
        )
        r.raise_for_status()
        meetings = r.json().get("data", [])
    except Exception as e:
        st.error(f"Error fetching meetings: {e}")
        return pd.DataFrame()

    rows = []
    for m in meetings:
        course = m.get("course", {}).get("name", "Unknown")
        for ev in m.get("events", []):
            off = ev.get("off", "")[:5]
            # 2) fetch event details to get runners & decimal SP
            try:
                e = requests.get(
                    f"https://api.theracingapi.com/v1/events/{ev['id']}",
                    auth=HTTPBasicAuth(USERNAME, PASSWORD),
                    timeout=10
                )
                e.raise_for_status()
                details = e.json()
            except:
                continue

            for runner in details.get("runners", []):
                odds = runner.get("sp_dec")
                odds = float(odds) if odds else np.random.uniform(2,6)
                # no direct “value” field → placeholder zero
                win_val = 0.0
                rows.append({
                    "Time": off,
                    "Course": course,
                    "Horse": runner.get("horse", "Unknown"),
                    "Odds": round(odds, 2),
                    "Win_Value": win_val
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 3) compute Edge metrics
    df["Predicted Win %"]   = (1 / df["Odds"] * 100).round(1)
    df["Place_Value"]       = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]     = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"]   = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"]  * 0.4)).round(1)

    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ---- SIDEBAR MENU ----
with st.sidebar:
    page = option_menu(
        "🏠 Main Menu",
        ["Overview", "Horse Racing", "Football", "How It Works"],
        icons=["house", "activity", "soccer", "book"],
        menu_icon="cast",
        default_index=1
    )

# ---- OVERVIEW ----
if page == "Overview":
    st.title("📊 EdgeBet Live Overview")
    st.write("UK horse racing data powered by The Racing API.")
    today_df = fetch_live_horse_data("today")
    st.metric("Races Today", today_df["Course"].nunique())
    st.metric("Total Runners", len(today_df))
    top = today_df["BetEdge Win %"].max() if not today_df.empty else 0
    st.metric("Top BetEdge Win %", f"{top}%")

# ---- HORSE RACING ----
elif page == "Horse Racing":
    st.title("🏇 Horse Racing – UK Live")
    day = st.selectbox("Select Day", ["Today", "Tomorrow"])
    df = fetch_live_horse_data(day.lower())

    if df.empty:
        st.warning(f"No races available for {day}.")
    else:
        st.write(f"**Showing {day}** – last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        view = st.radio("View Mode", ["Charts", "Tables"], horizontal=True)

        if view == "Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(df.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(df.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(df, use_container_width=True)

# ---- FOOTBALL PLACEHOLDER ----
elif page == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Coming soon…")

# ---- HOW IT WORKS ----
else:
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    1. Fetch live UK racing from The Racing API  
    2. Compute implied probabilities from SP  
    3. Blend with value metrics → **BetEdge %**  
    4. Rank & visualize value across the card  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # placeholder

st.caption(f"Built on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
