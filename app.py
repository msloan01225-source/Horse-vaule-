# app.py
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from streamlit_option_menu import option_menu

# ---------- CONFIG ----------
st.set_page_config(page_title="EdgeBet – Live Racing (Basic API)", layout="wide")

# Try to load creds from secrets; fall back to constants
USERNAME = st.secrets.get("racing_api", {}).get("username", os.getenv("RACING_API_USERNAME", "YOUR_USERNAME_HERE"))
PASSWORD = st.secrets.get("racing_api", {}).get("password", os.getenv("RACING_API_PASSWORD", "YOUR_PASSWORD_HERE"))

# ---------- THEME ----------
st.markdown("""
<style>
body { background-color: #111111; color: #f2f2f2; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.block-container { padding-top: 1.5rem; }
[data-testid="stMetricValue"] { color: #00ffcc !important; }
</style>
""", unsafe_allow_html=True)

# ---------- API: list racecards (BASIC) ----------
@st.cache_data(ttl=120)
def fetch_racecards(day: str = "today"):
    """
    Calls BASIC endpoint:
      GET https://api.theracingapi.com/v1/racecards?day=today|tomorrow
    Returns (ok, json, status_code, text) so we can debug easily in the UI.
    """
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"day": day.lower()}  # <- Basic endpoint uses 'day' NOT 'date'
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, timeout=12)
        status = resp.status_code
        text = resp.text
        resp.raise_for_status()
        return True, resp.json(), status, text
    except requests.HTTPError as e:
        return False, None, getattr(e.response, "status_code", None), getattr(e.response, "text", str(e))
    except Exception as e:
        return False, None, None, str(e)

def map_region_to_country(region: str) -> str:
    """
    Basic plan returns region like 'GB', 'IE', 'US', etc.
    We'll show 'UK' for GB, otherwise use region as a fallback.
    """
    if not isinstance(region, str):
        return "Unknown"
    if region.upper() == "GB":
        return "UK"
    if region.upper() in ("US", "USA"):
        return "USA"
    if region.upper() == "IE":
        return "Ireland"
    return region.upper()

def build_live_df_from_api(payload: dict) -> pd.DataFrame:
    """
    Converts the racecards JSON (Basic endpoint) into a DataFrame of runners.
    Expected shape (per official gist): top-level has 'racecards', each with:
      - region
      - course
      - off_time
      - runners: [{horse, horse_id, trainer, ...}, ...]
    Odds are NOT provided on Basic -> we’ll mock odds so the UI keeps working.
    """
    if not payload or "racecards" not in payload:
        return pd.DataFrame()

    rows = []
    for race in payload.get("racecards", []):
        region = race.get("region")
        course = race.get("course")
        off    = race.get("off_time")  # e.g. "14:10"
        for rnr in race.get("runners", []):
            horse = rnr.get("horse")
            if not horse:
                continue
            rows.append({
                "Country": map_region_to_country(region),
                "Course": course,
                "Time": off,
                "Horse": horse
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Mock odds & derived metrics (Basic plan doesn’t include odds)
    np.random.seed(42)
    df["Odds"] = np.random.uniform(2, 10, len(df)).round(2)
    df["Win_Value"] = np.random.uniform(5, 30, len(df)).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (1 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Source"] = "LIVE (names/times) + Mock Odds"  # honesty label
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ---------- EdgeBrain (stub until model is present) ----------
def edgebrain_score_stub(df: pd.DataFrame) -> pd.Series:
    # If you later drop in a trained model, replace this.
    return (0.5 * df["BetEdge Win %"] + 0.5 * df["Predicted Win %"]).round(1)

# ---------- SIDEBAR ----------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Debug"],
        icons=["house", "activity", "robot", "bug"],
        default_index=0
    )

# ---------- OVERVIEW ----------
if sel == "Overview":
    st.title("📊 EdgeBet – Live Racing (Basic API)")
    ok, data, status, text = fetch_racecards("today")
    if not ok:
        st.error(f"API error (today): HTTP {status}\n{text[:300]}")
        st.info("Tip: Basic endpoint is GET /v1/racecards?day=today|tomorrow (no 'date').")
        st.stop()

    live_df = build_live_df_from_api(data)
    st.metric("Meetings (today)", len({(rc.get('region'), rc.get('course')) for rc in data.get('racecards', [])}))
    st.metric("Races (today)", len(data.get("racecards", [])))
    st.metric("Runners (today)", len(live_df))

    if live_df.empty:
        st.warning("No runners returned by the API for today. Try 'Horse Racing' tab and switch to tomorrow.")
    else:
        st.subheader("Top 20 BetEdge (today)")
        st.bar_chart(live_df.head(20).set_index("Horse")["BetEdge Win %"])

# ---------- HORSE RACING ----------
elif sel == "Horse Racing":
    st.title("🏇 Horse Racing – Live (Basic API)")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)

    ok, data, status, text = fetch_racecards(day)
    if not ok:
        st.error(f"API error ({day}): HTTP {status}\n{text[:300]}")
        st.stop()

    df = build_live_df_from_api(data)

    # Filters & controls
    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country", ["All"] + sorted(df["Country"].dropna().unique().tolist()) if not df.empty else ["All"])
    with col2:
        # Placeholder – we’re not fetching real bookies yet (those would come from odds feeds)
        bookie = st.selectbox("Bookmaker", ["All", "SkyBet", "Bet365", "Betfair"])
    with col3:
        courses = sorted(df["Course"].dropna().unique().tolist()) if not df.empty else []
        course_filter = st.multiselect("Courses", courses, default=courses)

    if df.empty:
        st.warning("No data from API. Try switching the day to 'tomorrow'.")
        st.stop()

    # Apply filters
    filt = df.copy()
    if country != "All":
        filt = filt[filt["Country"] == country]
    if course_filter:
        filt = filt[filt["Course"].isin(course_filter)]

    # Sort & view mode
    min_val = int(filt["BetEdge Win %"].min()) if not filt.empty else 0
    max_val = int(filt["BetEdge Win %"].max()) if not filt.empty else 0
    edge_range = st.slider("🎯 Filter by BetEdge Win %", min_val, max_val, (min_val, max_val))
    filt = filt[filt["BetEdge Win %"].between(*edge_range)]
    view = st.radio("View", ["📊 Charts", "📋 Tables"], horizontal=True)

    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        if view == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(filt.sort_values("BetEdge Win %", ascending=False)
                             .head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(filt.sort_values("BetEdge Place %", ascending=False)
                             .head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("Full Runner Table")
            st.dataframe(
                filt[["Country", "Course", "Time", "Horse", "Odds", "BetEdge Win %", "BetEdge Place %", "Source"]],
                use_container_width=True
            )

# ---------- EDGEBRAIN ----------
elif sel == "EdgeBrain":
    st.title("🧠 EdgeBrain – Live Names + Stubbed Odds")
    ok, data, status, text = fetch_racecards("today")
    if not ok:
        st.error(f"API error (today): HTTP {status}\n{text[:300]}")
        st.stop()
    df = build_live_df_from_api(data)
    if df.empty:
        st.warning("No runners returned by API for today. Try tomorrow in Horse Racing tab.")
    else:
        df["EdgeBrain Win %"] = edgebrain_score_stub(df)
        st.subheader("Top 20 EdgeBrain Win %")
        st.bar_chart(df.sort_values("EdgeBrain Win %", ascending=False)
                       .head(20).set_index("Horse")["EdgeBrain Win %"])
        st.dataframe(df[["Country", "Course", "Time", "Horse", "Odds", "EdgeBrain Win %", "Risk", "Source"]],
                     use_container_width=True)

# ---------- DEBUG ----------
else:
    st.title("🐞 API Debug")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    ok, data, status, text = fetch_racecards(day)
    st.write("Request:", "`GET /v1/racecards`", "params:", {"day": day})
    st.write("HTTP Status:", status)
    if ok:
        st.success("API OK")
        st.write("Top-level keys:", list(data.keys()))
        st.write("racecards count:", len(data.get("racecards", [])))
        # show a small sample for inspection
        sample = data.get("racecards", [])[:2]
        st.json(sample)
        df = build_live_df_from_api(data)
        st.write("DataFrame shape:", df.shape)
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.error("API error:")
        st.code(text[:1000])

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
