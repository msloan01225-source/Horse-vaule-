# ------------------ IMPORTS ------------------
import os
import json
import math
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="EdgeBet – Live Racing", layout="wide")

# ------------------ THEME TOGGLE ------------------
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

def toggle_theme():
    st.session_state["theme_mode"] = "light" if st.session_state["theme_mode"] == "dark" else "dark"

col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("🏇 EdgeBet – Live Racing")
with col2:
    st.button(
        "🌗 Toggle Theme",
        on_click=toggle_theme,
        use_container_width=True
    )

# ------------------ THEME CSS ------------------
def apply_theme():
    is_dark = st.session_state["theme_mode"] == "dark"
    teal = "#00bfa6"
    gold = "#ffd700"
    if is_dark:
        bg = "#0f2a26"
        sbg = "#09211d"
        text = "#f2f2f2"
    else:
        bg = "#ffffff"
        sbg = "#f0f0f0"
        text = "#111111"

    st.markdown(f"""
    <style>
    /* App background & text */
    .stApp {{
        background-color: {bg};
        color: {text};
    }}
    /* Sidebar background */
    section[data-testid="stSidebar"] {{
        background-color: {sbg} !important;
    }}
    /* Headings with glow */
    h1, h2, h3, h4, h5, h6 {{
        color: {teal};
        text-shadow: 0px 0px 8px rgba(0, 191, 166, 0.4);
    }}
    /* Metric label & value */
    [data-testid="stMetricLabel"] {{
        color: {text} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {teal} !important;
    }}
    /* Sidebar menu items rounded + hover */
    .css-1d391kg, .css-1vq4p4l {{
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }}
    .css-1d391kg:hover, .css-1vq4p4l:hover {{
        background-color: rgba(0, 191, 166, 0.15) !important;
    }}
    /* Table scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-thumb {{
        background-color: {teal};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-track {{
        background-color: {bg};
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ------------------ SECRETS ------------------
BF_APP_KEY = st.secrets.get("betfair", {}).get("app_key", os.getenv("BF_APP_KEY", ""))
BF_SESSION  = st.secrets.get("betfair", {}).get("session_token", os.getenv("BF_SESSION", ""))
BF_USER     = st.secrets.get("betfair", {}).get("username", os.getenv("BF_USER", ""))
BF_PASS     = st.secrets.get("betfair", {}).get("password", os.getenv("BF_PASS", ""))

RA_USER = st.secrets.get("racing_api", {}).get("username", os.getenv("RACING_API_USERNAME", ""))
RA_PASS = st.secrets.get("racing_api", {}).get("password", os.getenv("RACING_API_PASSWORD", ""))

BF_IDENTITY_URL = "https://identitysso.betfair.com/api/login"
BF_API_URL      = "https://api.betfair.com/exchange/betting/json-rpc/v1"

# ------------------ HELPERS ------------------
def _iso_range_for_day(day: str):
    now = datetime.utcnow()
    if day == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    else:
        start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    return start.isoformat() + "Z", end.isoformat() + "Z"

@st.cache_resource
def bf_get_session_token():
    if not BF_APP_KEY:
        st.error("Betfair app_key missing in secrets.")
        return None
    if BF_SESSION:
        return BF_SESSION
    if BF_USER and BF_PASS:
        try:
            resp = requests.post(
                BF_IDENTITY_URL,
                headers={"X-Application": BF_APP_KEY, "Content-Type": "application/x-www-form-urlencoded"},
                data={"username": BF_USER, "password": BF_PASS},
                timeout=12
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "SUCCESS":
                return data.get("token")
            else:
                st.error(f"Betfair login failed: {data}")
                return None
        except Exception as e:
            st.error(f"Betfair login error: {e}")
            return None
    st.warning("No Betfair session token and no username/password provided.")
    return None

def bf_headers(token):
    return {
        "X-Application": BF_APP_KEY,
        "X-Authentication": token,
        "Content-Type": "application/json"
    }

def bf_call(method: str, params: dict, token: str):
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    r = requests.post(BF_API_URL, headers=bf_headers(token), data=json.dumps(payload), timeout=12)
    r.raise_for_status()
    out = r.json()
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["result"]

def map_region_to_country(region: str) -> str:
    if not isinstance(region, str):
        return "Unknown"
    if region.upper() == "GB": return "UK"
    if region.upper() in ("US", "USA"): return "USA"
    if region.upper() == "IE": return "Ireland"
    return region.upper()

# Odds helpers
def _best_back_and_sp(runner_book: dict):
    ex = runner_book.get("ex", {})
    atb = ex.get("availableToBack", [])
    best_back = atb[0].get("price") if atb else None
    sp = runner_book.get("sp", {})
    proj = sp.get("farPrice") or sp.get("nearPrice")
    return best_back, proj

def _hhmm(iso_str):
    try:
        return datetime.fromisoformat(iso_str.replace("Z","")).strftime("%H:%M")
    except Exception:
        return ""

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n] 

# ------------------ BETFAIR API FUNCTIONS ------------------
@st.cache_data(ttl=60)
def bf_list_win_markets(day="today"):
    token = bf_get_session_token()
    if not token:
        return []
    fr, to = _iso_range_for_day(day)
    params = {
        "filter": {
            "eventTypeIds": ["7"],
            "marketCountries": ["GB"],
            "marketTypeCodes": ["WIN"],
            "marketStartTime": {"from": fr, "to": to}
        },
        "maxResults": 200,
        "marketProjection": ["RUNNER_DESCRIPTION", "MARKET_START_TIME", "EVENT", "RUNNER_METADATA"]
    }
    try:
        return bf_call("listMarketCatalogue", params, token)
    except Exception as e:
        st.error(f"Betfair listMarketCatalogue error: {e}")
        return []

@st.cache_data(ttl=30)
def bf_list_market_books(market_ids):
    token = bf_get_session_token()
    if not token or not market_ids:
        return []
    books = []
    for chunk in _chunks(market_ids, 25):
        params = {
            "marketIds": chunk,
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "SP_PROJECTED", "SP_TRADED"],
                "virtualise": True
            }
        }
        try:
            res = bf_call("listMarketBook", params, token)
            books.extend(res)
        except Exception as e:
            st.error(f"Betfair listMarketBook error: {e}")
    return books

# ------------------ DATA BUILDERS ------------------
def _value_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    np.random.seed(42)
    df = df.copy()
    df["Win_Value"] = np.random.uniform(5, 30, len(df)).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (1.0 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅",
                   np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))
    return df

def build_live_df_betfair(day="today") -> pd.DataFrame:
    cat = bf_list_win_markets(day)
    if not cat:
        return pd.DataFrame()
    ids = [m["marketId"] for m in cat]
    books = bf_list_market_books(ids)
    price_map = {}
    for mb in books:
        mid = mb.get("marketId")
        sel_map = {}
        for rb in mb.get("runners", []):
            best_back, proj_sp = _best_back_and_sp(rb)
            sel_map[rb.get("selectionId")] = (best_back, proj_sp)
        price_map[mid] = sel_map
    rows = []
    for m in cat:
        mid = m["marketId"]
        venue = m.get("event", {}).get("venue", "Unknown")
        tstr = _hhmm(m.get("marketStartTime", ""))
        for r in m.get("runners", []):
            name = r.get("runnerName")
            sel = r.get("selectionId")
            bb, sp = price_map.get(mid, {}).get(sel, (None, None))
            if not name or bb is None:
                continue
            rows.append({
                "Country": "UK",
                "Course": venue,
                "Time": tstr,
                "Horse": name,
                "Odds": round(float(bb), 2),
                "Odds (SP proj)": round(float(sp), 2) if sp else None
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Source"] = "Betfair (live)"
    return _value_columns(df).sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=120)
def fetch_racecards_basic(day="today"):
    user = RA_USER; pwd = RA_PASS
    if not user or not pwd:
        return False, None, None, "No Racing API credentials in secrets."
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"day": day}
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(user, pwd), params=params, timeout=12)
        resp.raise_for_status()
        return True, resp.json(), resp.status_code, resp.text
    except Exception as e:
        return False, None, None, str(e)

def build_live_df_racingapi(payload: dict) -> pd.DataFrame:
    if not payload or "racecards" not in payload:
        return pd.DataFrame()
    rows = []
    for race in payload.get("racecards", []):
        region = race.get("region")
        course = race.get("course")
        off = race.get("off_time")
        for idx, rnr in enumerate(race.get("runners", [])):
            horse = rnr.get("horse")
            if not horse:
                continue
            if idx == 0:
                best_back = np.random.uniform(2.2, 3.8)
            elif idx <= 2:
                best_back = np.random.uniform(4.0, 7.0)
            elif idx <= 5:
                best_back = np.random.uniform(7.5, 15.0)
            else:
                best_back = np.random.uniform(16.0, 40.0)
            proj_sp = best_back * np.random.uniform(0.95, 1.05)
            rows.append({
                "Country": map_region_to_country(region),
                "Course": course,
                "Time": off,
                "Horse": horse,
                "Odds": round(best_back, 2),
                "Odds (SP proj)": round(proj_sp, 2)
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Source"] = "Racing API (names) + mock odds"
    return _value_columns(df).sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ------------------ EDGE BRAIN ------------------
def edgebrain_score(df: pd.DataFrame) -> pd.Series:
    return (0.5 * df["BetEdge Win %"] + 0.5 * df["Predicted Win %"]).round(1)

# ------------------ SIDEBAR NAV ------------------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Debug"],
        icons=["house", "activity", "robot", "bug"],
        default_index=0
    )

# ------------------ PAGES ------------------
if sel == "Overview":
    st.header("📊 Overview – Today's Racing")
    df_live = build_live_df_betfair("today")
    if df_live.empty:
        st.warning("Betfair feed not available. Falling back to Racing API.")
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df_live = build_live_df_racingapi(data)
    cols = st.columns(2)
    with cols[0]:
        st.metric("Races (today)", df_live["Course"].nunique() if not df_live.empty else 0)
    with cols[1]:
        st.metric("Runners (today)", len(df_live))
    if not df_live.empty:
        st.subheader("Top 20 BetEdge")
        st.bar_chart(df_live.head(20).set_index("Horse")["BetEdge Win %"])

elif sel == "Horse Racing":
    st.header("🏇 Horse Racing – Live")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = build_live_df_betfair(day)
    if df.empty:
        st.warning("Betfair feed not available. Using Racing API fallback.")
        ok, data, _, _ = fetch_racecards_basic(day)
        if ok:
            df = build_live_df_racingapi(data)
    if df.empty:
        st.error("No data available.")
    else:
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
        country = st.selectbox("Country", countries)
        courses = sorted(df["Course"].dropna().unique().tolist())
        course_filter = st.multiselect("Courses", courses, default=courses)
        filt = df.copy()
        if country != "All":
            filt = filt[filt["Country"] == country]
        if course_filter:
            filt = filt[filt["Course"].isin(course_filter)]
        min_v = int(filt["BetEdge Win %"].min())
        max_v = int(filt["BetEdge Win %"].max())
        edge_range = st.slider("🎯 Filter by BetEdge Win %", min_v, max_v, (min_v, max_v))
        filt = filt[filt["BetEdge Win %"].between(*edge_range)]
        st.dataframe(filt, use_container_width=True)

elif sel == "EdgeBrain":
    st.header("🧠 EdgeBrain Predictions")
    df = build_live_df_betfair("today")
    if df.empty:
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df = build_live_df_racingapi(data)
    if not df.empty:
        df["EdgeBrain Win %"] = edgebrain_score(df)
        st.dataframe(df[["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","Source"]], use_container_width=True)

else:
    st.header("🐞 Debug")
    token = bf_get_session_token()
    st.write("Has Betfair token:", bool(token))
    cat = bf_list_win_markets("today")
    st.write("Catalogue markets:", len(cat))
    if cat:
        st.json(cat[0])

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
