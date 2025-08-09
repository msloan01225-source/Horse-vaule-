# edgebet_app.py
# ------------------ IMPORTS ------------------
import os
import json
import math
import time
import hashlib
import random
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta, timezone
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
    .stApp {{
        background-color: {bg};
        color: {text};
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{ background-color: {sbg} !important; }}
    /* Headings with glow */
    h1, h2, h3, h4, h5, h6 {{
        color: {teal};
        text-shadow: 0px 0px 8px rgba(0, 191, 166, 0.4);
    }}
    /* Metric label & value */
    [data-testid="stMetricLabel"] {{ color: {text} !important; }}
    [data-testid="stMetricValue"] {{ color: {teal} !important; }}
    /* Generic hover affordance for sidebar items */
    [data-testid="stSidebarNav"] div:hover {{ background-color: rgba(0, 191, 166, 0.15) !important; border-radius: 8px; }}
    /* Table scrollbar */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background-color: {teal}; border-radius: 4px; }}
    ::-webkit-scrollbar-track {{ background-color: {bg}; }}
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
    # Betfair API expects UTC
    now = datetime.now(timezone.utc)
    if day == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    else:
        start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    return start.isoformat().replace("+00:00","Z"), end.isoformat().replace("+00:00","Z")

def _to_hhmm(iso_or_hhmm: str) -> str:
    """
    Accepts either an ISO8601 string (with or without Z/offset) or a plain 'HH:MM' string.
    Returns 'HH:MM' in UTC display to keep the UI consistent.
    """
    if not isinstance(iso_or_hhmm, str):
        return ""
    s = iso_or_hhmm.strip()
    # already looks like HH:MM
    if len(s) == 5 and s[2] == ":" and s[:2].isdigit() and s[3:].isdigit():
        return s
    # try robust parse
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.strftime("%H:%M")

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _stable_rand01(key: str) -> float:
    h = hashlib.sha256(key.encode()).hexdigest()
    rnd = random.Random(int(h[:8], 16))
    return rnd.random()

# ------------------ BETFAIR API FUNCTIONS ------------------
@st.cache_resource
def bf_get_session_token():
    if not BF_APP_KEY:
        st.error("Betfair app_key missing in secrets.")
        return None
    # explicit session provided
    if BF_SESSION:
        return BF_SESSION
    # login flow
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
                st.error(f"Betfair login failed.")
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

def _bf_post(method: str, params: dict, token: str):
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    r = requests.post(BF_API_URL, headers=bf_headers(token), data=json.dumps(payload), timeout=12)
    r.raise_for_status()
    return r.json()

def bf_call(method: str, params: dict, token: str):
    """
    Calls Betfair, auto-refreshing token once if session/app errors occur.
    """
    out = _bf_post(method, params, token)
    if "error" not in out:
        return out["result"]

    # handle auth errors and retry once
    code = (out["error"].get("data") or {}).get("APINGException", {}).get("errorCode")
    if code in {"INVALID_SESSION_INFORMATION", "NO_APP_KEY"}:
        # refresh session
        bf_get_session_token.clear()  # clear cache_resource
        new_token = bf_get_session_token()
        if not new_token:
            raise RuntimeError(out["error"])
        out2 = _bf_post(method, params, new_token)
        if "error" in out2:
            raise RuntimeError(out2["error"])
        return out2["result"]
    # other error
    raise RuntimeError(out["error"])

def map_region_to_country(region: str) -> str:
    if not isinstance(region, str):
        return "Unknown"
    r = region.upper()
    if r == "GB": return "UK"
    if r in ("US", "USA"): return "USA"
    if r == "IE": return "Ireland"
    return r

def _best_back_and_sp(runner_book: dict):
    ex = runner_book.get("ex", {}) or {}
    atb = ex.get("availableToBack", []) or []
    best_back = atb[0].get("price") if atb else None
    sp = runner_book.get("sp", {}) or {}
    proj = sp.get("farPrice") or sp.get("nearPrice")
    return best_back, proj

# ------------------ DATA BUILDERS ------------------
def _value_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    key = (df["Course"].astype(str) + "|" + df["Time"].astype(str) + "|" + df["Horse"].astype(str))
    base = key.apply(_stable_rand01)
    df["Win_Value"] = (5 + base * 25).round(1)   # 5–30
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (100.0 / df["Odds"].clip(lower=1.01)).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅",
                   np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))
    return df

@st.cache_data(ttl=60)
def bf_list_win_markets(day="today"):
    token = bf_get_session_token()
    if not token:
        return []
    fr, to = _iso_range_for_day(day)
    params = {
        "filter": {
            "eventTypeIds": ["7"],             # Horse Racing
            "marketCountries": ["GB"],         # extend later if needed
            "marketTypeCodes": ["WIN"],
            "marketStartTime": {"from": fr, "to": to}
        },
        "maxResults": 200,
        "marketProjection": ["RUNNER_DESCRIPTION", "MARKET_START_TIME", "EVENT", "RUNNER_METADATA"]
    }
    try:
        return bf_call("listMarketCatalogue", params, token)
    except Exception as e:
        st.error("Betfair listMarketCatalogue error.")
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
        except Exception:
            # soft-fail this chunk
            st.warning("A Betfair price chunk failed; continuing.")
            continue
        # be nice; if you increase chunk size, consider sleeping briefly
        # time.sleep(0.05)
    return books

def build_live_df_betfair(day="today") -> pd.DataFrame:
    cat = bf_list_win_markets(day)
    if not cat:
        return pd.DataFrame()
    ids = [m.get("marketId") for m in cat if m.get("marketId")]
    books = bf_list_market_books(ids) if ids else []
    price_map = {}
    for mb in books:
        mid = mb.get("marketId")
        if not mid:
            continue
        sel_map = {}
        for rb in mb.get("runners", []) or []:
            best_back, proj_sp = _best_back_and_sp(rb)
            sel_map[rb.get("selectionId")] = (best_back, proj_sp)
        price_map[mid] = sel_map

    rows = []
    for m in cat:
        mid = m.get("marketId")
        venue = (m.get("event", {}) or {}).get("venue", "Unknown")
        tstr = _to_hhmm(m.get("marketStartTime", ""))
        for r in (m.get("runners", []) or []):
            name = r.get("runnerName")
            sel = r.get("selectionId")
            bb, sp = (price_map.get(mid, {}) or {}).get(sel, (None, None))
            bb_val = float(bb) if bb is not None else None
            sp_val = float(sp) if sp is not None else None
            if not name or bb_val is None:
                continue
            rows.append({
                "Country": "UK",
                "Course": venue,
                "Time": tstr,
                "Horse": name,
                "Odds": round(bb_val, 2),
                "Odds (SP proj)": round(sp_val, 2) if sp_val is not None else None
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Source"] = "Betfair (live)"
    return _value_columns(df).sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# --------------- RACING API (FALLBACK) ---------------
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
    for race in payload.get("racecards", []) or []:
        region = race.get("region")
        course = race.get("course")
        off = _to_hhmm(race.get("off_time", ""))
        for idx, rnr in enumerate(race.get("runners", []) or []):
            horse = rnr.get("horse")
            if not horse:
                continue
            # mock odds by rank-ish bucket
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
                "Odds": round(float(best_back), 2),
                "Odds (SP proj)": round(float(proj_sp), 2)
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Source"] = "Racing API (names) + mock odds"
    return _value_columns(df).sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ------------------ EDGE BRAIN ------------------
def edgebrain_score(df: pd.DataFrame) -> pd.Series:
    p = (1.0 / df["Odds"].clip(lower=1.01))
    eb = 100 * (0.6 * p + 0.4 * (df["BetEdge Win %"] / 100.0))
    return eb.round(1)

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
        st.metric("Races (today)", int(df_live["Course"].nunique()) if not df_live.empty else 0)
    with cols[1]:
        st.metric("Runners (today)", int(len(df_live)))

    if not df_live.empty:
        st.subheader("Top 20 BetEdge")
        top = df_live.sort_values("BetEdge Win %", ascending=False).head(20).copy()
        top["Label"] = top["Horse"] + " (" + top["Course"].astype(str) + " " + top["Time"].astype(str) + ")"
        # bar_chart expects a numeric Series or DataFrame indexed sensibly
        st.bar_chart(top.set_index("Label")[["BetEdge Win %"]])

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

        if filt.empty:
            st.info("No selections match your filters.")
        else:
            min_v = int(np.floor(filt["BetEdge Win %"].min()))
            max_v = int(np.ceil(filt["BetEdge Win %"].max()))
            edge_range = st.slider("🎯 Filter by BetEdge Win %", min_v, max_v, (min_v, max_v))
            filt = filt[filt["BetEdge Win %"].between(*edge_range)]
            if filt.empty:
                st.info("No selections in the chosen BetEdge range.")
            else:
                st.dataframe(filt, use_container_width=True)

elif sel == "EdgeBrain":
    st.header("🧠 EdgeBrain Predictions")
    df = build_live_df_betfair("today")
    if df.empty:
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df = build_live_df_racingapi(data)
    if not df.empty:
        df = df.copy()
        df["EdgeBrain Win %"] = edgebrain_score(df)
        st.dataframe(df[["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","Source"]], use_container_width=True)
    else:
        st.info("No data available for EdgeBrain.")

else:
    st.header("🐞 Debug")
    token = bf_get_session_token()
    st.write("Has Betfair token:", bool(token))
    cat = bf_list_win_markets("today")
    st.write("Catalogue markets:", len(cat))
    if cat:
        st.json(cat[0])

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
