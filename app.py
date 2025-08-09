# app.py — EdgeBet (Betfair live odds + Racing API fallback) — improved

import os
import json
import math
import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from streamlit_option_menu import option_menu

# ------------------ CONFIG / THEME ------------------
st.set_page_config(page_title="EdgeBet – Live Racing", layout="wide")

st.markdown("""
<style>
body { background-color: #111111; color: #f2f2f2; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.block-container { padding-top: 1.2rem; }
[data-testid="stMetricValue"] { color: #00ffcc !important; }
.stRadio > div { flex-wrap: wrap; }
</style>
""", unsafe_allow_html=True)

# ------------------ CONSTANTS ------------------
LONDON_TZ = ZoneInfo("Europe/London")
UTC_TZ = ZoneInfo("UTC")

BF_IDENTITY_URL = "https://identitysso.betfair.com/api/login"
BF_API_URL      = "https://api.betfair.com/exchange/betting/json-rpc/v1"

# ------------------ SECRETS ------------------
BF_APP_KEY = st.secrets.get("betfair", {}).get("app_key", os.getenv("BF_APP_KEY", ""))
BF_SESSION  = st.secrets.get("betfair", {}).get("session_token", os.getenv("BF_SESSION", ""))
BF_USER     = st.secrets.get("betfair", {}).get("username", os.getenv("BF_USER", ""))
BF_PASS     = st.secrets.get("betfair", {}).get("password", os.getenv("BF_PASS", ""))

RA_USER = st.secrets.get("racing_api", {}).get("username", os.getenv("RACING_API_USERNAME", ""))
RA_PASS = st.secrets.get("racing_api", {}).get("password", os.getenv("RACING_API_PASSWORD", ""))

# ------------------ HELPERS ------------------
def _iso_range_for_day(day: str):
    """Return [from,to) window in ISO8601 UTC for 'today' or 'tomorrow'."""
    now_utc = datetime.now(tz=UTC_TZ)
    if day == "today":
        start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    else:  # 'tomorrow'
        start = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    # Betfair likes Z
    return start.isoformat().replace("+00:00","Z"), end.isoformat().replace("+00:00","Z")

def _hhmm(iso_str: str) -> str:
    """Format Betfair ISO8601 (UTC, 'Z') -> Europe/London HH:MM safely."""
    try:
        dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt_utc.astimezone(LONDON_TZ).strftime("%H:%M")
    except Exception:
        return ""

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ------------------ BETFAIR CORE ------------------
def _clear_cached_bf_session():
    try:
        bf_get_session_token.clear()  # clear st.cache_resource entry
    except Exception:
        pass

@st.cache_resource
def bf_get_session_token():
    """Return a Betfair session token from secrets or by logging in."""
    if not BF_APP_KEY:
        st.error("Betfair app_key missing in secrets.")
        return None
    if BF_SESSION:
        return BF_SESSION  # trust pre-supplied session
    if BF_USER and BF_PASS:
        try:
            resp = requests.post(
                BF_IDENTITY_URL,
                headers={
                    "X-Application": BF_APP_KEY,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"username": BF_USER, "password": BF_PASS},
                timeout=12
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "SUCCESS":
                return data.get("token")
            st.error(f"Betfair login failed: {data}")
            return None
        except Exception as e:
            st.error(f"Betfair login error: {e}")
            return None
    st.warning("No Betfair session token and no username/password provided.")
    return None

def bf_headers(token: str):
    return {
        "X-Application": BF_APP_KEY,
        "X-Authentication": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection": "keep-alive",
    }

def bf_call(method: str, params: dict, token: str, *, retries: int = 2):
    """JSON-RPC call to Betfair API-NG with simple retry and auto token refresh."""
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    backoff = 0.8
    for attempt in range(retries + 1):
        try:
            r = requests.post(BF_API_URL, headers=bf_headers(token), data=json.dumps(payload), timeout=12)
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < retries:
                    time.sleep(backoff)
                    backoff *= 1.6
                    continue
            r.raise_for_status()
            out = r.json()
            if "error" in out:
                code = out["error"].get("data", {}).get("APINGException", {}).get("errorCode")
                if code in ("INVALID_SESSION_INFORMATION", "NO_SESSION"):
                    if attempt < retries:
                        _clear_cached_bf_session()
                        token = bf_get_session_token() or ""
                        if not token:
                            raise RuntimeError("Unable to refresh Betfair session.")
                        continue
                raise RuntimeError(f"Betfair API error: {out['error']}")
            return out["result"]
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 1.6
                continue
            raise

@st.cache_data(ttl=60)
def bf_list_win_markets(day="today"):
    """List GB Win markets for the day."""
    token = bf_get_session_token()
    if not token:
        return []
    fr, to = _iso_range_for_day(day)
    params = {
        "filter": {
            "eventTypeIds": ["7"],             # Horse Racing
            "marketCountries": ["GB"],         # Great Britain
            "marketTypeCodes": ["WIN"],
            "marketStartTime": {"from": fr, "to": to}
        },
        "sort": "FIRST_TO_START",
        "maxResults": 200,
        "marketProjection": [
            "RUNNER_DESCRIPTION",
            "MARKET_START_TIME",
            "EVENT",
            "RUNNER_METADATA",
            "MARKET_DESCRIPTION"
        ]
    }
    try:
        return bf_call("listMarketCatalogue", params, token)
    except Exception as e:
        st.error(f"Betfair listMarketCatalogue error: {e}")
        return []

@st.cache_data(ttl=30)
def bf_list_market_books(market_ids):
    """Get market books (live prices) in chunks."""
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

def _best_back_price(runner_book: dict):
    """Return best availableToBack price or projected SP if none; ensure float or None."""
    ex = runner_book.get("ex") or {}
    atb = ex.get("availableToBack") or []
    price = atb[0].get("price") if atb else None
    if price is None:
        sp = runner_book.get("sp") or {}
        price = sp.get("farPrice") or sp.get("nearPrice")
    try:
        return float(price) if price is not None and not (isinstance(price, float) and math.isnan(price)) else None
    except Exception:
        return None

def build_live_df_betfair(day="today") -> pd.DataFrame:
    """Build a live DataFrame directly from Betfair catalogue + books."""
    with st.spinner("Fetching Betfair markets..."):
        cat = bf_list_win_markets(day)
    if not cat:
        return pd.DataFrame()

    ids = [m["marketId"] for m in cat]
    with st.spinner("Fetching live prices..."):
        books = bf_list_market_books(ids)

    # map books by marketId -> {selectionId: bestPrice}
    price_map = {}
    for mb in books or []:
        mid = mb.get("marketId")
        sel_map = {}
        for rb in (mb.get("runners") or []):
            price = _best_back_price(rb)
            if price is not None:
                sel_map[rb.get("selectionId")] = price
        price_map[mid] = sel_map

    rows = []
    for m in cat:
        mid   = m["marketId"]
        venue = m.get("event", {}).get("venue", "Unknown")
        tstr  = _hhmm(m.get("marketStartTime", ""))
        for r in (m.get("runners") or []):
            name = r.get("runnerName")
            sel  = r.get("selectionId")
            price = price_map.get(mid, {}).get(sel)
            if not name or price is None:
                continue
            rows.append({
                "Country": "UK",
                "Course": venue,
                "Time": tstr,
                "Horse": name,
                "Odds": round(price, 2)
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Value metrics (placeholder logic)
    np.random.seed(42)
    df["Win_Value"] = np.random.uniform(5, 30, len(df)).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (1.0 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅", np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))
    df["Source"] = "Betfair (live)"
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ------ Racing API (fallback for names if Betfair is down) ------
@st.cache_data(ttl=120)
def fetch_racecards_basic(day="today"):
    user = RA_USER; pwd = RA_PASS
    if not user or not pwd:
        return False, None, None, "No Racing API credentials in secrets."
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"day": day}
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(user, pwd), params=params, timeout=12)
        status = resp.status_code; text = resp.text
        resp.raise_for_status()
        return True, resp.json(), status, text
    except requests.HTTPError as e:
        return False, None, e.response.status_code if e.response else None, e.response.text if e.response else str(e)
    except Exception as e:
        return False, None, None, str(e)

def map_region_to_country(region: str) -> str:
    if not isinstance(region, str): return "Unknown"
    if region.upper() == "GB": return "UK"
    if region.upper() in ("US", "USA"): return "USA"
    if region.upper() == "IE": return "Ireland"
    return region.upper()

def build_live_df_racingapi(payload: dict) -> pd.DataFrame:
    if not payload or "racecards" not in payload:
        return pd.DataFrame()
    rows = []
    for race in payload.get("racecards", []):
        region = race.get("region")
        course = race.get("course")
        off    = race.get("off_time")
        for rnr in race.get("runners", []):
            horse = rnr.get("horse")
            if not horse: continue
            rows.append({
                "Country": map_region_to_country(region),
                "Course": course,
                "Time": off,
                "Horse": horse
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # As fallback, mock odds so UI keeps working
    np.random.seed(42)
    df["Odds"] = np.random.uniform(2, 10, len(df)).round(2)
    df["Win_Value"] = np.random.uniform(5, 30, len(df)).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (1.0 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅", np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))
    df["Source"] = "Racing API (names) + mock odds"
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ------------------ EDGE BRAIN (stub) ------------------
def edgebrain_score_stub(df: pd.DataFrame) -> pd.Series:
    return (0.5 * df["BetEdge Win %"] + 0.5 * df["Predicted Win %"]).round(1)

# ------------------ SIDEBAR NAV ------------------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Debug"],
        icons=["house", "activity", "robot", "bug"],
        default_index=0
    )

# ------------------ OVERVIEW ------------------
if sel == "Overview":
    st.title("📊 EdgeBet – Live Racing (Betfair)")
    df_live = build_live_df_betfair("today")
    if df_live.empty:
        st.warning("Betfair feed not available right now. Falling back to Racing API names.")
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df_live = build_live_df_racingapi(data)
    colA, colB = st.columns(2)
    with colA:
        st.metric("Races (today)", int(df_live["Course"].nunique()) if not df_live.empty else 0)
    with colB:
        st.metric("Runners (today)", int(len(df_live)) if not df_live.empty else 0)
    st.caption("All times shown in Europe/London.")
    if not df_live.empty:
        st.subheader("Top 20 BetEdge (today)")
        st.bar_chart(df_live.head(20).set_index("Horse")["BetEdge Win %"], use_container_width=True)

# ------------------ HORSE RACING ------------------
elif sel == "Horse Racing":
    st.title("🏇 Horse Racing – Live")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)

    df = build_live_df_betfair(day)
    if df.empty:
        st.warning("Betfair feed not available. Using Racing API names as fallback.")
        ok, data, _, _ = fetch_racecards_basic(day)
        if ok:
            df = build_live_df_racingapi(data)
        else:
            st.error("No data available.")
            st.stop()

    st.caption("All times shown in Europe/London.")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
        country = st.selectbox("Country", countries, index=0)
    with col2:
        # Placeholder: bookies would come from a bookmaker odds feed; keep as a UI filter only
        bookie = st.selectbox("Bookmaker", ["All", "SkyBet", "Bet365", "Betfair"], index=0)
    with col3:
        courses = sorted(df["Course"].dropna().unique().tolist())
        course_filter = st.multiselect("Courses", courses, default=courses)

    # Apply filters
    filt = df.copy()
    if country != "All":
        filt = filt[filt["Country"] == country]
    if course_filter:
        filt = filt[filt["Course"].isin(course_filter)]

    # Range slider + view toggle
    min_v = int(filt["BetEdge Win %"].min()) if not filt.empty else 0
    max_v = int(filt["BetEdge Win %"].max()) if not filt.empty else 0
    if min_v > max_v:
        min_v, max_v = max_v, min_v
    edge_range = st.slider("🎯 Filter by BetEdge Win %", min_v, max_v, (min_v, max_v))
    filt = filt[filt["BetEdge Win %"].between(*edge_range)]

    view = st.radio("View", ["📊 Charts", "📋 Tables"], horizontal=True)

    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        if view == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(
                filt.sort_values("BetEdge Win %", ascending=False)
                    .head(20).set_index("Horse")["BetEdge Win %"],
                use_container_width=True
            )
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(
                filt.sort_values("BetEdge Place %", ascending=False)
                    .head(20).set_index("Horse")["BetEdge Place %"],
                use_container_width=True
            )
        else:
            st.subheader("Full Runner Table")
            st.dataframe(
                filt[["Country", "Course", "Time", "Horse", "Odds", "BetEdge Win %", "BetEdge Place %", "Source"]],
                use_container_width=True
            )

# ------------------ EDGE BRAIN ------------------
elif sel == "EdgeBrain":
    st.title("🧠 EdgeBrain – Live Scoring (stub)")
    df = build_live_df_betfair("today")
    if df.empty:
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df = build_live_df_racingapi(data)
    if df.empty:
        st.warning("No data available.")
    else:
        df["EdgeBrain Win %"] = edgebrain_score_stub(df)
        st.caption("All times shown in Europe/London.")
        st.subheader("Top 20 EdgeBrain Win %")
        st.bar_chart(
            df.sort_values("EdgeBrain Win %", ascending=False)
              .head(20).set_index("Horse")["EdgeBrain Win %"],
            use_container_width=True
        )
        st.dataframe(
            df[["Country", "Course", "Time", "Horse", "Odds", "EdgeBrain Win %", "Risk", "Source"]],
            use_container_width=True
        )

# ------------------ DEBUG ------------------
else:
    st.title("🐞 Debug")
    token = bf_get_session_token()
    st.write("Has Betfair token:", bool(token))
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    cat = bf_list_win_markets(day)
    st.write("Catalogue markets:", len(cat))
    if cat:
        st.json(cat[0])
    ids = [m["marketId"] for m in cat][:10]
    books = bf_list_market_books(ids)
    st.write("Books fetched:", len(books))
    if books:
        st.json(books[0])

st.caption(f"All times Europe/London • Last updated {datetime.now(tz=UTC_TZ).strftime('%Y-%m-%d %H:%M UTC')}")
