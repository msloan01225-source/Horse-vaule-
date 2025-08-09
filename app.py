# app.py — EdgeBet (Betfair live odds + Racing API fallback)

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

# ------------------ CONFIG / THEME ------------------
st.set_page_config(page_title="EdgeBet – Live Racing", layout="wide")

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

mode_toggle = st.sidebar.radio("Theme", ["dark", "light"], index=0 if st.session_state.theme_mode == "dark" else 1)
st.session_state.theme_mode = mode_toggle

if st.session_state.theme_mode == "dark":
    st.markdown("""
    <style>
    body { background-color: #0d1b1e; color: #f2f2f2; }
    h1, h2, h3, h4, h5, h6 { color: #00bfa6; }
    .block-container { padding-top: 1.2rem; }
    [data-testid="stMetricValue"] { color: #00bfa6 !important; }
    .stRadio > div { flex-wrap: wrap; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body { background-color: #ffffff; color: #000000; }
    h1, h2, h3, h4, h5, h6 { color: #00796b; }
    .block-container { padding-top: 1.2rem; }
    [data-testid="stMetricValue"] { color: #00796b !important; }
    .stRadio > div { flex-wrap: wrap; }
    </style>
    """, unsafe_allow_html=True)

# Force highlight colors to show in st.dataframe for both dark and light themes
st.markdown("""
<style>
.dataframe td, .stDataFrame td {
    background-color: inherit !important;
}
</style>
""", unsafe_allow_html=True)

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

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

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

def _best_back_price(runner_book: dict):
    ex = runner_book.get("ex", {})
    atb = ex.get("availableToBack", [])
    if atb:
        return atb[0].get("price")
    sp = runner_book.get("sp", {})
    proj = sp.get("farPrice") or sp.get("nearPrice")
    return proj

def _hhmm(iso_str):
    try:
        return datetime.fromisoformat(iso_str.replace("Z","")).strftime("%H:%M")
    except Exception:
        return ""

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
            sel_map[rb.get("selectionId")] = _best_back_price(rb)
        price_map[mid] = sel_map

    rows = []
    for m in cat:
        mid   = m["marketId"]
        venue = m.get("event", {}).get("venue", "Unknown")
        tstr  = _hhmm(m.get("marketStartTime", ""))
        for r in m.get("runners", []):
            name = r.get("runnerName")
            sel  = r.get("selectionId")
            price = None
            if mid in price_map:
                price = price_map[mid].get(sel)
            if not name or price is None:
                continue
            rows.append({
                "Country": "UK",
                "Course": venue,
                "Time": tstr,
                "Horse": name,
                "Odds": round(float(price), 2)
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

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

# ------ Racing API (fallback) ------
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

# ------ RaceKey fix & EdgeBrain ------
def ensure_race_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    comp = (
        df.get("Country", "").astype(str) + "|" +
        df.get("Course", "").astype(str) + "|" +
        df.get("Time", "").astype(str)
    )
    if "EventId" in df.columns:
        eid = df["EventId"]
        rk = np.where(eid.notna(), eid.astype(str), comp)
    else:
        rk = comp
    df["RaceKey"] = rk
    return df

def add_edgebrain_features(df: pd.DataFrame) -> pd.DataFrame:
    eb = ensure_race_key(df)
    odds = eb["Odds"].replace(0, np.nan)
    eb["imp_ex"] = 1.0 / odds
    eb["imp_sp"] = 1.0 / odds
    eb["field_size"] = eb.groupby("RaceKey")["Horse"].transform("count")
    eb["fav_rank"] = eb.groupby("RaceKey")["Odds"].rank(method="min", ascending=True)
    eb["gap_to_fav"] = eb["Odds"] - eb.groupby("RaceKey")["Odds"].transform("min")
    eb["p_blend"] = (eb["imp_ex"] + eb["imp_sp"]) / 2.0
    eb["p_blend"] = eb["p_blend"].fillna(0.0)
    return eb

def edgebrain_softmax(df_feat: pd.DataFrame) -> pd.Series:
    lin = (0.6 * df_feat["p_blend"]) - (0.1 * df_feat["fav_rank"]) - (0.3 * df_feat["gap_to_fav"])
    def _softmax_group(g: pd.DataFrame) -> pd.Series:
        x = g["lin"].to_numpy()
        x = x - np.max(x)
        p = np.exp(x)
        p /= p.sum() if p.sum() else 1.0
        return pd.Series(p, index=g.index)
    f = df_feat.copy()
    f["lin"] = lin.fillna(lin.median() if np.isfinite(lin.median()) else 0.0)
    return f.groupby("RaceKey", group_keys=False).apply(_softmax_group).clip(0, 1)

def edgebrain_score(df: pd.DataFrame) -> pd.Series:
    feats = add_edgebrain_features(df)
    p = edgebrain_softmax(feats)
    return (p * 100).round(1)

def get_history_data():
    np.random.seed(123)
    return pd.DataFrame({
        "Horse": [],
        "hist_win_pct": [],
        "hist_place_pct": [],
        "course_win_pct": [],
        "jockey_win_pct": []
    })

def merge_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_race_key(df)
    hist_df = get_history_data()
    if hist_df.empty:
        for col in ["hist_win_pct","hist_place_pct","course_win_pct","jockey_win_pct"]:
            df[col] = 0.0
        return df
    return df.merge(hist_df, on="Horse", how="left")

def edgebrain_plus_score(df: pd.DataFrame) -> pd.Series:
    df = ensure_race_key(df)
    feats = add_edgebrain_features(df)
    feats["hist_factor"] = (
        0.3 * feats.get("hist_win_pct", 0) +
        0.2 * feats.get("hist_place_pct", 0) +
        0.3 * feats.get("course_win_pct", 0) +
        0.2 * feats.get("jockey_win_pct", 0)
    ) / 100.0
    feats["p_blend"] = feats["p_blend"] * (1 + feats["hist_factor"])
    p = edgebrain_softmax(feats)
    return (p * 100).round(1)

# ------ SIDEBAR NAV ------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Debug"],
        icons=["house", "activity", "robot", "bug"],
        default_index=0
    )

# ------ PAGES ------
if sel == "Overview":
    st.title("📊 EdgeBet – Live Racing")
    df_live = build_live_df_betfair("today")
    if df_live.empty:
        st.warning("Betfair feed not available. Falling back to Racing API names.")
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df_live = build_live_df_racingapi(data)
    st.metric("Races (today)", df_live["Course"].nunique())
    st.metric("Runners (today)", len(df_live))
    if not df_live.empty:
        st.subheader("Top 20 BetEdge (today)")
        st.bar_chart(df_live.head(20).set_index("Horse")["BetEdge Win %"])

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

    with st.expander("🔎 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
            country = st.selectbox("Country", countries)
        with col2:
            bookie = st.selectbox("Bookmaker", ["All", "Betfair"])
        with col3:
            courses = sorted(df["Course"].dropna().unique().tolist())
            course_filter = st.multiselect("Courses", courses, default=courses)

        if country != "All":
            df = df[df["Country"] == country]
        if course_filter:
            df = df[df["Course"].isin(course_filter)]

        min_v, max_v = int(df["BetEdge Win %"].min()), int(df["BetEdge Win %"].max())
        edge_range = st.slider("🎯 Filter by BetEdge Win %", min_v, max_v, (min_v, max_v))
        df = df[df["BetEdge Win %"].between(*edge_range)]

    if df.empty:
        st.warning("No horses match your filters.")
    else:
        view = st.radio("View", ["📊 Charts", "📋 Tables"], horizontal=True)
        if view == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(df.sort_values("BetEdge Win %", ascending=False).head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(df.sort_values("BetEdge Place %", ascending=False).head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.dataframe(df[["Country","Course","Time","Horse","Odds","BetEdge Win %","BetEdge Place %","Source"]], use_container_width=True)

elif sel == "EdgeBrain":
    st.title("🧠 EdgeBrain – Predictions")
    top_cols = st.columns([1, 2])
    with top_cols[0]:
        day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    with top_cols[1]:
        model_view = st.radio(
            "Model view",
            ["EdgeBrain v1", "EdgeBrain+ (history)", "Both"],
            horizontal=True
        )

    df = build_live_df_betfair(day)
    if df.empty:
        ok, data, _, _ = fetch_racecards_basic(day)
        if ok:
            df = build_live_df_racingapi(data)

    if df.empty:
        st.warning("No data available.")
    else:
        df["EdgeBrain Win %"] = edgebrain_score(df)
        df_hist = merge_history_features(df)
        df["EdgeBrain+ Win %"] = edgebrain_plus_score(df_hist)

        df["Risk_v1"] = np.where(df["EdgeBrain Win %"] >= 25, "✅", np.where(df["EdgeBrain Win %"] >= 15, "⚠️", "❌"))
        df["Risk+"]   = np.where(df["EdgeBrain+ Win %"] >= 25, "✅", np.where(df["EdgeBrain+ Win %"] >= 15, "⚠️", "❌"))

        if model_view == "EdgeBrain v1":
            st.bar_chart(df.sort_values("EdgeBrain Win %", ascending=False).head(20).set_index("Horse")["EdgeBrain Win %"])
            st.dataframe(df[["Country","Course","Time","Horse","Odds","EdgeBrain Win %","Risk_v1","Source"]], use_container_width=True)
        elif model_view == "EdgeBrain+ (history)":
            st.bar_chart(df.sort_values("EdgeBrain+ Win %", ascending=False).head(20).set_index("Horse")["EdgeBrain+ Win %"])
            st.dataframe(df[["Country","Course","Time","Horse","Odds","EdgeBrain+ Win %","Risk+","Source"]], use_container_width=True)
        else:
            def highlight_row(row):
                teal_dark = "background-color: #00bfa6 !important; color: #000000 !important;"
                teal_light = "background-color: #00796b !important; color: #ffffff !important;"
                gold = "background-color: #ffd700 !important; color: #000000 !important;"
                empty = ""
                teal = teal_dark if st.session_state.theme_mode == "dark" else teal_light
                if pd.isna(row["EdgeBrain Win %"]) or pd.isna(row["EdgeBrain+ Win %"]):
                    return [empty] * len(row)
                if row["EdgeBrain+ Win %"] > row["EdgeBrain Win %"]:
                    return [teal] * len(row)
                elif row["EdgeBrain Win %"] > row["EdgeBrain+ Win %"]:
                    return [gold] * len(row)
                return [empty] * len(row)
            styled_df = df[["Country","Course","Time","Horse","Odds","EdgeBrain Win %","Risk_v1","EdgeBrain+ Win %","Risk+","Source"]].style.apply(highlight_row, axis=1)
            st.dataframe(styled_df, use_container_width=True)

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

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
