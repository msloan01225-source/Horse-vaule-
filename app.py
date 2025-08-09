# app.py — EdgeBet (Betfair live odds + Racing API fallback)
# Real EX odds + PLACE odds + EX↔SP blend + multi-country + timezone selector

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
UTC_TZ = ZoneInfo("UTC")

TZ_OPTIONS = {
    "Europe/London": ZoneInfo("Europe/London"),
    "Europe/Dublin": ZoneInfo("Europe/Dublin"),
    "Australia/Sydney": ZoneInfo("Australia/Sydney"),
    "US/Eastern": ZoneInfo("US/Eastern"),
    "US/Pacific": ZoneInfo("US/Pacific"),
    "UTC": ZoneInfo("UTC"),
}

COUNTRY_NAME = {
    "GB": "UK",
    "IE": "Ireland",
    "AU": "Australia",
    "US": "USA",
}

BF_IDENTITY_URL = "https://identitysso.betfair.com/api/login"
BF_API_URL      = "https://api.betfair.com/exchange/betting/json-rpc/v1"

# ------------------ SECRETS ------------------
BF_APP_KEY = st.secrets.get("betfair", {}).get("app_key", os.getenv("BF_APP_KEY", ""))
BF_SESSION  = st.secrets.get("betfair", {}).get("session_token", os.getenv("BF_SESSION", ""))
BF_USER     = st.secrets.get("betfair", {}).get("username", os.getenv("BF_USER", ""))
BF_PASS     = st.secrets.get("betfair", {}).get("password", os.getenv("BF_PASS", ""))

RA_USER = st.secrets.get("racing_api", {}).get("username", os.getenv("RACING_API_USERNAME", ""))
RA_PASS = st.secrets.get("racing_api", {}).get("password", os.getenv("RACING_API_PASSWORD", ""))

# ------------------ TIME HELPERS ------------------
def _iso_range_for_day(day: str):
    """Return [from,to) window in ISO8601 UTC for 'today' or 'tomorrow'."""
    now_utc = datetime.now(tz=UTC_TZ)
    if day == "today":
        start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    else:  # 'tomorrow'
        start = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    return start.isoformat().replace("+00:00","Z"), end.isoformat().replace("+00:00","Z")

def _fmt_time(iso_str: str, tz: ZoneInfo) -> str:
    """Format Betfair ISO8601 (UTC, 'Z') -> HH:MM in chosen tz."""
    try:
        dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt_utc.astimezone(tz).strftime("%H:%M")
    except Exception:
        return ""

def _mins_to_off(iso_str: str) -> float:
    try:
        off_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now_utc = datetime.now(tz=UTC_TZ)
        dmins = (off_utc - now_utc).total_seconds() / 60.0
        return max(-9999.0, dmins)
    except Exception:
        return 9999.0

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
        except requests.RequestException:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 1.6
                continue
            raise

# -------- Generic market catalogue fetch (WIN or PLACE) ----------
@st.cache_data(ttl=60)
def bf_list_markets(day="today", market_types=("WIN",), countries=("GB",)):
    """List racing markets for the day, parameterized by market type(s) and country(ies)."""
    token = bf_get_session_token()
    if not token:
        return []
    fr, to = _iso_range_for_day(day)
    params = {
        "filter": {
            "eventTypeIds": ["7"],                   # Horse Racing
            "marketCountries": list(countries),
            "marketTypeCodes": list(market_types),   # e.g., ["WIN"] or ["PLACE"]
            "marketStartTime": {"from": fr, "to": to}
        },
        "sort": "FIRST_TO_START",
        "maxResults": 800,
        "marketProjection": [
            "RUNNER_DESCRIPTION",
            "MARKET_START_TIME",
            "EVENT",
            "RUNNER_METADATA",
            "MARKET_DESCRIPTION",
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

def _sp_price(runner_book: dict):
    """Projected SP if available, else None."""
    sp = runner_book.get("sp") or {}
    price = sp.get("farPrice") or sp.get("nearPrice")
    try:
        return float(price) if price is not None else None
    except Exception:
        return None

# ------------- Probability helpers (normalize for overround) -------------
def _normalize_probs(prices: pd.Series) -> pd.Series:
    """Given decimal odds prices, compute normalized implied probabilities."""
    prices = prices.replace(0, np.nan)
    raw = 1.0 / prices
    s = raw.sum(skipna=True)
    if s and np.isfinite(s) and s > 0:
        return (raw / s).fillna(0.0)
    return raw.fillna(0.0)

# ------------------ BUILD DATAFRAME (WIN + PLACE, blended model) ------------------
def build_live_df_betfair(day="today", sel_countries=("GB", "IE", "AU", "US"), display_tz=ZoneInfo("Europe/London")) -> pd.DataFrame:
    """Build a live DataFrame from Betfair for WIN and PLACE markets with EX↔SP blend, multi-country."""
    # 1) Fetch catalogues
    with st.spinner("Fetching Betfair WIN markets..."):
        cat_win = bf_list_markets(day, market_types=("WIN",), countries=sel_countries)
    if not cat_win:
        return pd.DataFrame()

    # Index WIN catalogue
    win_markets = {m["marketId"]: m for m in cat_win}
    # PLACE catalogue
    with st.spinner("Fetching Betfair PLACE markets..."):
        cat_place = bf_list_markets(day, market_types=("PLACE",), countries=sel_countries)

    # Map event -> list of PLACE marketIds
    place_by_event = {}
    for m in cat_place or []:
        place_by_event.setdefault(m["event"]["id"], []).append(m["marketId"])

    # 2) Fetch market books
    with st.spinner("Fetching live EX prices + projected SP..."):
        books_win   = bf_list_market_books(list(win_markets.keys()))
        books_place = bf_list_market_books([mid for mids in place_by_event.values() for mid in mids])

    # 3) Build price lookup maps
    # WIN: (eventId, runnerName) -> data
    win_prices = {}
    for mb in books_win or []:
        mid = mb.get("marketId")
        mcat = win_markets.get(mid)
        if not mcat: continue
        event = mcat.get("event", {})
        event_id = event.get("id")
        start_iso = mcat.get("marketStartTime","")
        venue = event.get("venue", "Unknown")
        ccode = event.get("countryCode") or "GB"  # default to GB if missing
        country = COUNTRY_NAME.get(ccode, ccode)
        for rb in (mb.get("runners") or []):
            name = None
            for rcat in (mcat.get("runners") or []):
                if rcat.get("selectionId") == rb.get("selectionId"):
                    name = rcat.get("runnerName"); break
            if not name: continue
            ex = _best_back_price(rb)
            sp = _sp_price(rb)
            win_prices[(event_id, name)] = {
                "ex": ex, "sp": sp, "start": start_iso, "venue": venue, "country": country
            }

    # PLACE: choose the earliest PLACE market per event
    place_prices = {}
    place_cat_by_mid = {m["marketId"]: m for m in cat_place or []}
    chosen_place_mids = []
    for eid, mids in place_by_event.items():
        chosen_place_mids.append(mids[0])

    for mb in books_place or []:
        if mb.get("marketId") not in chosen_place_mids:
            continue
        mcat = place_cat_by_mid.get(mb.get("marketId"))
        if not mcat: continue
        eid = mcat["event"]["id"]
        for rb in (mb.get("runners") or []):
            name = None
            for rcat in (mcat.get("runners") or []):
                if rcat.get("selectionId") == rb.get("selectionId"):
                    name = rcat.get("runnerName"); break
            if not name: continue
            ex = _best_back_price(rb)
            if ex is not None:
                place_prices[(eid, name)] = ex

    # 4) Build DataFrame rows
    rows = []
    for (eid, name), vals in win_prices.items():
        ex = vals.get("ex"); sp = vals.get("sp")
        if ex is None and sp is None:
            continue
        start_iso = vals.get("start", "")
        rows.append({
            "EventId": eid,
            "Country": vals.get("country", "UK"),
            "Course": vals.get("venue", "Unknown"),
            "Time": _fmt_time(start_iso, display_tz),
            "MinsToOff": _mins_to_off(start_iso),
            "Horse": name,
            "Odds_Win_EX": ex,
            "Odds_Win_SP": sp,
            "Odds_Place_EX": place_prices.get((eid, name))  # may be None
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 5) Predicted win prob using EX↔SP blend (weight leaning to SP near the off)
    def blend_weight(mins_to_off: float) -> float:
        if mins_to_off <= 0:
            return 0.85
        if mins_to_off <= 10:
            return 0.70
        if mins_to_off <= 60:
            return 0.50
        return 0.20

    df["w_sp"] = df["MinsToOff"].apply(blend_weight)

    # Group-normalized EX and SP implied probabilities per event
    df["p_ex_norm"] = df.groupby("EventId", group_keys=False)["Odds_Win_EX"].apply(lambda s: _normalize_probs(s))
    df["p_sp_norm"] = df.groupby("EventId", group_keys=False)["Odds_Win_SP"].apply(lambda s: _normalize_probs(s))
    df["p_sp_norm"] = df["p_sp_norm"].fillna(df["p_ex_norm"])

    df["Pred_Win_Prob"] = (df["w_sp"] * df["p_sp_norm"] + (1.0 - df["w_sp"]) * df["p_ex_norm"]).clip(0, 1)
    df["Mkt_Win_Prob"]  = df.groupby("EventId", group_keys=False)["Odds_Win_EX"].apply(lambda s: _normalize_probs(s))

    # 6) Predicted place prob
    def place_prob_group(g: pd.DataFrame) -> pd.Series:
        prices = g["Odds_Place_EX"]
        if prices.notna().sum() == 0:
            n = len(g)
            if n <= 4:
                k = 2
            elif n <= 7:
                k = 3
            elif n <= 15:
                k = 4
            else:
                k = 5
            approx = (g["Pred_Win_Prob"] * k * 1.2).clip(0, 0.98)
            return approx
        # Normalize the place market implied probabilities within the event
        return _normalize_probs(prices)
    df["Pred_Place_Prob"] = df.groupby("EventId", group_keys=False).apply(place_prob_group)

    # 7) Public columns
    df["Predicted Win %"]   = (df["Pred_Win_Prob"]   * 100.0).round(1)
    df["Predicted Place %"] = (df["Pred_Place_Prob"] * 100.0).round(1)
    df["Odds"] = df["Odds_Win_EX"].round(2)
    df["Odds (SP proj)"] = df["Odds_Win_SP"].round(2)

    # Risk labels (tweak thresholds as you like)
    df["BetEdge Win %"] = df["Predicted Win %"]
    df["BetEdge Place %"] = df["Predicted Place %"]
    df["Risk"] = np.where(df["BetEdge Win %"] >= 25, "✅",
                   np.where(df["BetEdge Win %"] >= 15, "⚠️", "❌"))
    df["Source"] = "Betfair (live EX + SP blend)"
    return df.sort_values(["Country", "Course", "Time", "BetEdge Win %"], ascending=[True, True, True, False]).reset_index(drop=True)

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
    if region.upper() in COUNTRY_NAME: return COUNTRY_NAME[region.upper()]
    return region.upper()

def build_live_df_racingapi(payload: dict) -> pd.DataFrame:
    """Fallback only. No odds in this API, so we approximate just to keep UI alive."""
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
    # Approx to keep charts functional (clearly labeled)
    np.random.seed(42)
    df["Odds"] = np.random.uniform(2, 10, len(df)).round(2)
    df["Predicted Win %"] = (1.0 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 1.8).clip(0, 98).round(1)
    df["BetEdge Win %"] = df["Predicted Win %"]
    df["BetEdge Place %"] = df["Predicted Place %"]
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅", np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))
    df["Source"] = "Racing API (names) + mock odds"
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ------------------ EDGE BRAIN (stub) ------------------
def edgebrain_score_stub(df: pd.DataFrame) -> pd.Series:
    if "Mkt_Win_Prob" in df:
        return (0.6 * df["Pred_Win_Prob"] + 0.4 * df["Mkt_Win_Prob"]).mul(100).round(1)
    return df["BetEdge Win %"]

# ------------------ SIDEBAR NAV ------------------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Debug"],
        icons=["house", "activity", "robot", "bug"],
        default_index=0
    )
    st.markdown("---")
    st.caption("Display timezone")
    tz_choice = st.selectbox("Timezone", list(TZ_OPTIONS.keys()), index=0)
    DISPLAY_TZ = TZ_OPTIONS[tz_choice]
    st.markdown("---")
    st.caption("Countries to include")
    country_opts = ["GB (UK)", "IE (Ireland)", "AU (Australia)", "US (USA)"]
    chosen = st.multiselect("Betfair countries", country_opts, default=country_opts[:2])  # default GB+IE
    def _parse_cc(opt):
        return opt.split(" ", 1)[0]
    SELECTED_COUNTRIES = tuple(_parse_cc(o) for o in chosen) or ("GB",)

# ------------------ OVERVIEW ------------------
if sel == "Overview":
    st.title("📊 EdgeBet – Live Racing (Betfair)")
    df_live = build_live_df_betfair("today", SELECTED_COUNTRIES, DISPLAY_TZ)
    if df_live.empty:
        st.warning("Betfair feed not available right now. Falling back to Racing API names.")
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df_live = build_live_df_racingapi(data)

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Countries", int(df_live["Country"].nunique()) if not df_live.empty else 0)
    with colB:
        st.metric("Courses (today)", int(df_live["Course"].nunique()) if not df_live.empty else 0)
    with colC:
        st.metric("Runners (today)", int(len(df_live)) if not df_live.empty else 0)
    with colD:
        st.metric("With PLACE market", int(df_live["Odds_Place_EX"].notna().sum()) if "Odds_Place_EX" in df_live else 0)

    st.caption(f"All times shown in {tz_choice}.")
    if not df_live.empty:
        st.subheader("Top 20 Predicted Win % (EX↔SP blend)")
        st.bar_chart(df_live.sort_values("BetEdge Win %", ascending=False)
                             .head(20).set_index("Horse")["BetEdge Win %"], use_container_width=True)

# ------------------ HORSE RACING ------------------
elif sel == "Horse Racing":
    st.title("🏇 Horse Racing – Live")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)

    df = build_live_df_betfair(day, SELECTED_COUNTRIES, DISPLAY_TZ)
    if df.empty:
        st.warning("Betfair feed not available. Using Racing API names as fallback.")
        ok, data, _, _ = fetch_racecards_basic(day)
        if ok:
            df = build_live_df_racingapi(data)
        else:
            st.error("No data available.")
            st.stop()

    st.caption(f"All times shown in {tz_choice}.")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist()) if "Country" in df else ["All"]
        country = st.selectbox("Country", countries, index=0)
    with col2:
        bookie = st.selectbox("Bookmaker", ["All", "Betfair Exchange"], index=1)
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
    edge_range = st.slider("🎯 Filter by Predicted Win %", min_v, max_v, (min_v, max_v))
    filt = filt[filt["BetEdge Win %"].between(*edge_range)]

    view = st.radio("View", ["📊 Charts", "📋 Tables"], horizontal=True)

    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        if view == "📊 Charts":
            st.subheader("Top 20 Predicted Win % (EX↔SP blend)")
            st.bar_chart(
                filt.sort_values("BetEdge Win %", ascending=False)
                    .head(20).set_index("Horse")["BetEdge Win %"],
                use_container_width=True
            )
            st.subheader("Top 20 Predicted Place %")
            st.bar_chart(
                filt.sort_values("BetEdge Place %", ascending=False)
                    .head(20).set_index("Horse")["BetEdge Place %"],
                use_container_width=True
            )
        else:
            cols_to_show = ["Country", "Course", "Time", "Horse", "Odds", "Odds (SP proj)",
                            "BetEdge Win %", "BetEdge Place %", "Risk", "Source"]
            if "Odds_Place_EX" in filt.columns:
                filt["Odds_Place_EX"] = filt["Odds_Place_EX"].round(2)
                cols_to_show.insert(5, "Odds_Place_EX")
            st.subheader("Full Runner Table")
            st.dataframe(
                filt[cols_to_show],
                use_container_width=True
            )

# ------------------ EDGE BRAIN ------------------
elif sel == "EdgeBrain":
    st.title("🧠 EdgeBrain – Live Scoring (stub)")
    df = build_live_df_betfair("today", SELECTED_COUNTRIES, DISPLAY_TZ)
    if df.empty:
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df = build_live_df_racingapi(data)
    if df.empty:
        st.warning("No data available.")
    else:
        df["EdgeBrain Win %"] = edgebrain_score_stub(df)
        st.caption(f"All times shown in {tz_choice}.")
        st.subheader("Top 20 EdgeBrain Win %")
        st.bar_chart(
            df.sort_values("EdgeBrain Win %", ascending=False)
              .head(20).set_index("Horse")["EdgeBrain Win %"],
            use_container_width=True
        )
        show_cols = ["Country", "Course", "Time", "Horse", "Odds", "Odds (SP proj)",
                     "EdgeBrain Win %", "Risk", "Source"]
        if "Odds_Place_EX" in df.columns:
            show_cols.insert(5, "Odds_Place_EX")
        st.dataframe(df[show_cols], use_container_width=True)

# ------------------ DEBUG ------------------
else:
    st.title("🐞 Debug")
    token = bf_get_session_token()
    st.write("Has Betfair token:", bool(token))
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    st.write("Selected countries:", SELECTED_COUNTRIES)
    cat_win = bf_list_markets(day, ("WIN",), SELECTED_COUNTRIES)
    st.write("WIN catalogue markets:", len(cat_win))
    if cat_win:
        st.json(cat_win[0])
    ids = [m["marketId"] for m in cat_win][:10]
    books = bf_list_market_books(ids)
    st.write("Books fetched (WIN):", len(books))
    if books:
        st.json(books[0])
    cat_place = bf_list_markets(day, ("PLACE",), SELECTED_COUNTRIES)
    st.write("PLACE catalogue markets:", len(cat_place))
    if cat_place:
        st.json(cat_place[0])

st.caption(f"Times in {tz_choice} • Last updated {datetime.now(tz=UTC_TZ).strftime('%Y-%m-%d %H:%M UTC')}")
