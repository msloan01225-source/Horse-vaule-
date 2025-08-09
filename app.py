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

# ------------------ CONFIG ------------------
st.set_page_config(page_title="EdgeBet – Live Racing", layout="wide")

# ------------------ THEME MODE ------------------
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"
mode = st.toggle("🌙 Dark / ☀️ Light", value=(st.session_state["theme_mode"] == "dark"))
st.session_state["theme_mode"] = "dark" if mode else "light"

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

def map_region_to_country(region: str) -> str:
    if not isinstance(region, str): return "Unknown"
    if region.upper() == "GB": return "UK"
    if region.upper() in ("US", "USA"): return "USA"
    if region.upper() == "IE": return "Ireland"
    return region.upper()

@st.cache_resource
def bf_get_session_token():
    if not BF_APP_KEY:
        st.error("Betfair app_key missing.")
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
        except Exception as e:
            st.error(f"Betfair login error: {e}")
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

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _best_back_and_sp(runner_book: dict):
    ex = runner_book.get("ex", {})
    atb = ex.get("availableToBack", [])
    best_back = atb[0].get("price") if atb else None
    sp = runner_book.get("sp", {})
    proj_sp = sp.get("farPrice") or sp.get("nearPrice")
    return best_back, proj_sp

def _hhmm(iso_str):
    try:
        return datetime.fromisoformat(iso_str.replace("Z","")).strftime("%H:%M")
    except:
        return ""

# ------------------ INLINE TABLE RENDERER ------------------
def render_html_table(df: pd.DataFrame, height: int = 600):
    is_dark = st.session_state.get("theme_mode", "dark") == "dark"
    teal  = "#00bfa6" if is_dark else "#00796b"
    gold  = "#ffd700"
    text  = "#f2f2f2" if is_dark else "#111111"
    bg    = "#111111" if is_dark else "#ffffff"
    zebra = "#093c37" if is_dark else "#e0f2f1"
    bord  = "#1a4d47" if is_dark else "#9adad3"

    def fmt(v): return "" if pd.isna(v) else f"{v}"

    th_style = f"position:sticky;top:0;background:{teal};color:#fff;padding:8px 10px;border-bottom:1px solid {bord};text-align:left;z-index:2;"
    wrap_style = f"max-height:{height}px;overflow:auto;border:1px solid {teal};border-radius:6px;background:{bg};"
    table_style = f"width:100%;border-collapse:collapse;color:{text};font-family:system-ui,sans-serif;font-size:14px;"
    td_base = f"padding:8px 10px;border-bottom:1px solid {bord};"

    def td_cell(val, col):
        if col == "Odds" and val != "" and isinstance(val, (int, float)):
            if val <= 4:  return f"<td style='{td_base}color:lime;font-weight:700;'>{fmt(val)}</td>"
            if val >= 15: return f"<td style='{td_base}color:#ff6b6b;font-weight:700;'>{fmt(val)}</td>"
        if col == "BetEdge Win %" and val != "" and isinstance(val, (int, float)) and val >= 25:
            return f"<td style='{td_base}color:{gold};font-weight:700;'>{fmt(val)}</td>"
        return f"<td style='{td_base}'>{fmt(val)}</td>"

    thead = "<tr>" + "".join([f"<th style='{th_style}'>{c}</th>" for c in df.columns]) + "</tr>"
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        row_bg = zebra if (i % 2 == 1) else "transparent"
        tds = "".join([td_cell(r[c], c) for c in df.columns])
        rows.append(f"<tr style='background:{row_bg};'>{tds}</tr>")

    html = f"<div style='{wrap_style}'><table style='{table_style}'><thead>{thead}</thead><tbody>{''.join(rows)}</tbody></table></div>"
    components.html(html, height=height + 24, scrolling=True)

# ------------------ DATA FETCHERS ------------------
@st.cache_data(ttl=60)
def bf_list_win_markets(day="today"):
    token = bf_get_session_token()
    if not token: return []
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
    if not token or not market_ids: return []
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

# ------------------ DATA BUILDERS (Betfair + Racing API) ------------------
def _value_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    np.random.seed(42)
    df = df.copy()
    df["Win_Value"] = np.random.uniform(5, 30, len(df)).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    # basic probability from price
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

    # marketId -> selectionId -> (best_back, proj_sp)
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
        mid   = m["marketId"]
        venue = m.get("event", {}).get("venue", "Unknown")
        tstr  = _hhmm(m.get("marketStartTime", ""))
        for r in m.get("runners", []):
            name = r.get("runnerName")
            sel  = r.get("selectionId")
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
        return pd.DataFrame()
    df["Source"] = "Betfair (live)"
    df = _value_columns(df)
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

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
        off    = race.get("off_time")
        runners = race.get("runners", [])
        # Make odds look realistic & fill SP
        for idx, rnr in enumerate(runners):
            horse = rnr.get("horse")
            if not horse:
                continue
            if idx == 0:
                best_back = np.random.uniform(2.2, 3.8)      # fav
            elif idx <= 2:
                best_back = np.random.uniform(4.0, 7.0)      # next tier
            elif idx <= 5:
                best_back = np.random.uniform(7.5, 15.0)     # mids
            else:
                best_back = np.random.uniform(16.0, 40.0)    # outsiders
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
    df = _value_columns(df)
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ------------------ EDGEBRAIN ------------------
def edgebrain_score(df: pd.DataFrame) -> pd.Series:
    # simple blend of model & market
    return (0.5 * df["BetEdge Win %"] + 0.5 * df["Predicted Win %"]).round(1)

def merge_history_features(df: pd.DataFrame) -> pd.DataFrame:
    # placeholder historical signals for now
    df = df.copy()
    np.random.seed(123)
    df["Jockey_History"] = np.random.randint(40, 70, len(df))
    df["Trainer_History"] = np.random.randint(50, 80, len(df))
    return df

def edgebrain_plus_score(df: pd.DataFrame) -> pd.Series:
    # history-adjusted EB score
    hist_boost = (0.002 * df["Jockey_History"] + 0.003 * df["Trainer_History"])  # ~0.2–0.5 scaling
    return (df["EdgeBrain Win %"] * (1.0 + hist_boost)).round(1)

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
    st.title("📊 EdgeBet – Live Racing")

    df_live = build_live_df_betfair("today")
    if df_live.empty:
        st.warning("Betfair feed not available. Falling back to Racing API.")
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df_live = build_live_df_racingapi(data)

    colm = st.columns(2)
    with colm[0]:
        st.metric("Races (today)", df_live["Course"].nunique() if not df_live.empty else 0)
    with colm[1]:
        st.metric("Runners (today)", len(df_live))

    if not df_live.empty:
        st.subheader("Top 20 BetEdge (today)")
        st.bar_chart(df_live.head(20).set_index("Horse")["BetEdge Win %"])

# ------------------ HORSE RACING ------------------
elif sel == "Horse Racing":
    st.title("🏇 Horse Racing – Live")

    # Sticky filter bar (no external CSS, just inline)
    st.markdown(
        '<div style="position:sticky;top:0;background-color:transparent;'
        'padding:10px 8px;z-index:1000;border-bottom:1px solid rgba(255,255,255,0.08);">',
        unsafe_allow_html=True
    )
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    countries_placeholder = st.empty()
    bookie_placeholder = st.empty()
    courses_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    df = build_live_df_betfair(day)
    if df.empty:
        st.warning("Betfair feed not available. Using Racing API fallback.")
        ok, data, _, _ = fetch_racecards_basic(day)
        if ok:
            df = build_live_df_racingapi(data)
        if df.empty:
            st.error("No data available."); st.stop()

    with countries_placeholder:
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
        country = st.selectbox("Country", countries)
    with bookie_placeholder:
        _bookie = st.selectbox("Bookmaker", ["All", "SkyBet 🟦", "Bet365 🟩", "Betfair 🟧"])  # visual only for now
    with courses_placeholder:
        courses = sorted(df["Course"].dropna().unique().tolist())
        course_filter = st.multiselect("Courses", courses, default=courses)

    filt = df.copy()
    if country != "All":
        filt = filt[filt["Country"] == country]
    if course_filter:
        filt = filt[filt["Course"].isin(course_filter)]

    min_v = int(filt["BetEdge Win %"].min()) if not filt.empty else 0
    max_v = int(filt["BetEdge Win %"].max()) if not filt.empty else 0
    edge_range = st.slider("🎯 Filter by BetEdge Win %", min_v, max_v, (min_v, max_v))
    filt = filt[filt["BetEdge Win %"].between(*edge_range)]

    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        # unified inline-colored table
        render_html_table(
            filt[["Country","Course","Time","Horse","Odds","Odds (SP proj)","BetEdge Win %","BetEdge Place %","Source"]],
            height=580
        )

# ------------------ EDGE BRAIN ------------------
elif sel == "EdgeBrain":
    st.title("🧠 EdgeBrain – Live Scoring")

    # Sticky model bar
    st.markdown(
        '<div style="position:sticky;top:0;background-color:transparent;'
        'padding:10px 8px;z-index:1000;border-bottom:1px solid rgba(255,255,255,0.08);">',
        unsafe_allow_html=True
    )
    model_view = st.radio("View", ["EdgeBrain", "EdgeBrain+", "Both"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    df = build_live_df_betfair("today")
    if df.empty:
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df = build_live_df_racingapi(data)

    if df.empty:
        st.warning("No data available.")
    else:
        df["EdgeBrain Win %"] = edgebrain_score(df)
        df = merge_history_features(df)
        df["EdgeBrain+ Win %"] = edgebrain_plus_score(df)

        if model_view == "Both":
            # Build fully inline-styled table so row highlights always appear
            is_dark = st.session_state.get("theme_mode", "dark") == "dark"
            teal  = "#00bfa6" if is_dark else "#00796b"
            gold  = "#ffd700"
            text  = "#f2f2f2" if is_dark else "#111111"
            bg    = "#111111" if is_dark else "#ffffff"
            zebra = "#093c37" if is_dark else "#e0f2f1"
            bord  = "#1a4d47" if is_dark else "#9adad3"

            cols = ["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","EdgeBrain+ Win %","Source"]
            th_style = f"position:sticky;top:0;background:{teal};color:#fff;padding:8px 10px;border-bottom:1px solid {bord};text-align:left;z-index:2;"
            wrap_style = f"max-height:600px;overflow:auto;border:1px solid {teal};border-radius:6px;background:{bg};"
            table_style = f"width:100%;border-collapse:collapse;color:{text};font-family:system-ui,sans-serif;font-size:14px;"
            td_base = f"padding:8px 10px;border-bottom:1px solid {bord};"

            def fmt(v): return "" if pd.isna(v) else f"{v}"
            def td_cell(val, col):
                if col == "Odds" and val != "" and isinstance(val, (int, float)):
                    if val <= 4:  return f"<td style='{td_base}color:lime;font-weight:700;'>{fmt(val)}</td>"
                    if val >= 15: return f"<td style='{td_base}color:#ff6b6b;font-weight:700;'>{fmt(val)}</td>"
                if col == "EdgeBrain Win %" and val != "" and isinstance(val, (int, float)) and val >= 25:
                    return f"<td style='{td_base}color:{gold};font-weight:700;'>{fmt(val)}</td>"
                return f"<td style='{td_base}'>{fmt(val)}</td>"

            thead = "<tr>" + "".join([f"<th style='{th_style}'>{c}</th>" for c in cols]) + "</tr>"
            body_rows = []
            for i, (_, r) in enumerate(df.iterrows()):
                if pd.notna(r["EdgeBrain+ Win %"]) and pd.notna(r["EdgeBrain Win %"]):
                    if r["EdgeBrain+ Win %"] > r["EdgeBrain Win %"]:
                        tr_bg = teal; tr_color = "#000000"
                    elif r["EdgeBrain Win %"] > r["EdgeBrain+ Win %"]:
                        tr_bg = gold; tr_color = "#000000"
                    else:
                        tr_bg = (zebra if (i % 2 == 1) else "transparent"); tr_color = text
                else:
                    tr_bg = (zebra if (i % 2 == 1) else "transparent"); tr_color = text

                tds = "".join([td_cell(r[c], c) for c in cols])
                body_rows.append(f"<tr style='background:{tr_bg};color:{tr_color};'>{tds}</tr>")

            html = f"<div style='{wrap_style}'><table style='{table_style}'><thead>{thead}</thead><tbody>{''.join(body_rows)}</tbody></table></div>"
            components.html(html, height=624, scrolling=True)

        elif model_view == "EdgeBrain":
            render_html_table(df[["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","Source"]], height=580)
        else:
            render_html_table(df[["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain+ Win %","Source"]], height=580)

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

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
