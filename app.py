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

# ------------------ CONFIG / THEME ------------------
st.set_page_config(page_title="EdgeBet – Live Racing", layout="wide")

# Theme mode state
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

# Toggle theme in sidebar
with st.sidebar:
    if st.button("🌗 Toggle Theme"):
        st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"

# CSS variables for both modes
def inject_theme():
    is_dark = st.session_state.theme_mode == "dark"
    teal = "#00bfa6" if is_dark else "#00796b"
    gold = "#ffd700"
    text_color = "#f2f2f2" if is_dark else "#111111"
    bg_color = "#111111" if is_dark else "#ffffff"
    zebra = "rgba(0,191,166,0.06)" if is_dark else "rgba(0,121,107,0.08)"
    hover = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.04)"
    border = "rgba(255,255,255,0.15)" if is_dark else "#ddd"

    st.markdown(f"""
    <style>
    :root {{
        --teal: {teal};
        --gold: {gold};
        --text-color: {text_color};
        --bg-color: {bg_color};
        --zebra: {zebra};
        --hover: {hover};
        --border: {border};
    }}
    body {{
        background-color: var(--bg-color);
        color: var(--text-color);
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: var(--teal);
    }}
    [data-testid="stMetricValue"] {{
        color: var(--teal) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_theme()

# ------------------ HTML TABLE RENDERER ------------------
def render_html_table(df, height: int = 600):
    """Render a DataFrame as fully controlled HTML so colors always show."""
    try:
        html_table = df.to_html(escape=False, index=False)
    except AttributeError:
        html_table = df.to_html(escape=False, index=False)

    html = f"""
    <div class="edge-table-wrap" style="max-height:{height}px; overflow:auto; border:1px solid var(--teal); border-radius:6px;">
      <style>
        .edge-table-wrap table {{
          width: 100%;
          border-collapse: collapse;
          color: var(--text-color);
          font-family: system-ui, sans-serif;
          font-size: 14px;
        }}
        .edge-table-wrap thead th {{
          position: sticky;
          top: 0;
          background: var(--teal) !important;
          color: #fff !important;
          z-index: 2;
        }}
        .edge-table-wrap tbody tr:nth-child(even) td {{
          background: var(--zebra) !important;
        }}
        .edge-table-wrap tbody tr:hover td {{
          background: var(--hover) !important;
        }}
        .edge-table-wrap td, .edge-table-wrap th {{
          padding: 8px 10px;
          border-bottom: 1px solid var(--border);
          text-align: left;
          vertical-align: middle;
        }}
        .gold-highlight td {{
          background-color: var(--gold) !important;
          color: #000000 !important;
        }}
        .teal-highlight td {{
          background-color: var(--teal) !important;
          color: #ffffff !important;
        }}
      </style>
      {html_table}
    </div>
    """
    components.html(html, height=height + 24, scrolling=True)

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
    except Exception:
        return ""

# ------------------ EDGE BRAIN ------------------
def edgebrain_score(df: pd.DataFrame) -> pd.Series:
    return (0.5 * df["BetEdge Win %"] + 0.5 * df["Predicted Win %"]).round(1)

def edgebrain_plus_score(df: pd.DataFrame) -> pd.Series:
    hist_factor = np.random.uniform(0.9, 1.1, len(df))
    return (df["EdgeBrain Win %"] * hist_factor).round(1)

def merge_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Jockey_History"] = np.random.randint(40, 70, len(df))
    df["Trainer_History"] = np.random.randint(50, 80, len(df))
    return df

def highlight_row(row):
    if pd.isna(row["EdgeBrain Win %"]) or pd.isna(row["EdgeBrain+ Win %"]):
        return ""
    if row["EdgeBrain+ Win %"] > row["EdgeBrain Win %"]:
        return 'class="teal-highlight"'
    elif row["EdgeBrain Win %"] > row["EdgeBrain+ Win %"]:
        return 'class="gold-highlight"'
    return ""

# ------------------ BETFAIR DATA ------------------
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
        mid   = m["marketId"]
        venue = m.get("event", {}).get("venue", "Unknown")
        tstr  = _hhmm(m.get("marketStartTime", ""))
        for r in m.get("runners", []):
            name = r.get("runnerName")
            sel  = r.get("selectionId")
            if mid in price_map:
                best_back, proj_sp = price_map[mid].get(sel, (None, None))
            else:
                best_back = proj_sp = None
            if not name or best_back is None:
                continue
            rows.append({
                "Country": "UK",
                "Course": venue,
                "Time": tstr,
                "Horse": name,
                "Odds": round(float(best_back), 2),
                "Odds (SP proj)": round(float(proj_sp), 2) if proj_sp else None
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

# ------ Racing API ------
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
        for idx, rnr in enumerate(runners):
            horse = rnr.get("horse")
            if not horse: continue
            # realistic odds distribution
            if idx == 0:
                best_back = np.random.uniform(2.2, 3.8)  # favourite
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
    np.random.seed(42)
    df["Win_Value"] = np.random.uniform(5, 30, len(df)).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (1.0 / df["Odds"] * 100).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)
    df["BetEdge Place %"] = ((df["Predicted Place %"] * 0.6) + (df["Place_Value"] * 0.4)).round(1)
    df["Risk"] = np.where(df["BetEdge Win %"] > 25, "✅", np.where(df["BetEdge Win %"] > 15, "⚠️", "❌"))
    df["Source"] = "Racing API (names) + mock odds"
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

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
        st.warning("Betfair feed not available right now. Falling back to Racing API.")
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df_live = build_live_df_racingapi(data)
    st.metric("Races (today)", df_live["Course"].nunique())
    st.metric("Runners (today)", len(df_live))
    if not df_live.empty:
        st.subheader("Top 20 BetEdge (today)")
        st.bar_chart(df_live.head(20).set_index("Horse")["BetEdge Win %"])

# ------------------ HORSE RACING ------------------
elif sel == "Horse Racing":
    st.title("🏇 Horse Racing – Live")
    st.markdown('<div style="position:sticky;top:0;background:var(--bg-color);padding:10px;z-index:100;border-bottom:1px solid var(--border);">', unsafe_allow_html=True)
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = build_live_df_betfair(day)
    if df.empty:
        st.warning("Betfair feed not available. Using Racing API fallback.")
        ok, data, _, _ = fetch_racecards_basic(day)
        if ok:
            df = build_live_df_racingapi(data)
    countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
    country = st.selectbox("Country", countries)
    bookie = st.selectbox("Bookmaker", ["All", "SkyBet 🟦", "Bet365 🟩", "Betfair 🟧"])
    courses = sorted(df["Course"].dropna().unique().tolist())
    course_filter = st.multiselect("Courses", courses, default=courses)
    st.markdown('</div>', unsafe_allow_html=True)

    filt = df.copy()
    if country != "All":
        filt = filt[filt["Country"] == country]
    if course_filter:
        filt = filt[filt["Course"].isin(course_filter)]
    if filt.empty:
        min_v = 0; max_v = 0
    else:
        min_v = int(filt["BetEdge Win %"].min()); max_v = int(filt["BetEdge Win %"].max())
    edge_range = st.slider("🎯 Filter by BetEdge Win %", min_v, max_v, (min_v, max_v))
    filt = filt[filt["BetEdge Win %"].between(*edge_range)]

    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        render_html_table(filt[["Country","Course","Time","Horse","Odds","Odds (SP proj)","BetEdge Win %","BetEdge Place %","Source"]])

# ------------------ EDGE BRAIN ------------------
elif sel == "EdgeBrain":
    st.title("🧠 EdgeBrain – Live Scoring")
    st.markdown('<div style="position:sticky;top:0;background:var(--bg-color);padding:10px;z-index:100;border-bottom:1px solid var(--border);">', unsafe_allow_html=True)
    view = st.radio("View", ["EdgeBrain", "EdgeBrain+", "Both"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)
    df = build_live_df_betfair("today")
    if df.empty:
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok:
            df = build_live_df_racingapi(data)
    if not df.empty:
        df["EdgeBrain Win %"] = edgebrain_score(df)
        df = merge_history_features(df)
        df["EdgeBrain+ Win %"] = edgebrain_plus_score(df)
        if view == "Both":
            rows_html = []
            for _, row in df.iterrows():
                cls = highlight_row(row)
                rows_html.append(f'<tr {cls}>' + ''.join(f'<td>{row[col]}</td>' for col in ["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","EdgeBrain+ Win %","Source"]) + '</tr>')
            table_html = "<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in ["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","EdgeBrain+ Win %","Source"]) + "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
            components.html(f'<div class="edge-table-wrap">{table_html}</div>', height=600, scrolling=True)
        elif view == "EdgeBrain":
            render_html_table(df[["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","Source"]])
        else:
            render_html_table(df[["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain+ Win %","Source"]])

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
