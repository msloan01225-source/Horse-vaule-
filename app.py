# app.py — EdgeBet (Betfair live odds + Racing API fallback)
# Real EX odds + PLACE odds + EX↔SP blend + multi-country + timezone selector
# + Slick UI theme (teal brand) + Dark/Light toggle + crash fix on "Up Next"

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
import plotly.express as px

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="EdgeBet – Live Racing", layout="wide")

# ========= THEME =========
def inject_theme(mode: str = "Dark"):
    """Inject CSS variables for Dark or Light theme, using a single teal brand color."""
    # Brand (teal)
    BRAND = "#06B6D4"       # teal 400
    BRAND_SOFT_D = "rgba(6,182,212,0.16)"
    BRAND_SOFT_L = "rgba(6,182,212,0.12)"
    ACCENT = "#0891B2"      # deeper teal
    SUCCESS = "#16a34a"
    WARN = "#f59e0b"
    DANGER = "#ef4444"

    if mode == "Light":
        BG = "#FFFFFF"
        CARD_BG = "#F7FAFC"
        TEXT = "#0F172A"
        TEXT_MUTED = "#4B5563"
        BRAND_SOFT = BRAND_SOFT_L
        # plotly defaults
        plot_bg = BG
        font_color = TEXT
        grid = "rgba(15,23,42,0.08)"
    else:  # Dark
        BG = "#0B0F14"
        CARD_BG = "#0F1621"
        TEXT = "#E8ECF1"
        TEXT_MUTED = "#9AA6B2"
        BRAND_SOFT = BRAND_SOFT_D
        plot_bg = BG
        font_color = TEXT
        grid = "rgba(255,255,255,0.06)"

    st.session_state["_theme_vars"] = dict(
        BRAND=BRAND, BRAND_SOFT=BRAND_SOFT, ACCENT=ACCENT, BG=BG, CARD_BG=CARD_BG,
        TEXT=TEXT, TEXT_MUTED=TEXT_MUTED, SUCCESS=SUCCESS, WARN=WARN, DANGER=DANGER,
        plot_bg=plot_bg, font_color=font_color, grid=grid, mode=mode
    )

    st.markdown(f"""
    <style>
    :root {{
      --brand: {BRAND};
      --brand-soft: {BRAND_SOFT};
      --accent: {ACCENT};
      --bg: {BG};
      --card: {CARD_BG};
      --text: {TEXT};
      --muted: {TEXT_MUTED};
      --success: {SUCCESS};
      --warn: {WARN};
      --danger: {DANGER};
    }}
    html, body, [data-testid="stAppViewContainer"] {{
      background: var(--bg) !important;
      color: var(--text);
    }}
    h1, h2, h3, h4, h5, h6 {{ color: var(--text); letter-spacing: .2px; }}
    h1::after {{
      content:""; display:block; width:64px; height:3px; margin-top:8px;
      background: linear-gradient(90deg, var(--brand), transparent); border-radius:2px;
    }}
    a, a:visited {{ color: var(--accent) !important; }}
    .block-container {{ padding-top: 1rem; }}

    .stButton>button, .stDownloadButton>button {{
      background: var(--brand) !important; color: white !important; border: 0 !important;
      border-radius: 10px !important; padding: .55rem .9rem !important;
      box-shadow: 0 8px 18px rgba(6,182,212,.28) !important;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
      filter: brightness(1.08); transform: translateY(-1px);
    }}

    .stRadio [role="radiogroup"] > label, .stSelectbox label, .stMultiSelect label, .stSlider label {{
      color: var(--muted) !important;
    }}

    .stSlider [data-baseweb="slider"] > div > div {{ background: var(--brand) !important; }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
      background: var(--brand) !important; border: 2px solid rgba(6,182,212,.5) !important;
    }}

    [data-testid="stMetricValue"] {{ color: var(--text) !important; }}
    [data-testid="stMetricLabel"] {{ color: var(--muted) !important; }}

    div.card {{
      background: linear-gradient(180deg, var(--card), rgba(255,255,255,0.02));
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 14px;
      padding: 16px 16px;
      box-shadow: 0 10px 28px rgba(0,0,0,0.12);
    }}

    div.kpi {{ display:flex; flex-direction:column; gap:8px; }}
    div.kpi .label {{ color: var(--muted); font-size:.9rem; }}
    div.kpi .value {{ color: var(--text); font-size:1.6rem; font-weight:700; }}

    .badge {{
      display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
      border:1px solid rgba(0,0,0,0.08); background: var(--brand-soft); color: var(--text);
    }}

    .risk-chip {{ padding:2px 8px;border-radius:8px;font-size:12px;font-weight:700;display:inline-block; }}
    .risk-ok  {{ background: rgba(22,163,74,0.15);  color:#86efac; border:1px solid rgba(22,163,74,0.25); }}
    .risk-mid {{ background: rgba(245,158,11,0.12); color:#facc15; border:1px solid rgba(245,158,11,0.25); }}
    .risk-low {{ background: rgba(239,68,68,0.12);  color:#fca5a5; border:1px solid rgba(239,68,68,0.25); }}

    [data-testid="stSidebar"] > div > div {{
      background: var(--card);
      border-right: 1px solid rgba(0,0,0,0.06);
    }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar FIRST so theme choice applies everywhere
with st.sidebar:
    theme_mode = st.radio("Theme", ["Dark", "Light"], index=0, horizontal=True)
inject_theme(theme_mode)

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
    now_utc = datetime.now(tz=UTC_TZ)
    if day == "today":
        start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    else:
        start = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    return start.isoformat().replace("+00:00","Z"), end.isoformat().replace("+00:00","Z")

def _fmt_time(iso_str: str, tz: ZoneInfo) -> str:
    try:
        dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt_utc.astimezone(tz).strftime("%H:%M")
    except Exception:
        return ""

def _mins_to_off(iso_str: str) -> float:
    try:
        off_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now_utc = datetime.now(tz=UTC_TZ)
        return (off_utc - now_utc).total_seconds() / 60.0
    except Exception:
        return 9999.0

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ------------------ BETFAIR CORE ------------------
def _clear_cached_bf_session():
    try:
        bf_get_session_token.clear()
    except Exception:
        pass

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
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    backoff = 0.8
    for attempt in range(retries + 1):
        try:
            r = requests.post(BF_API_URL, headers=bf_headers(token), data=json.dumps(payload), timeout=12)
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < retries:
                    time.sleep(backoff); backoff *= 1.6; continue
            r.raise_for_status()
            out = r.json()
            if "error" in out:
                code = out["error"].get("data", {}).get("APINGException", {}).get("errorCode")
                if code in ("INVALID_SESSION_INFORMATION", "NO_SESSION"):
                    if attempt < retries:
                        _clear_cached_bf_session()
                        token = bf_get_session_token() or ""
                        if not token: raise RuntimeError("Unable to refresh Betfair session.")
                        continue
                raise RuntimeError(f"Betfair API error: {out['error']}")
            return out["result"]
        except requests.RequestException:
            if attempt < retries:
                time.sleep(backoff); backoff *= 1.6; continue
            raise

@st.cache_data(ttl=60)
def bf_list_markets(day="today", market_types=("WIN",), countries=("GB",)):
    token = bf_get_session_token()
    if not token:
        return []
    fr, to = _iso_range_for_day(day)
    params = {
        "filter": {
            "eventTypeIds": ["7"],
            "marketCountries": list(countries),
            "marketTypeCodes": list(market_types),
            "marketStartTime": {"from": fr, "to": to}
        },
        "sort": "FIRST_TO_START",
        "maxResults": 800,
        "marketProjection": [
            "RUNNER_DESCRIPTION","MARKET_START_TIME","EVENT","RUNNER_METADATA","MARKET_DESCRIPTION"
        ]
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
                "priceData": ["EX_BEST_OFFERS","SP_PROJECTED","SP_TRADED"],
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
    sp = runner_book.get("sp") or {}
    price = sp.get("farPrice") or sp.get("nearPrice")
    try:
        return float(price) if price is not None else None
    except Exception:
        return None

def _normalize_probs(prices: pd.Series) -> pd.Series:
    prices = prices.replace(0, np.nan)
    raw = 1.0 / prices
    s = raw.sum(skipna=True)
    if s and np.isfinite(s) and s > 0:
        return (raw / s).fillna(0.0)
    return raw.fillna(0.0)

# ------------------ BUILD DATAFRAME ------------------
def build_live_df_betfair(day="today", sel_countries=("GB", "IE", "AU", "US"), display_tz=ZoneInfo("Europe/London")) -> pd.DataFrame:
    with st.spinner("Fetching Betfair WIN markets…"):
        cat_win = bf_list_markets(day, market_types=("WIN",), countries=sel_countries)
    if not cat_win:
        return pd.DataFrame()

    win_markets = {m["marketId"]: m for m in cat_win}
    with st.spinner("Fetching Betfair PLACE markets…"):
        cat_place = bf_list_markets(day, market_types=("PLACE",), countries=sel_countries)

    place_by_event = {}
    for m in cat_place or []:
        place_by_event.setdefault(m["event"]["id"], []).append(m["marketId"])

    with st.spinner("Fetching live prices…"):
        books_win   = bf_list_market_books(list(win_markets.keys()))
        books_place = bf_list_market_books([mid for mids in place_by_event.values() for mid in mids])

    win_prices = {}
    for mb in books_win or []:
        mid = mb.get("marketId")
        mcat = win_markets.get(mid)
        if not mcat: continue
        event = mcat.get("event", {})
        event_id = event.get("id")
        start_iso = mcat.get("marketStartTime","")
        venue = event.get("venue", "Unknown")
        ccode = event.get("countryCode") or "GB"
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

    place_prices = {}
    place_cat_by_mid = {m["marketId"]: m for m in cat_place or []}
    chosen_place_mids = []
    for eid, mids in place_by_event.items():
        chosen_place_mids.append(mids[0])

    for mb in books_place or []:
        if mb.get("marketId") not in chosen_place_mids: continue
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

    rows = []
    for (eid, name), vals in win_prices.items():
        ex = vals.get("ex"); sp = vals.get("sp")
        if ex is None and sp is None: continue
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
            "Odds_Place_EX": place_prices.get((eid, name))
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def blend_weight(mins_to_off: float) -> float:
        if mins_to_off <= 0: return 0.85
        if mins_to_off <= 10: return 0.70
        if mins_to_off <= 60: return 0.50
        return 0.20

    df["w_sp"] = df["MinsToOff"].apply(blend_weight)
    df["p_ex_norm"] = df.groupby("EventId", group_keys=False)["Odds_Win_EX"].apply(lambda s: _normalize_probs(s))
    df["p_sp_norm"] = df.groupby("EventId", group_keys=False)["Odds_Win_SP"].apply(lambda s: _normalize_probs(s))
    df["p_sp_norm"] = df["p_sp_norm"].fillna(df["p_ex_norm"])
    df["Pred_Win_Prob"] = (df["w_sp"] * df["p_sp_norm"] + (1.0 - df["w_sp"]) * df["p_ex_norm"]).clip(0,1)
    df["Mkt_Win_Prob"]  = df.groupby("EventId", group_keys=False)["Odds_Win_EX"].apply(lambda s: _normalize_probs(s))

    def place_prob_group(g: pd.DataFrame) -> pd.Series:
        prices = g["Odds_Place_EX"]
        if prices.notna().sum() == 0:
            n = len(g)
            k = 2 if n<=4 else 3 if n<=7 else 4 if n<=15 else 5
            return (g["Pred_Win_Prob"] * k * 1.2).clip(0,0.98)
        return _normalize_probs(prices)
    df["Pred_Place_Prob"] = df.groupby("EventId", group_keys=False).apply(place_prob_group)

    df["Predicted Win %"]   = (df["Pred_Win_Prob"]   * 100.0).round(1)
    df["Predicted Place %"] = (df["Pred_Place_Prob"] * 100.0).round(1)
    df["Odds"] = df["Odds_Win_EX"].round(2)
    df["Odds (SP proj)"] = df["Odds_Win_SP"].round(2)
    df["BetEdge Win %"] = df["Predicted Win %"]
    df["BetEdge Place %"] = df["Predicted Place %"]
    df["Risk"] = np.where(df["BetEdge Win %"] >= 25, "✅",
                   np.where(df["BetEdge Win %"] >= 15, "⚠️", "❌"))
    df["Source"] = "Betfair (live EX + SP blend)"
    return df.sort_values(["Country","Course","Time","BetEdge Win %"], ascending=[True,True,True,False]).reset_index(drop=True)

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
    if region.upper() in COUNTRY_NAME: return COUNTRY_NAME[region.upper()]
    return region.upper()

def build_live_df_racingapi(payload: dict) -> pd.DataFrame:
    if not payload or "racecards" not in payload:
        return pd.DataFrame()
    rows = []
    for race in payload.get("racecards", []):
        region = race.get("region"); course = race.get("course"); off = race.get("off_time")
        for rnr in race.get("runners", []):
            horse = rnr.get("horse")
            if not horse: continue
            rows.append({"Country": map_region_to_country(region), "Course": course, "Time": off, "Horse": horse})
    df = pd.DataFrame(rows)
    if df.empty: return df
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

# -------------- UTIL: components for UI ---------------
def _risk_chip(risk:str)->str:
    if risk == "✅": return '<span class="risk-chip risk-ok">Low</span>'
    if risk == "⚠️": return '<span class="risk-chip risk-mid">Medium</span>'
    return '<span class="risk-chip risk-low">High</span>'

def _plot_top_bar(series: pd.Series, title: str):
    tv = st.session_state["_theme_vars"]
    fig = px.bar(
        series.sort_values(ascending=False).head(20),
        x=series.sort_values(ascending=False).head(20).index,
        y=series.sort_values(ascending=False).head(20).values,
        color_discrete_sequence=[tv["BRAND"]],
        title=None
    )
    fig.update_layout(
        paper_bgcolor=tv["plot_bg"], plot_bgcolor=tv["plot_bg"], font_color=tv["font_color"],
        margin=dict(t=10,l=10,r=10,b=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=11), tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor=tv["grid"], zerolinecolor=tv["grid"]),
        bargap=0.25,
    )
    fig.update_traces(marker=dict(line=dict(color="rgba(0,0,0,0.10)", width=1), opacity=0.95))
    st.markdown(f'<div class="card" style="margin-top:8px;"><h4 style="margin:0 0 10px 0">{title}</h4>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

def _hero(title: str, subtitle: str, tz_label: str):
    st.markdown(f"""
    <div class="card" style="padding:22px 22px; margin-bottom: 10px; background:
      radial-gradient(1200px 400px at -10% -50%, var(--brand-soft), transparent 60%),
      linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    ">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
        <div>
          <div class="badge">Live Racing</div>
          <h1 style="margin:4px 0 4px 0;">{title}</h1>
          <div style="color:var(--muted)">{subtitle}</div>
        </div>
        <div style="text-align:right;">
          <div class="badge">Times in {tz_label}</div>
          <div style="color:var(--muted); font-size:12px;">Updated {datetime.now(tz=UTC_TZ).strftime('%Y-%m-%d %H:%M UTC')}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def _kpi(label, value, delta=None):
    delta_html = f"<div style='color:var(--muted);font-size:12px'>{delta}</div>" if delta else ""
    st.markdown(f"""
    <div class="card kpi">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)

# ------------------ SIDEBAR NAV ------------------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Debug"],
        icons=["house", "activity", "robot", "bug"],
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "nav": {"background-color": st.session_state["_theme_vars"]["CARD_BG"] if "_theme_vars" in st.session_state else "#0F1621"},
            "icon": {"color": st.session_state["_theme_vars"]["BRAND"] if "_theme_vars" in st.session_state else "#06B6D4", "font-size": "18px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "var(--brand-soft)"},
            "nav-link-selected": {"background-color": "var(--brand-soft)", "color": "var(--text)"},
        }
    )
    st.markdown("---")
    st.caption("Display timezone")
    tz_choice = st.selectbox("Timezone", list(TZ_OPTIONS.keys()), index=0)
    DISPLAY_TZ = TZ_OPTIONS[tz_choice]
    st.markdown("---")
    st.caption("Countries to include")
    country_opts = ["GB (UK)", "IE (Ireland)", "AU (Australia)", "US (USA)"]
    chosen = st.multiselect("Betfair countries", country_opts, default=country_opts[:2])
    def _parse_cc(opt): return opt.split(" ", 1)[0]
    SELECTED_COUNTRIES = tuple(_parse_cc(o) for o in chosen) or ("GB",)

# ------------------ OVERVIEW ------------------
if sel == "Overview":
    _hero("EdgeBet – Live Racing", "Real-time exchange odds + place markets with a smart EX↔SP blend.", tz_choice)

    df_live = build_live_df_betfair("today", SELECTED_COUNTRIES, DISPLAY_TZ)
    if df_live.empty:
        st.warning("Betfair feed not available right now. Falling back to Racing API names.")
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok: df_live = build_live_df_racingapi(data)

    c1, c2, c3, c4 = st.columns(4)
    with c1: _kpi("Countries Today", int(df_live["Country"].nunique()) if not df_live.empty and "Country" in df_live else 0)
    with c2: _kpi("Courses Today", int(df_live["Course"].nunique()) if not df_live.empty and "Course" in df_live else 0)
    with c3: _kpi("Runners Today", int(len(df_live)) if not df_live.empty else 0)
    with c4: _kpi("With PLACE Market", int(df_live["Odds_Place_EX"].notna().sum()) if not df_live.empty and "Odds_Place_EX" in df_live else 0)

    # ---- Up Next strip (CRASH FIX: only if MinsToOff exists) ----
    if not df_live.empty and "MinsToOff" in df_live.columns:
        st.markdown('<div class="card"><h4 style="margin:0 0 8px 0">Up Next</h4>', unsafe_allow_html=True)
        upcoming = df_live.copy()
        upcoming = upcoming[upcoming["MinsToOff"] > -5].sort_values("MinsToOff").head(5)
        if not upcoming.empty:
            cols = st.columns(len(upcoming))
            for i, (_, row) in enumerate(upcoming.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="padding:8px;border-radius:12px;background:rgba(0,0,0,0.03);border:1px solid rgba(0,0,0,0.06)">
                      <div style="color:var(--muted); font-size:12px;">{row.get('Country','')} • {row.get('Course','')}</div>
                      <div style="font-weight:700; font-size:18px; color:var(--text)">{row.get('Time','')}</div>
                      <div style="color:var(--muted); font-size:12px;">{row.get('Horse','')}</div>
                      <div style="margin-top:6px;">
                        <span class="badge">Odds {row['Odds'] if pd.notna(row.get('Odds')) else '-'}</span>
                        <span class="badge">Win {row['BetEdge Win %']}%</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Top charts
    if not df_live.empty and "BetEdge Win %" in df_live and "BetEdge Place %" in df_live:
        _plot_top_bar(df_live.set_index("Horse")["BetEdge Win %"], "Top 20 Predicted Win %")
        _plot_top_bar(df_live.set_index("Horse")["BetEdge Place %"], "Top 20 Predicted Place %")

# ------------------ HORSE RACING ------------------
elif sel == "Horse Racing":
    st.markdown('<div class="card"><h1 style="margin:0">Horse Racing – Live</h1></div>', unsafe_allow_html=True)
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)

    df = build_live_df_betfair(day, SELECTED_COUNTRIES, DISPLAY_TZ)
    if df.empty:
        st.warning("Betfair feed not available. Using Racing API names as fallback.")
        ok, data, _, _ = fetch_racecards_basic(day)
        if ok: df = build_live_df_racingapi(data)
        else:
            st.error("No data available."); st.stop()

    st.caption(f"All times shown in {tz_choice}.")

    # Filters
    with st.container():
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
    if country != "All": filt = filt[filt["Country"] == country]
    if course_filter: filt = filt[filt["Course"].isin(course_filter)]

    # Range slider + view toggle
    min_v = int(filt["BetEdge Win %"].min()) if not filt.empty else 0
    max_v = int(filt["BetEdge Win %"].max()) if not filt.empty else 0
    if min_v > max_v: min_v, max_v = max_v, min_v
    edge_range = st.slider("🎯 Filter by Predicted Win %", min_v, max_v, (min_v, max_v))
    filt = filt[filt["BetEdge Win %"].between(*edge_range)]

    view = st.radio("View", ["📊 Charts", "📋 Tables"], horizontal=True)

    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        if view == "📊 Charts":
            _plot_top_bar(
                filt.sort_values("BetEdge Win %", ascending=False).set_index("Horse")["BetEdge Win %"],
                "Top 20 Predicted Win % (EX↔SP blend)"
            )
            _plot_top_bar(
                filt.sort_values("BetEdge Place %", ascending=False).set_index("Horse")["BetEdge Place %"],
                "Top 20 Predicted Place %"
            )
        else:
            show = filt[["Country","Course","Time","Horse","Odds","Odds (SP proj)","BetEdge Win %","BetEdge Place %","Risk","Source"]].copy()
            if "Odds_Place_EX" in filt.columns:
                show.insert(5, "Odds_Place_EX", filt["Odds_Place_EX"].round(2))
            show["Risk"] = show["Risk"].map(lambda r: _risk_chip(r), na_action='ignore')
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin:0 0 10px 0'>Full Runner Table</h4>", unsafe_allow_html=True)
            st.write(
                show.to_html(escape=False, index=False).replace("<table", "<table style='width:100%;border-collapse:separate;border-spacing:0 6px;color:var(--text)'").replace(
                    "<th", "<th style='text-align:left;color:var(--muted);padding:8px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'").replace(
                    "<td", "<td style='padding:6px 6px;border-bottom:1px solid rgba(0,0,0,0.03)'"),
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------ EDGE BRAIN ------------------
elif sel == "EdgeBrain":
    st.markdown('<div class="card"><h1 style="margin:0">EdgeBrain – Live Scoring</h1></div>', unsafe_allow_html=True)
    df = build_live_df_betfair("today", SELECTED_COUNTRIES, DISPLAY_TZ)
    if df.empty:
        ok, data, _, _ = fetch_racecards_basic("today")
        if ok: df = build_live_df_racingapi(data)
    if df.empty:
        st.warning("No data available.")
    else:
        df["EdgeBrain Win %"] = edgebrain_score_stub(df)
        st.caption(f"All times shown in {tz_choice}.")
        _plot_top_bar(
            df.sort_values("EdgeBrain Win %", ascending=False).set_index("Horse")["EdgeBrain Win %"],
            "Top 20 EdgeBrain Win %"
        )
        show = df[["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","Risk","Source"]].copy()
        if "Odds_Place_EX" in df.columns:
            show.insert(5, "Odds_Place_EX", df["Odds_Place_EX"].round(2))
        show["Risk"] = show["Risk"].map(lambda r: _risk_chip(r), na_action='ignore')
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0 0 10px 0'>EdgeBrain Table</h4>", unsafe_allow_html=True)
        st.write(
            show.to_html(escape=False, index=False).replace("<table", "<table style='width:100%;border-collapse:separate;border-spacing:0 6px;color:var(--text)'").replace(
                "<th", "<th style='text-align:left;color:var(--muted);padding:8px 6px;border-bottom:1px solid rgba(0,0,0,0.06)'").replace(
                "<td", "<td style='padding:6px 6px;border-bottom:1px solid rgba(0,0,0,0.03)'"),
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ DEBUG ------------------
else:
    st.markdown('<div class="card"><h1 style="margin:0">Debug</h1></div>', unsafe_allow_html=True)
    token = bf_get_session_token()
    st.write("Has Betfair token:", bool(token))
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    st.write("Selected countries:", SELECTED_COUNTRIES)
    cat_win = bf_list_markets(day, ("WIN",), SELECTED_COUNTRIES)
    st.write("WIN catalogue markets:", len(cat_win))
    if cat_win: st.json(cat_win[0])
    ids = [m["marketId"] for m in cat_win][:10]
    books = bf_list_market_books(ids)
    st.write("Books fetched (WIN):", len(books))
    if books: st.json(books[0])
    cat_place = bf_list_markets(day, ("PLACE",), SELECTED_COUNTRIES)
    st.write("PLACE catalogue markets:", len(cat_place))
    if cat_place: st.json(cat_place[0])

st.caption(f"Times in {tz_choice} • Last updated {datetime.now(tz=UTC_TZ).strftime('%Y-%m-%d %H:%M UTC')}")
