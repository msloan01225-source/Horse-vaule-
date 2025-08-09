# app.py — EdgeBet (Betfair live odds + Racing API fallback)
# Theme (teal) + Light/Dark + Logo + Sticky Filters + Plotly
# Betfair EX/PLACE odds + SP blend + multi-country + timezone
# EdgeBrain v1 (features + softmax)
# HistoryProvider scaffold (plug real endpoints later)

import os
import json
import math
import time
import base64
from pathlib import Path
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
st.set_page_config(page_title="BetEdge – Live Racing", layout="wide")

# ========= THEME (teal) + LIGHT/DARK =========
def inject_theme(mode: str = "Dark"):
    BRAND = "#06B6D4"       # teal 400
    BRAND_SOFT_D = "rgba(6,182,212,0.16)"
    BRAND_SOFT_L = "rgba(6,182,212,0.12)"
    ACCENT = "#0891B2"
    SUCCESS = "#16a34a"; WARN = "#f59e0b"; DANGER = "#ef4444"
    if mode == "Light":
        BG = "#FFFFFF"; CARD_BG = "#F7FAFC"; TEXT = "#0F172A"; TEXT_MUTED = "#4B5563"
        BRAND_SOFT = BRAND_SOFT_L; plot_bg = BG; font_color = TEXT; grid = "rgba(15,23,42,0.08)"
        shadow = "0 8px 20px rgba(2,6,23,0.08)"; border = "rgba(2,6,23,0.08)"; sticky_bg = BG
    else:
        BG = "#0B0F14"; CARD_BG = "#0F1621"; TEXT = "#E8ECF1"; TEXT_MUTED = "#9AA6B2"
        BRAND_SOFT = BRAND_SOFT_D; plot_bg = BG; font_color = TEXT; grid = "rgba(255,255,255,0.06)"
        shadow = "0 10px 28px rgba(0,0,0,0.35)"; border = "rgba(255,255,255,0.06)"; sticky_bg = CARD_BG

    st.session_state["_theme_vars"] = dict(
        BRAND=BRAND, BRAND_SOFT=BRAND_SOFT, ACCENT=ACCENT, BG=BG, CARD_BG=CARD_BG,
        TEXT=TEXT, TEXT_MUTED=TEXT_MUTED, SUCCESS=SUCCESS, WARN=WARN, DANGER=DANGER,
        plot_bg=plot_bg, font_color=font_color, grid=grid, mode=mode, shadow=shadow,
        border=border, sticky_bg=sticky_bg
    )
    st.markdown(f"""
    <style>
    :root {{
      --brand:{BRAND}; --brand-soft:{BRAND_SOFT}; --accent:{ACCENT};
      --bg:{BG}; --card:{CARD_BG}; --text:{TEXT}; --muted:{TEXT_MUTED};
      --success:{SUCCESS}; --warn:{WARN}; --danger:{DANGER};
      --shadow:{shadow}; --border:{border}; --sticky-bg:{sticky_bg};
    }}
    html, body, [data-testid="stAppViewContainer"] {{ background:var(--bg)!important;color:var(--text); }}
    h1,h2,h3,h4,h5,h6 {{ color:var(--text); letter-spacing:.2px; }}
    h1::after {{ content:""; display:block; width:64px; height:3px; margin-top:8px;
                 background:linear-gradient(90deg,var(--brand),transparent); border-radius:2px; }}
    .block-container {{ padding-top: 1rem; }}
    a, a:visited {{ color: var(--accent)!important; }}

    .stButton>button, .stDownloadButton>button {{
      background:var(--brand)!important; color:white!important; border:0!important;
      border-radius:10px!important; padding:.55rem .9rem!important;
      box-shadow:0 8px 18px rgba(6,182,212,.28)!important;
    }}
    .stButton>button:hover {{ filter:brightness(1.08); transform:translateY(-1px); }}

    .stRadio [role="radiogroup"]>label, .stSelectbox label, .stMultiSelect label, .stSlider label {{ color:var(--muted)!important; }}
    .stSlider [data-baseweb="slider"]>div>div {{ background:var(--brand)!important; }}
    .stSlider [role="slider"] {{ background:var(--brand)!important; border:2px solid rgba(6,182,212,.5)!important; }}

    [data-testid="stMetricValue"] {{ color:var(--text)!important; }}
    [data-testid="stMetricLabel"] {{ color:var(--muted)!important; }}

    div.card {{
      background:linear-gradient(180deg,var(--card),rgba(255,255,255,0.02));
      border:1px solid var(--border); border-radius:14px; padding:16px;
      box-shadow:var(--shadow);
    }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
              border:1px solid var(--border); background:var(--brand-soft); color:var(--text); }}
    .risk-chip {{ padding:2px 8px; border-radius:8px; font-size:12px; font-weight:700; display:inline-block; }}
    .risk-ok  {{ background:rgba(22,163,74,.15); color:#86efac; border:1px solid rgba(22,163,74,.25); }}
    .risk-mid {{ background:rgba(245,158,11,.12); color:#facc15; border:1px solid rgba(245,158,11,.25); }}
    .risk-low {{ background:rgba(239,68,68,.12); color:#fca5a5; border:1px solid rgba(239,68,68,.25); }}

    /* Sticky filters */
    #sticky-filters {{ position:sticky; top:0; z-index:60; background:var(--sticky-bg);
      border-radius:12px; box-shadow:0 10px 24px rgba(0,0,0,.18); border:1px solid var(--border);
      padding:10px 12px; margin-top:-8px; }}

    [data-testid="stSidebar"]>div>div {{ background:var(--card); border-right:1px solid var(--border); }}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    theme_mode = st.radio("Theme", ["Dark", "Light"], index=0, horizontal=True)
inject_theme(theme_mode)

# --------------- LOGO UTIL ---------------
def logo_img_tag(path: str, height: int = 52) -> str:
    p = Path(path)
    if not p.exists(): return ""
    if p.suffix.lower() == ".svg":
        try:
            svg = p.read_text(encoding="utf-8")
            return f"<span style='display:inline-flex;align-items:center;height:{height}px'>{svg}</span>"
        except Exception:
            return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    mime = "image/png" if p.suffix.lower()==".png" else "image/jpeg"
    return f"<img src='data:{mime};base64,{data}' style='height:{height}px;display:block'/>"

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
COUNTRY_NAME = {"GB":"UK","IE":"Ireland","AU":"Australia","US":"USA"}
BF_IDENTITY_URL = "https://identitysso.betfair.com/api/login"
BF_API_URL      = "https://api.betfair.com/exchange/betting/json-rpc/v1"

# ------------------ SECRETS ------------------
BF_APP_KEY = st.secrets.get("betfair", {}).get("app_key", os.getenv("BF_APP_KEY",""))
BF_SESSION  = st.secrets.get("betfair", {}).get("session_token", os.getenv("BF_SESSION",""))
BF_USER     = st.secrets.get("betfair", {}).get("username", os.getenv("BF_USER",""))
BF_PASS     = st.secrets.get("betfair", {}).get("password", os.getenv("BF_PASS",""))
RA_USER = st.secrets.get("racing_api", {}).get("username", os.getenv("RACING_API_USERNAME",""))
RA_PASS = st.secrets.get("racing_api", {}).get("password", os.getenv("RACING_API_PASSWORD",""))

# ------------------ TIME HELPERS ------------------
def _iso_range_for_day(day: str):
    now_utc = datetime.now(tz=UTC_TZ)
    if day == "today":
        start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0); end = start + timedelta(days=1)
    else:
        start = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0); end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00","Z"), end.isoformat().replace("+00:00","Z")

def _fmt_time(iso_str: str, tz: ZoneInfo) -> str:
    try:
        dt_utc = datetime.fromisoformat(iso_str.replace("Z","+00:00"))
        return dt_utc.astimezone(tz).strftime("%H:%M")
    except Exception:
        return ""

def _mins_to_off(iso_str: str) -> float:
    try:
        off_utc = datetime.fromisoformat(iso_str.replace("Z","+00:00"))
        now_utc = datetime.now(tz=UTC_TZ)
        return (off_utc - now_utc).total_seconds()/60.0
    except Exception:
        return 9999.0

def _chunks(lst, n):
    for i in range(0,len(lst),n): yield lst[i:i+n]

# ------------------ BETFAIR CORE ------------------
def _clear_cached_bf_session():
    try: bf_get_session_token.clear()
    except Exception: pass

@st.cache_resource
def bf_get_session_token():
    if not BF_APP_KEY:
        st.error("Betfair app_key missing in secrets."); return None
    if BF_SESSION: return BF_SESSION
    if BF_USER and BF_PASS:
        try:
            resp = requests.post(
                BF_IDENTITY_URL,
                headers={"X-Application":BF_APP_KEY,"Content-Type":"application/x-www-form-urlencoded","Accept":"application/json"},
                data={"username":BF_USER,"password":BF_PASS}, timeout=12
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status")=="SUCCESS": return data.get("token")
            st.error(f"Betfair login failed: {data}"); return None
        except Exception as e:
            st.error(f"Betfair login error: {e}"); return None
    st.warning("No Betfair session token and no username/password provided."); return None

def bf_headers(token:str):
    return {"X-Application":BF_APP_KEY,"X-Authentication":token,"Content-Type":"application/json","Accept":"application/json","Connection":"keep-alive"}

def bf_call(method:str, params:dict, token:str, *, retries:int=2):
    payload={"jsonrpc":"2.0","method":f"SportsAPING/v1.0/{method}","params":params,"id":1}
    backoff=0.8
    for attempt in range(retries+1):
        try:
            r=requests.post(BF_API_URL, headers=bf_headers(token), data=json.dumps(payload), timeout=12)
            if r.status_code in (429,500,502,503,504) and attempt<retries:
                time.sleep(backoff); backoff*=1.6; continue
            r.raise_for_status()
            out=r.json()
            if "error" in out:
                code=out["error"].get("data",{}).get("APINGException",{}).get("errorCode")
                if code in ("INVALID_SESSION_INFORMATION","NO_SESSION") and attempt<retries:
                    _clear_cached_bf_session(); token=bf_get_session_token() or ""; 
                    if not token: raise RuntimeError("Unable to refresh Betfair session.")
                    continue
                raise RuntimeError(f"Betfair API error: {out['error']}")
            return out["result"]
        except requests.RequestException:
            if attempt<retries: time.sleep(backoff); backoff*=1.6; continue
            raise

@st.cache_data(ttl=60)
def bf_list_markets(day="today", market_types=("WIN",), countries=("GB",)):
    token=bf_get_session_token()
    if not token: return []
    fr,to=_iso_range_for_day(day)
    params={"filter":{"eventTypeIds":["7"],"marketCountries":list(countries),"marketTypeCodes":list(market_types),"marketStartTime":{"from":fr,"to":to}},
            "sort":"FIRST_TO_START","maxResults":800,
            "marketProjection":["RUNNER_DESCRIPTION","MARKET_START_TIME","EVENT","RUNNER_METADATA","MARKET_DESCRIPTION"]}
    try: return bf_call("listMarketCatalogue", params, token)
    except Exception as e:
        st.error(f"Betfair listMarketCatalogue error: {e}"); return []

@st.cache_data(ttl=30)
def bf_list_market_books(market_ids):
    token=bf_get_session_token()
    if not token or not market_ids: return []
    books=[]
    for chunk in _chunks(market_ids,25):
        params={"marketIds":chunk,"priceProjection":{"priceData":["EX_BEST_OFFERS","SP_PROJECTED","SP_TRADED"],"virtualise":True}}
        try:
            res=bf_call("listMarketBook", params, token); books.extend(res)
        except Exception as e:
            st.error(f"Betfair listMarketBook error: {e}")
    return books

def _best_back_price(runner_book:dict):
    ex=runner_book.get("ex") or {}; atb=ex.get("availableToBack") or []
    price=atb[0].get("price") if atb else None
    if price is None:
        sp=runner_book.get("sp") or {}; price=sp.get("farPrice") or sp.get("nearPrice")
    try: return float(price) if price is not None and not (isinstance(price,float) and math.isnan(price)) else None
    except Exception: return None

def _sp_price(runner_book:dict):
    sp=runner_book.get("sp") or {}; price=sp.get("farPrice") or sp.get("nearPrice")
    try: return float(price) if price is not None else None
    except Exception: return None

def _normalize_probs(prices: pd.Series) -> pd.Series:
    prices=prices.replace(0,np.nan); raw=1.0/prices
    s=raw.sum(skipna=True)
    if s and np.isfinite(s) and s>0: return (raw/s).fillna(0.0)
    return raw.fillna(0.0)

# ------------------ BUILD LIVE DF (EX/PLACE + SP blend) ------------------
def build_live_df_betfair(day="today", sel_countries=("GB","IE","AU","US"), display_tz=ZoneInfo("Europe/London"))->pd.DataFrame:
    with st.spinner("Fetching Betfair WIN markets…"):
        cat_win=bf_list_markets(day, market_types=("WIN",), countries=sel_countries)
    if not cat_win: return pd.DataFrame()
    win_markets={m["marketId"]:m for m in cat_win}
    with st.spinner("Fetching Betfair PLACE markets…"):
        cat_place=bf_list_markets(day, market_types=("PLACE",), countries=sel_countries)
    place_by_event={}
    for m in cat_place or []:
        place_by_event.setdefault(m["event"]["id"],[]).append(m["marketId"])
    with st.spinner("Fetching live prices…"):
        books_win=bf_list_market_books(list(win_markets.keys()))
        books_place=bf_list_market_books([mid for mids in place_by_event.values() for mid in mids])

    win_prices={}
    for mb in books_win or []:
        mid=mb.get("marketId"); mcat=win_markets.get(mid)
        if not mcat: continue
        event=mcat.get("event",{}); eid=event.get("id")
        start_iso=mcat.get("marketStartTime",""); venue=event.get("venue","Unknown")
        ccode=event.get("countryCode") or "GB"; country=COUNTRY_NAME.get(ccode, ccode)
        for rb in (mb.get("runners") or []):
            name=None
            for rcat in (mcat.get("runners") or []):
                if rcat.get("selectionId")==rb.get("selectionId"): name=rcat.get("runnerName"); break
            if not name: continue
            ex=_best_back_price(rb); sp=_sp_price(rb)
            win_prices[(eid,name)]={"ex":ex,"sp":sp,"start":start_iso,"venue":venue,"country":country}

    place_prices={}
    place_cat_by_mid={m["marketId"]:m for m in cat_place or []}
    chosen=[mids[0] for mids in place_by_event.values()]
    for mb in books_place or []:
        if mb.get("marketId") not in chosen: continue
        mcat=place_cat_by_mid.get(mb.get("marketId")); 
        if not mcat: continue
        eid=mcat["event"]["id"]
        for rb in (mb.get("runners") or []):
            name=None
            for rcat in (mcat.get("runners") or []):
                if rcat.get("selectionId")==rb.get("selectionId"): name=rcat.get("runnerName"); break
            if not name: continue
            ex=_best_back_price(rb)
            if ex is not None: place_prices[(eid,name)]=ex

    rows=[]
    for (eid,name),vals in win_prices.items():
        ex=vals.get("ex"); sp=vals.get("sp")
        if ex is None and sp is None: continue
        iso=vals.get("start","")
        rows.append({
            "EventId":eid,"Country":vals.get("country","UK"),"Course":vals.get("venue","Unknown"),
            "Time":_fmt_time(iso, display_tz),"MinsToOff":_mins_to_off(iso),"Horse":name,
            "Odds_Win_EX":ex,"Odds_Win_SP":sp,"Odds_Place_EX":place_prices.get((eid,name))
        })
    df=pd.DataFrame(rows)
    if df.empty: return df

    def w_sp(mins):
        if mins<=0: return 0.85
        if mins<=10: return 0.70
        if mins<=60: return 0.50
        return 0.20
    df["w_sp"]=df["MinsToOff"].apply(w_sp)
    df["p_ex_norm"]=df.groupby("EventId", group_keys=False)["Odds_Win_EX"].apply(lambda s:_normalize_probs(s))
    df["p_sp_norm"]=df.groupby("EventId", group_keys=False)["Odds_Win_SP"].apply(lambda s:_normalize_probs(s)).fillna(df["p_ex_norm"])
    df["Pred_Win_Prob"]=(df["w_sp"]*df["p_sp_norm"]+(1.0-df["w_sp"])*df["p_ex_norm"]).clip(0,1)
    df["Mkt_Win_Prob"]=df.groupby("EventId", group_keys=False)["Odds_Win_EX"].apply(lambda s:_normalize_probs(s))

    def place_prob_group(g:pd.DataFrame)->pd.Series:
        prices=g["Odds_Place_EX"]
        if prices.notna().sum()==0:
            n=len(g); k=2 if n<=4 else 3 if n<=7 else 4 if n<=15 else 5
            return (g["Pred_Win_Prob"]*k*1.2).clip(0,0.98)
        return _normalize_probs(prices)
    df["Pred_Place_Prob"]=df.groupby("EventId", group_keys=False).apply(place_prob_group)

    df["Predicted Win %"]=(df["Pred_Win_Prob"]*100).round(1)
    df["Predicted Place %"]=(df["Pred_Place_Prob"]*100).round(1)
    df["Odds"]=df["Odds_Win_EX"].round(2); df["Odds (SP proj)"]=df["Odds_Win_SP"].round(2)
    df["BetEdge Win %"]=df["Predicted Win %"]; df["BetEdge Place %"]=df["Predicted Place %"]
    df["Risk"]=np.where(df["BetEdge Win %"]>=25,"✅",np.where(df["BetEdge Win %"]>=15,"⚠️","❌"))
    df["Source"]="Betfair (live EX + SP blend)"
    return df.sort_values(["Country","Course","Time","BetEdge Win %"], ascending=[True,True,True,False]).reset_index(drop=True)

# ------ Racing API (fallback) ------
@st.cache_data(ttl=120)
def fetch_racecards_basic(day="today"):
    user=RA_USER; pwd=RA_PASS
    if not user or not pwd: return False,None,None,"No Racing API credentials in secrets."
    url="https://api.theracingapi.com/v1/racecards"; params={"day":day}
    try:
        resp=requests.get(url, auth=HTTPBasicAuth(user,pwd), params=params, timeout=12)
        status=resp.status_code; text=resp.text; resp.raise_for_status()
        return True, resp.json(), status, text
    except requests.HTTPError as e:
        return False, None, e.response.status_code if e.response else None, e.response.text if e.response else str(e)
    except Exception as e:
        return False, None, None, str(e)

def map_region_to_country(region:str)->str:
    if not isinstance(region,str): return "Unknown"
    return COUNTRY_NAME.get(region.upper(), region.upper())

def build_live_df_racingapi(payload:dict)->pd.DataFrame:
    if not payload or "racecards" not in payload: return pd.DataFrame()
    rows=[]
    for race in payload.get("racecards", []):
        region=race.get("region"); course=race.get("course"); off=race.get("off_time")
        for rnr in race.get("runners", []):
            horse=rnr.get("horse"); 
            if not horse: continue
            rows.append({"Country":map_region_to_country(region),"Course":course,"Time":off,"Horse":horse})
    df=pd.DataFrame(rows); if df.empty: return df
    np.random.seed(42)
    df["Odds"]=np.random.uniform(2,10,len(df)).round(2)
    df["Predicted Win %"]=(1.0/df["Odds"]*100).round(1)
    df["Predicted Place %"]=(df["Predicted Win %"]*1.8).clip(0,98).round(1)
    df["BetEdge Win %"]=df["Predicted Win %"]; df["BetEdge Place %"]=df["Predicted Place %"]
    df["Risk"]=np.where(df["BetEdge Win %"]>25,"✅",np.where(df["BetEdge Win %"]>15,"⚠️","❌"))
    df["Source"]="Racing API (names) + mock odds"
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ------------------ HistoryProvider (scaffold) ------------------
class HistoryProvider:
    """
    Plug your results & entities API here. All methods return pandas DataFrames.
    Fill the 'TODO' bits with your real API calls and column mappings.
    """
    def __init__(self, user=RA_USER, pwd=RA_PASS, base_url="https://api.theracingapi.com/v1"):
        self.user=user; self.pwd=pwd; self.base=base_url

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_horse_history(self, horse_name:str, n:int=10)->pd.DataFrame:
        # TODO: replace with real endpoint
        # Example columns expected: date, pos, runners, going, dist_furlongs, course, sp, jockey, trainer, days_since
        return pd.DataFrame()

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_entity_form(self, entity:str, name:str)->pd.DataFrame:
        # entity in {"jockey","trainer"}
        # TODO: replace with real endpoint; return last 30-90 days results with 'win' column
        return pd.DataFrame()

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_course_bias(self, course:str)->pd.DataFrame:
        # Return course-level aggregates (e.g., win rates by draw/distance/going)
        return pd.DataFrame()

HP = HistoryProvider()

# ------------------ EdgeBrain v1 (features + softmax) ------------------
def _safe_div(a,b,default=np.nan):
    try: return a/b if (b not in (0,None) and pd.notna(b)) else default
    except Exception: return default

def add_edgebrain_features(df: pd.DataFrame) -> pd.DataFrame:
    eb=df.copy()
    # Base odds features
    eb["odds_ex"]=eb["Odds"].astype(float)
    eb["odds_sp"]=eb.get("Odds (SP proj)", np.nan).astype(float)
    eb["imp_ex"]=eb.groupby("EventId")["Odds"].transform(lambda s:_normalize_probs(s))
    eb["imp_sp"]=eb.groupby("EventId")["Odds (SP proj)"].transform(lambda s:_normalize_probs(s))
    eb["imp_sp"]=eb["imp_sp"].fillna(eb["imp_ex"])
    # Place ratio
    eb["place_odds"]=eb.get("Odds_Place_EX", np.nan).astype(float)
    eb["pl_win_ratio"]=eb.apply(lambda r:_safe_div(r["place_odds"], r["odds_ex"]), axis=1)
    # Field size & ranks
    eb["field_size"]=eb.groupby("EventId")["Horse"].transform("count")
    eb["fav_rank"]=eb.groupby("EventId")["Odds"].rank(method="min", ascending=True)
    eb["prob_rank"]=eb.groupby("EventId")["imp_ex"].rank(method="min", ascending=False)
    # Gap to fav (odds)
    def _fav_gap(g): return g["Odds"] - g["Odds"].min()
    eb["gap_to_fav"]=eb.groupby("EventId").apply(_fav_gap).reset_index(level=0, drop=True)
    # Time to off
    eb["mins_to_off"]=np.clip(eb.get("MinsToOff", 30).fillna(30), 0, 120)
    # Blend EX/SP
    def _w(mins):
        if mins<=0: return 0.85
        if mins<=10: return 0.70
        if mins<=60: return 0.50
        return 0.20
    eb["w_sp"]=eb["mins_to_off"].apply(_w)
    eb["p_blend"]=eb["w_sp"]*eb["imp_sp"]+(1.0-eb["w_sp"])*eb["imp_ex"]

    # --- History features (populate when HP returns data) ---
    # Examples: horse_win_rate_365, jockey_sr_30d, trainer_sr_30d, layoff_days, last_pos, avg_sp
    eb["horse_win_rate_365"]=np.nan
    eb["jockey_sr_30d"]=np.nan
    eb["trainer_sr_30d"]=np.nan
    eb["layoff_days"]=np.nan

    # TODO (when endpoints exist): vectorized join by horse/jockey/trainer/course keys

    # Standardize race-relative
    def _z(g,col): v=g[col]; return (v - v.mean()) / (v.std(ddof=0)+1e-9)
    for col in ["pl_win_ratio","gap_to_fav","mins_to_off","p_blend"]:
        eb[f"z_{col}"]=eb.groupby("EventId", group_keys=False).apply(lambda g:_z(g,col))
    # history z (won't hurt if NaN)
    for col in ["horse_win_rate_365","jockey_sr_30d","trainer_sr_30d","layoff_days"]:
        if col in eb: eb[f"z_{col}"]=eb.groupby("EventId", group_keys=False).apply(lambda g:_z(g,col)) if eb[col].notna().any() else 0.0

    eb["is_fav"]=(eb["fav_rank"]==1).astype(int)
    return eb

def edgebrain_softmax(df_feat: pd.DataFrame, weights: dict|None=None)->pd.Series:
    # priors (tweak via UI)
    w={"bias":-0.10,"blend":2.00,"fav":0.60,"rank":-0.10,"gap":-0.35,"ratio":-0.30,"field":-0.05,"mins":0.10,
       "horse365":0.40,"jock30":0.30,"train30":0.25,"layoff":-0.10}
    if isinstance(weights,dict): w.update(weights)
    f=df_feat.copy()
    f["z_p_blend"]=f["z_p_blend"] if "z_p_blend" in f else f["p_blend"]  # safety

    lin=( w["bias"]
        + w["blend"]*f["z_p_blend"]
        + w["fav"]*f["is_fav"]
        + w["rank"]*(f["fav_rank"]-1.0)
        + w["gap"]*f["z_gap_to_fav"]
        + w["ratio"]*f["z_pl_win_ratio"].fillna(0.0)
        + w["field"]*(np.log1p(f["field_size"]) - np.log(8))
        + w["mins"]*(1.0 - (f["mins_to_off"]/120.0))
        + w["horse365"]*f.get("z_horse_win_rate_365", 0)
        + w["jock30"]*f.get("z_jockey_sr_30d", 0)
        + w["train30"]*f.get("z_trainer_sr_30d", 0)
        + w["layoff"]*f.get("z_layoff_days", 0)
        )
    def _soft(g):
        x=g["lin"].values; x=x-np.max(x); p=np.exp(x); p/=p.sum()
        return pd.Series(p, index=g.index)
    f["lin"]=lin
    probs=f.groupby("EventId", group_keys=False).apply(_soft).clip(0,1)
    return probs

def edgebrain_score(df: pd.DataFrame, weights: dict|None=None)->pd.Series:
    feats=add_edgebrain_features(df)
    p=edgebrain_softmax(feats, weights=weights)
    return (p*100).round(1)

# -------------- UI UTIL ---------------
def _risk_chip(risk:str)->str:
    return '<span class="risk-chip risk-ok">Low</span>' if risk=="✅" else ('<span class="risk-chip risk-mid">Medium</span>' if risk=="⚠️" else '<span class="risk-chip risk-low">High</span>')

def _plot_top_bar(series: pd.Series, title:str):
    tv=st.session_state["_theme_vars"]
    s=series.sort_values(ascending=False).head(20)
    fig=px.bar(x=s.index, y=s.values, color_discrete_sequence=[tv["BRAND"]])
    fig.update_layout(paper_bgcolor=tv["plot_bg"], plot_bgcolor=tv["plot_bg"], font_color=tv["font_color"],
                      margin=dict(t=10,l=10,r=10,b=10),
                      xaxis=dict(showgrid=False, tickfont=dict(size=11), tickangle=-30),
                      yaxis=dict(showgrid=True, gridcolor=tv["grid"], zerolinecolor=tv["grid"]), bargap=0.25)
    fig.update_traces(marker=dict(line=dict(color="rgba(0,0,0,0.10)", width=1), opacity=0.95))
    st.markdown(f'<div class="card" style="margin-top:8px;"><h4 style="margin:0 0 10px 0">{title}</h4>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

def _hero(title:str, subtitle:str, tz_label:str):
    st.markdown(f"""
    <div class="card" style="padding:22px; margin-bottom:10px; background:
         radial-gradient(1200px 400px at -10% -50%, var(--brand-soft), transparent 60%),
         linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
        <div style="display:flex;align-items:center;gap:14px;">
          <div>{logo_img_tag("assets/betedge_logo.png", 56)}</div>
          <div>
            <div class="badge">Live Racing</div>
            <h1 style="margin:4px 0 4px 0;">{title}</h1>
            <div style="color:var(--muted)">{subtitle}</div>
          </div>
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
    st.markdown(f"<div class='card kpi'><div class='label'>{label}</div><div class='value' style='font-size:1.6rem;font-weight:700'>{value}</div>{delta_html}</div>", unsafe_allow_html=True)

def _render_html_table(df: pd.DataFrame, *, compact: bool):
    pad="4px 6px" if compact else "8px 6px"; fsize="13px" if compact else "14.5px"; thpad="6px 6px" if compact else "8px 6px"
    html=(df.to_html(escape=False, index=False)
        .replace("<table", "<table style='width:100%;border-collapse:separate;border-spacing:0 6px;color:var(--text);font-size:"+fsize+"'")
        .replace("<th", f"<th style='text-align:left;color:var(--muted);padding:{thpad};border-bottom:1px solid var(--border)'")
        .replace("<td", f"<td style='padding:{pad};border-bottom:1px solid rgba(0,0,0,0.03)'"))
    st.write(html, unsafe_allow_html=True)

# ------------------ SIDEBAR NAV + brand row ------------------
with st.sidebar:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>{logo_img_tag('assets/betedge_logo.png',40)}<div style='font-weight:800;letter-spacing:.5px;color:var(--text);font-size:18px;'>BETEDGE</div></div>",
        unsafe_allow_html=True
    )
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview","Horse Racing","EdgeBrain","Debug"],
        icons=["house","activity","robot","bug"],
        default_index=0,
        styles={
            "container":{"padding":"0!important"},
            "nav":{"background-color": st.session_state["_theme_vars"]["CARD_BG"]},
            "icon":{"color":st.session_state["_theme_vars"]["BRAND"],"font-size":"18px"},
            "nav-link":{"font-size":"14px","text-align":"left","margin":"0px","--hover-color":"var(--brand-soft)"},
            "nav-link-selected":{"background-color":"var(--brand-soft)","color":"var(--text)"}
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
    def _parse_cc(opt): return opt.split(" ",1)[0]
    SELECTED_COUNTRIES = tuple(_parse_cc(o) for o in chosen) or ("GB",)

# ------------------ OVERVIEW ------------------
if sel=="Overview":
    _hero("EdgeBet – Live Racing","Real-time exchange odds + place markets with a smart EX↔SP blend.", tz_choice)
    df_live=build_live_df_betfair("today", SELECTED_COUNTRIES, DISPLAY_TZ)
    if df_live.empty:
        st.warning("Betfair feed not available right now. Falling back to Racing API names.")
        ok,data,_,_=fetch_racecards_basic("today")
        if ok: df_live=build_live_df_racingapi(data)

    c1,c2,c3,c4=st.columns(4)
    with c1: _kpi("Countries Today", int(df_live["Country"].nunique()) if not df_live.empty and "Country" in df_live else 0)
    with c2: _kpi("Courses Today", int(df_live["Course"].nunique()) if not df_live.empty and "Course" in df_live else 0)
    with c3: _kpi("Runners Today", int(len(df_live)) if not df_live.empty else 0)
    with c4: _kpi("With PLACE Market", int(df_live["Odds_Place_EX"].notna().sum()) if not df_live.empty and "Odds_Place_EX" in df_live else 0)

    if not df_live.empty and "MinsToOff" in df_live.columns:
        st.markdown('<div class="card"><h4 style="margin:0 0 8px 0">Up Next</h4>', unsafe_allow_html=True)
        upcoming=df_live.copy()
        upcoming=upcoming[upcoming["MinsToOff"]>-5].sort_values("MinsToOff").head(5)
        if not upcoming.empty:
            cols=st.columns(len(upcoming))
            for i,(_,row) in enumerate(upcoming.iterrows()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="padding:8px;border-radius:12px;background:rgba(0,0,0,0.03);border:1px solid var(--border)">
                      <div style="color:var(--muted); font-size:12px;">{row.get('Country','')} • {row.get('Course','')}</div>
                      <div style="font-weight:700; font-size:18px; color:var(--text)">{row.get('Time','')}</div>
                      <div style="color:var(--muted); font-size:12px;">{row.get('Horse','')}</div>
                      <div style="margin-top:6px;">
                        <span class="badge">Odds {row['Odds'] if pd.notna(row.get('Odds')) else '-'}</span>
                        <span class="badge">Win {row['BetEdge Win %']}%</span>
                      </div>
                    </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not df_live.empty and "BetEdge Win %" in df_live and "BetEdge Place %" in df_live:
        _plot_top_bar(df_live.set_index("Horse")["BetEdge Win %"], "Top 20 Predicted Win %")
        _plot_top_bar(df_live.set_index("Horse")["BetEdge Place %"], "Top 20 Predicted Place %")

# ------------------ HORSE RACING ------------------
elif sel=="Horse Racing":
    st.markdown('<div class="card"><h1 style="margin:0">Horse Racing – Live</h1></div>', unsafe_allow_html=True)
    day=st.radio("Day",["today","tomorrow"], horizontal=True)
    df=build_live_df_betfair(day, SELECTED_COUNTRIES, DISPLAY_TZ)
    if df.empty:
        st.warning("Betfair feed not available. Using Racing API names as fallback.")
        ok,data,_,_=fetch_racecards_basic(day)
        if ok: df=build_live_df_racingapi(data)
        else: st.error("No data available."); st.stop()
    st.caption(f"All times shown in {tz_choice}.")

    with st.expander("🔎 Filters", expanded=True):
        st.markdown("<div id='sticky-filters'></div>", unsafe_allow_html=True)
        col1,col2,col3,col4=st.columns([1,1,2,1.2])
        with col1:
            countries=["All"]+sorted(df["Country"].dropna().unique().tolist()) if "Country" in df else ["All"]
            country=st.selectbox("Country", countries, index=0, key="flt_country")
        with col2:
            bookie=st.selectbox("Bookmaker", ["All","Betfair Exchange"], index=1, key="flt_bookie")
        with col3:
            courses=sorted(df["Course"].dropna().unique().tolist()); 
            course_filter=st.multiselect("Courses", courses, default=courses, key="flt_courses")
        with col4:
            density=st.radio("Row density", ["Comfortable","Compact"], index=0, horizontal=True, key="flt_density")
        min_v=int(df["BetEdge Win %"].min()) if not df.empty else 0
        max_v=int(df["BetEdge Win %"].max()) if not df.empty else 0
        if min_v>max_v: min_v,max_v=max_v,min_v
        edge_range=st.slider("🎯 Predicted Win %", min_v, max_v, (min_v, max_v), key="flt_edge_range")

    # Apply filters
    filt=df.copy()
    if country!="All": filt=filt[filt["Country"]==country]
    if course_filter: filt=filt[filt["Course"].isin(course_filter)]
    filt=filt[filt["BetEdge Win %"].between(*edge_range)]

    view=st.radio("View", ["📊 Charts","📋 Tables"], horizontal=True)
    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        if view=="📊 Charts":
            _plot_top_bar(filt.sort_values("BetEdge Win %", ascending=False).set_index("Horse")["BetEdge Win %"], "Top 20 Predicted Win % (EX↔SP blend)")
            _plot_top_bar(filt.sort_values("BetEdge Place %", ascending=False).set_index("Horse")["BetEdge Place %"], "Top 20 Predicted Place %")
        else:
            show=filt[["Country","Course","Time","Horse","Odds","Odds (SP proj)","BetEdge Win %","BetEdge Place %","Risk","Source"]].copy()
            if "Odds_Place_EX" in filt.columns: show.insert(5,"Odds_Place_EX",filt["Odds_Place_EX"].round(2))
            show["Risk"]=show["Risk"].map(lambda r:_risk_chip(r), na_action='ignore')
            st.markdown('<div class="card"><h4 style="margin:0 0 10px 0">Full Runner Table</h4>', unsafe_allow_html=True)
            _render_html_table(show, compact=(density=="Compact"))
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------ EDGE BRAIN ------------------
elif sel=="EdgeBrain":
    st.markdown('<div class="card"><h1 style="margin:0">EdgeBrain – Predictions</h1></div>', unsafe_allow_html=True)
    with st.expander("🧪 Model settings (advanced)", expanded=False):
        colA,colB,colC,colD=st.columns(4)
        with colA:
            w_blend=st.slider("Weight: market blend", 0.0, 4.0, 2.0, 0.1)
            w_fav  =st.slider("Weight: favourite bonus", 0.0, 2.0, 0.6, 0.05)
        with colB:
            w_rank =st.slider("Weight: rank penalty", -0.5, 0.0, -0.10, 0.01)
            w_gap  =st.slider("Weight: gap-to-fav", -1.0, 0.0, -0.35, 0.01)
        with colC:
            w_ratio=st.slider("Weight: place/win ratio", -1.0, 1.0, -0.30, 0.05)
            w_field=st.slider("Weight: field size", -0.2, 0.2, -0.05, 0.01)
        with colD:
            w_mins =st.slider("Weight: near post boost", -0.2, 0.5, 0.10, 0.01)
            w_bias =st.slider("Bias", -1.0, 1.0, -0.10, 0.01)
        user_w={"blend":w_blend,"fav":w_fav,"rank":w_rank,"gap":w_gap,"ratio":w_ratio,"field":w_field,"mins":w_mins,"bias":w_bias}

    day=st.radio("Day",["today","tomorrow"], horizontal=True)
    df=build_live_df_betfair(day, SELECTED_COUNTRIES, DISPLAY_TZ)
    if df.empty:
        ok,data,_,_=fetch_racecards_basic(day)
        if ok: df=build_live_df_racingapi(data)
    if df.empty:
        st.warning("No data available.")
    else:
        df["EdgeBrain Win %"]=edgebrain_score(df, user_w)
        df["Delta vs Market (pp)"]=(df["EdgeBrain Win %"]-df["BetEdge Win %"]).round(1)
        st.caption(f"All times shown in {tz_choice}.")
        _plot_top_bar(df.sort_values("EdgeBrain Win %", ascending=False).set_index("Horse")["EdgeBrain Win %"], "Top 20 EdgeBrain Win %")
        density = st.radio("Row density", ["Comfortable","Compact"], index=0, horizontal=True, key="eb_density")
        show=df[["Country","Course","Time","Horse","Odds","Odds (SP proj)","EdgeBrain Win %","BetEdge Win %","Delta vs Market (pp)","Risk","Source"]].copy()
        if "Odds_Place_EX" in df.columns: show.insert(5,"Odds_Place_EX",df["Odds_Place_EX"].round(2))
        show["Risk"]=show["Risk"].map(lambda r:_risk_chip(r), na_action='ignore')
        st.markdown('<div class="card"><h4 style="margin:0 0 10px 0">Predictions Table</h4>', unsafe_allow_html=True)
        _render_html_table(show.sort_values(["Course","Time","EdgeBrain Win %"], ascending=[True,True,False]), compact=(density=="Compact"))
        st.markdown('</div>', unsafe_allow_html=True)
        st.info("EdgeBrain v1: market signals + structure. Hook up HistoryProvider methods to enrich with horse/jockey/trainer/course data.")

# ------------------ DEBUG ------------------
else:
    st.markdown('<div class="card"><h1 style="margin:0">Debug</h1></div>', unsafe_allow_html=True)
    token=bf_get_session_token(); st.write("Has Betfair token:", bool(token))
    day=st.radio("Day",["today","tomorrow"], horizontal=True)
    st.write("Selected countries:", SELECTED_COUNTRIES)
    cat_win=bf_list_markets(day, ("WIN",), SELECTED_COUNTRIES); st.write("WIN catalogue markets:", len(cat_win)); 
    if cat_win: st.json(cat_win[0])
    ids=[m["marketId"] for m in cat_win][:10]
    books=bf_list_market_books(ids); st.write("Books fetched (WIN):", len(books))
    if books: st.json(books[0])
    cat_place=bf_list_markets(day, ("PLACE",), SELECTED_COUNTRIES); st.write("PLACE catalogue markets:", len(cat_place))
    if cat_place: st.json(cat_place[0])

st.caption(f"Times in {tz_choice} • Last updated {datetime.now(tz=UTC_TZ).strftime('%Y-%m-%d %H:%M UTC')}")
