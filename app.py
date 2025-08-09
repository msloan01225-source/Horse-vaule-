# edgebet_app.py — EdgeBet Pro (All-in)
# -----------------------------------------------------------------------------
# Restores "Horse Racing" tab + keeps all upgrades:
# - EdgeBrain v2 (de‑vig, momentum, pressure, SP anchor, EV/Kelly/Stake)
# - Value Finder, Dutching Lab, Ladder Inspector, Each‑Way EV
# - GB/IE toggle, desktop/audio alerts
# - Altair charts with highest bars in green
# -----------------------------------------------------------------------------

# ------------------ IMPORTS ------------------
import os, json, time, hashlib, random, math
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta, timezone
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from zoneinfo import ZoneInfo
from contextlib import suppress
from collections import deque, defaultdict
import altair as alt  # for colored charts

with suppress(Exception):
    from streamlit_autorefresh import st_autorefresh

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="EdgeBet Pro – Live Racing", layout="wide", page_icon="🏇")

# ------------------ THEME TOGGLE ------------------
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

def toggle_theme():
    st.session_state["theme_mode"] = "light" if st.session_state["theme_mode"] == "dark" else "dark"

col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("🏇 EdgeBet Pro – Live Racing")
with col2:
    st.button("🌗 Theme", on_click=toggle_theme, use_container_width=True)

# ------------------ THEME CSS ------------------
def apply_theme():
    is_dark = st.session_state["theme_mode"] == "dark"
    teal = "#00bfa6"
    if is_dark:
        bg, sbg, text, card = "#0f2a26", "#09211d", "#f2f2f2", "#123932"
        axis = "#bfeee8"
    else:
        bg, sbg, text, card = "#ffffff", "#f0f0f0", "#111111", "#f7fffd"
        axis = "#33665f"

    st.markdown(f"""
    <style>
      .stApp {{ background-color:{bg}; color:{text}; }}
      section[data-testid="stSidebar"] {{ background-color:{sbg} !important; }}
      .stMetric, .stTable, .stDataFrame {{ background:{card} !important; border-radius:10px; }}
      h1, h2, h3, h4 {{ color:#00bfa6; text-shadow:0 0 8px rgba(0,191,166,0.35); }}
      [data-testid="stMetricValue"] {{ color:{teal} !important; }}
      ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
      ::-webkit-scrollbar-thumb {{ background-color: {teal}; border-radius: 4px; }}
      .badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-weight:600; font-size:12px; }}
      .badge-hot {{ background:#ffefe8; color:#bb2e00; border:1px solid #ff9b73; }}
      .badge-cold {{ background:#e8f3ff; color:#004a99; border:1px solid #8ec6ff; }}
      .badge-ev {{ background:#e8fff7; color:#00664f; border:1px solid #00bfa6; }}
      .badge-risk-hi {{ background:#ffe8ef; color:#9a0036; border:1px solid #ff7aa5; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# Enable altair theme
alt.themes.enable('none')

# ------------------ DESKTOP NOTIFICATIONS BRIDGE ------------------
components.html("""
<script>
  async function notify(title, body) {
    try {
      if (!('Notification' in window)) return;
      if (Notification.permission === 'granted') {
        new Notification(title, { body: body });
      } else if (Notification.permission !== 'denied') {
        let p = await Notification.requestPermission();
        if (p === 'granted') new Notification(title, { body: body });
      }
    } catch(e) {}
  }
  window.edgebetNotify = notify;
</script>
""", height=0)

# ------------------ SECRETS ------------------
BF_APP_KEY = st.secrets.get("betfair", {}).get("app_key", os.getenv("BF_APP_KEY", ""))
BF_SESSION  = st.secrets.get("betfair", {}).get("session_token", os.getenv("BF_SESSION", ""))
BF_USER     = st.secrets.get("betfair", {}).get("username", os.getenv("BF_USER", ""))
BF_PASS     = st.secrets.get("betfair", {}).get("password", os.getenv("BF_PASS", ""))

RA_USER = st.secrets.get("racing_api", {}).get("username", os.getenv("RACING_API_USERNAME", ""))
RA_PASS = st.secrets.get("racing_api", {}).get("password", os.getenv("RACING_API_PASSWORD", ""))

BF_IDENTITY_URL = "https://identitysso.betfair.com/api/login"
BF_API_URL      = "https://api.betfair.com/exchange/betting/json-rpc/v1"

# ------------------ SESSION STATE ------------------
if "price_hist" not in st.session_state:
    st.session_state.price_hist = defaultdict(lambda: deque(maxlen=60))
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 1000.0

# ------------------ HELPERS ------------------
def _iso_range_for_day(day: str) -> tuple[str, str]:
    now_ldn = datetime.now(ZoneInfo("Europe/London"))
    base = now_ldn.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ldn = base if day == "today" else base + timedelta(days=1)
    end_ldn = start_ldn + timedelta(days=1)
    start_utc, end_utc = start_ldn.astimezone(timezone.utc), end_ldn.astimezone(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return start_utc.strftime(fmt), end_utc.strftime(fmt)

def _to_hhmm(iso_or_hhmm: str) -> str:
    if not isinstance(iso_or_hhmm, str):
        return ""
    s = iso_or_hhmm.strip()
    if len(s) == 5 and s[2] == ":" and s[:2].isdigit() and s[3:].isdigit():
        return s
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(dt): return ""
    return dt.tz_convert("Europe/London").strftime("%H:%M")

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _stable_rand01(key: str) -> float:
    h = hashlib.sha256(key.encode()).hexdigest()
    rnd = random.Random(int(h[:8], 16))
    return rnd.random()

def _format_cols(df: pd.DataFrame) -> dict:
    fmt = {}
    for col in df.columns:
        if col.endswith("%"): fmt[col] = "{:.1f}%"
        if col in ("Odds","Fair Odds","SP (proj)","EV","EW EV"): fmt[col] = "{:.2f}"
        if col == "Kelly %": fmt[col] = "{:.1f}%"
        if col == "Stake £": fmt[col] = "£{:.2f}"
    return fmt

def _toolbar(df: pd.DataFrame, key_prefix: str = "tb"):
    c1, c2, c3, c4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with c1:
        refresh_sec = st.selectbox("Auto-refresh", ["Off", 5, 10, 20, 30, 60], index=0, key=f"{key_prefix}_rf")
        if isinstance(refresh_sec, int):
            with suppress(Exception):
                st_autorefresh(interval=refresh_sec * 1000, key=f"{key_prefix}_autorf_{refresh_sec}")
    with c2:
        st.download_button(
            "⬇️ CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"edgebet_{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c3:
        st.number_input("Bankroll £", min_value=0.0, step=50.0, value=float(st.session_state.bankroll),
                        key=f"{key_prefix}_bk", on_change=lambda: st.session_state.update(bankroll=st.session_state[f"{key_prefix}_bk"]))
    with c4:
        st.write(f"Rows: **{len(df):,}**")

# ------------------ BETFAIR API ------------------
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
                st.error("Betfair login failed.")
                return None
        except Exception as e:
            st.error(f"Betfair login error: {e}")
            return None
    st.warning("No Betfair session token and no username/password provided.")
    return None

def bf_headers(token):
    return {"X-Application": BF_APP_KEY, "X-Authentication": token, "Content-Type": "application/json"}

def _bf_post(method: str, params: dict, token: str):
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    r = requests.post(BF_API_URL, headers=bf_headers(token), data=json.dumps(payload), timeout=12)
    r.raise_for_status()
    return r.json()

def bf_call(method: str, params: dict, token: str):
    out = _bf_post(method, params, token)
    if "error" not in out:
        return out["result"]
    code = (out["error"].get("data") or {}).get("APINGException", {}).get("errorCode")
    if code in {"INVALID_SESSION_INFORMATION", "NO_APP_KEY"}:
        bf_get_session_token.clear()
        new_token = bf_get_session_token()
        if not new_token: raise RuntimeError(out["error"])
        out2 = _bf_post(method, params, new_token)
        if "error" in out2: raise RuntimeError(out2["error"])
        return out2["result"]
    raise RuntimeError(out["error"])

def map_region_to_country(region: str) -> str:
    if not isinstance(region, str): return "Unknown"
    r = region.upper()
    if r == "GB": return "UK"
    if r in ("US", "USA"): return "USA"
    if r == "IE": return "Ireland"
    return r

def _best_prices_sizes(rb: dict):
    ex = rb.get("ex", {}) or {}
    atb = ex.get("availableToBack", []) or []
    atl = ex.get("availableToLay", []) or []
    best_back = atb[0]["price"] if atb else None
    best_lay  = atl[0]["price"] if atl else None
    back_sz = sum([lvl.get("size", 0) for lvl in atb[:3]])
    lay_sz  = sum([lvl.get("size", 0) for lvl in atl[:3]])
    sp = (rb.get("sp", {}) or {}).get("farPrice") or (rb.get("sp", {}) or {}).get("nearPrice")
    return best_back, best_lay, back_sz, lay_sz, sp

# ------------------ DATA BUILDERS ------------------
@st.cache_data(ttl=60, show_spinner=False)
def bf_list_win_markets(day="today", countries=None):
    countries = countries or ["GB"]
    token = bf_get_session_token()
    if not token: return []
    fr, to = _iso_range_for_day(day)
    params = {
        "filter": {
            "eventTypeIds": ["7"],
            "marketCountries": countries,
            "marketTypeCodes": ["WIN"],
            "marketStartTime": {"from": fr, "to": to}
        },
        "maxResults": 200,
        "marketProjection": ["RUNNER_DESCRIPTION", "MARKET_START_TIME", "EVENT", "RUNNER_METADATA"]
    }
    try:
        return bf_call("listMarketCatalogue", params, token)
    except Exception:
        st.error("Betfair listMarketCatalogue error.")
        return []

@st.cache_data(ttl=30, show_spinner=False)
def bf_list_market_books(market_ids: list[str]):
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
        last_err = None
        for attempt in range(3):
            try:
                res = bf_call("listMarketBook", params, token)
                books.extend(res or [])
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.25 * (attempt + 1))
        if last_err:
            st.warning(f"Betfair price chunk failed ({len(chunk)} IDs). Continuing.")
        time.sleep(0.05)
    return books

def _update_hist(key: str, odds: float):
    if odds is None: return
    ts = int(time.time())
    dq = st.session_state.price_hist[key]
    if not dq or dq[-1][0] != ts:
        dq.append((ts, float(odds)))

def _sparkline_series(key: str, window: int = 30):
    dq = st.session_state.price_hist.get(key, deque(maxlen=60))
    if not dq: return []
    arr = list(dq)[-window:]
    return [x[1] for x in arr]

def _overround_and_fair_probs(df: pd.DataFrame):
    df = df.copy()
    df["RaceKey"] = df["Course"].astype(str) + " | " + df["Time"].astype(str)
    over = df.groupby("RaceKey")["Odds"].apply(lambda s: (1.0 / s.clip(lower=1.01)).sum()).rename("Overround")
    df = df.merge(over, on="RaceKey", how="left")
    df["Fair p"] = (1.0 / df["Odds"].clip(lower=1.01)) / df["Overround"].clip(lower=1e-6)
    df["Fair Odds"] = (1.0 / df["Fair p"]).round(2)
    return df

def build_live_df_betfair(day="today", countries=None) -> pd.DataFrame:
    cat = bf_list_win_markets(day, countries=countries)
    if not cat:
        return pd.DataFrame()
    ids = [m.get("marketId") for m in cat if m.get("marketId")]
    books = bf_list_market_books(ids) if ids else []

    price_map, status_map = {}, {}
    for mb in books:
        mid = mb.get("marketId")
        status_map[mid] = mb.get("status", "UNKNOWN")
        sel_map = {}
        for rb in (mb.get("runners") or []):
            bb, bl, bsz, lsz, sp = _best_prices_sizes(rb)
            sel_map[rb.get("selectionId")] = (bb, bl, bsz, lsz, sp)
        price_map[mid] = sel_map

    rows = []
    now_utc = datetime.utcnow()
    for m in cat:
        mid = m.get("marketId")
        venue = (m.get("event", {}) or {}).get("venue", "Unknown")
        tstr = _to_hhmm(m.get("marketStartTime", ""))
        mstatus = status_map.get(mid, "UNKNOWN")
        for r in (m.get("runners", []) or []):
            name = r.get("runnerName")
            sel = r.get("selectionId")
            bb, bl, bsz, lsz, sp = (price_map.get(mid, {}) or {}).get(sel, (None, None, 0.0, 0.0, None))
            if not name or bb is None:
                continue
            key = f"{mid}:{sel}"
            _update_hist(key, bb)
            rows.append({
                "MarketId": mid,
                "SelectionId": sel,
                "Country": "UK" if (countries or ["GB"]) == ["GB"] else "Mixed",
                "Course": venue,
                "Time": tstr,
                "Horse": name,
                "Odds": round(float(bb), 2),
                "Best Lay": round(float(bl), 2) if bl else None,
                "Back Size": float(bsz or 0.0),
                "Lay Size": float(lsz or 0.0),
                "SP (proj)": round(float(sp), 2) if sp else None,
                "Market Status": mstatus,
                "Last Seen (UTC)": now_utc.strftime("%H:%M:%S"),
                "Source": "Betfair (live)"
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _overround_and_fair_probs(df)
    # legacy engineered cols for continuity
    key = (df["Course"].astype(str) + "|" + df["Time"].astype(str) + "|" + df["Horse"].astype(str))
    base = key.apply(_stable_rand01)
    df["Win_Value"] = (5 + base * 25).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (100.0 / df["Odds"].clip(lower=1.01)).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    return df.reset_index(drop=True)

# --------------- RACING API (FALLBACK) ---------------
@st.cache_data(ttl=120, show_spinner=False)
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
            if not horse: continue
            if idx == 0: best_back = np.random.uniform(2.2, 3.8)
            elif idx <= 2: best_back = np.random.uniform(4.0, 7.0)
            elif idx <= 5: best_back = np.random.uniform(7.5, 15.0)
            else: best_back = np.random.uniform(16.0, 40.0)
            proj_sp = best_back * np.random.uniform(0.95, 1.05)
            key = f"MOCK:{course}:{off}:{horse}"
            _update_hist(key, best_back)
            rows.append({
                "MarketId": f"MOCK:{course}:{off}",
                "SelectionId": idx + 1,
                "Country": map_region_to_country(region),
                "Course": course,
                "Time": off,
                "Horse": horse,
                "Odds": round(float(best_back), 2),
                "Best Lay": round(float(best_back*1.02), 2),
                "Back Size": float(np.random.uniform(50, 1000)),
                "Lay Size": float(np.random.uniform(50, 1000)),
                "SP (proj)": round(float(proj_sp), 2),
                "Market Status": "OPEN",
                "Last Seen (UTC)": datetime.utcnow().strftime("%H:%M:%S"),
                "Source": "Racing API (names) + mock odds"
            })
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = _overround_and_fair_probs(df)
    key = (df["Course"].astype(str) + "|" + df["Time"].astype(str) + "|" + df["Horse"].astype(str))
    base = key.apply(_stable_rand01)
    df["Win_Value"] = (5 + base * 25).round(1)
    df["Place_Value"] = (df["Win_Value"] * 0.6).round(1)
    df["Predicted Win %"] = (100.0 / df["Odds"].clip(lower=1.01)).round(1)
    df["Predicted Place %"] = (df["Predicted Win %"] * 0.6).round(1)
    return df.reset_index(drop=True)

# ------------------ EDGEBRAIN v2 ------------------
def _momentum(series_odds: list[float]) -> float:
    if len(series_odds) < 3: return 0.0
    y = np.array(series_odds[-10:])
    x = np.arange(len(y))
    p = 1.0 / np.clip(y, 1.01, None)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, p, rcond=None)[0]
    return float(np.clip(slope * 100.0, -5.0, 5.0))

def _pressure(back_sz: float, lay_sz: float) -> float:
    total = max(back_sz + lay_sz, 1e-6)
    ratio = (back_sz - lay_sz) / total
    return float(np.clip(ratio * 10.0, -5.0, 5.0))

def edgebrain_enhance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()

    # Momentum from sparkline
    keys = (out["MarketId"].astype(str) + ":" + out["SelectionId"].astype(str)).tolist()
    out["Momentum"] = [ _momentum(_sparkline_series(k, 30)) for k in keys ]

    # Pressure
    out["Pressure"] = out.apply(lambda r: _pressure(r["Back Size"], r["Lay Size"]), axis=1)

    # SP anchor deviation
    out["SP Deviation %"] = np.where(out["SP (proj)"].notna(),
                                     (out["SP (proj)"] - out["Odds"]) / out["Odds"] * 100.0, 0.0).round(1)

    # Base (de‑vig) probability
    out["Fair p"] = out["Fair p"].clip(0.001, 0.999)
    out["Base p %"] = (out["Fair p"] * 100.0).round(2)

    # Value boost term
    value_boost = ((1.0/np.clip(out["Odds"],1.01,None)) - out["Fair p"]) * 100.0

    # Composite score
    w_base, w_mom, w_press, w_sp, w_value = 0.56, 0.16, 0.10, 0.08, 0.10
    out["EdgeBrain Score"] = (
        w_base * out["Base p %"] +
        w_mom  * np.clip(out["Momentum"] + 2.0, -5.0, 5.0) +
        w_press* out["Pressure"] +
        w_sp   * np.clip(-out["SP Deviation %"]/2.0, -5.0, 5.0) +
        w_value* value_boost
    ).clip(0, 100).round(1)

    # EV & Kelly
    p = out["Fair p"]
    q = 1 - p
    b = out["Odds"] - 1.0
    out["EV"] = (p * (out["Odds"]) - q).round(2)
    kelly = (b * p - q) / b
    bankroll = max(float(st.session_state.bankroll), 0.0)
    cap = bankroll * 0.02
    out["Kelly %"] = (np.clip(kelly, 0, 1.0) * 100.0).round(1)
    out["Stake £"] = np.minimum(cap, bankroll * np.clip(kelly, 0, 1.0)).round(2)

    # Risk & Confidence
    out["Risk"] = np.select(
        [out["EdgeBrain Score"] >= 35, out["EdgeBrain Score"] >= 22],
        ["✅", "⚠️"], default="❌"
    )
    conf_raw = (out["Momentum"].clip(-3,3) + out["Pressure"].clip(-3,3) + (value_boost.clip(-6,6))) / 3.0
    out["Confidence %"] = (50 + conf_raw * 8).clip(0, 100).round(0)

    # Badges
    out["Badges"] = out.apply(lambda r: "".join([
        f'<span class="badge badge-ev">EV {r["EV"]:+.2f}</span> ' if r["EV"] >= 0.05 else "",
        f'<span class="badge badge-hot">🔥</span> ' if r["Momentum"]>0.4 else "",
        f'<span class="badge badge-cold">🧊</span> ' if r["Momentum"]<-0.5 else "",
        f'<span class="badge badge-risk-hi">Hi Risk</span> ' if r["Risk"]=="❌" else ""
    ]), axis=1)

    return out

# ------------------ DUTCHING & EW ------------------
def dutch_stakes(odds: list[float], total_stake: float, method: str = "equal_return"):
    odds = np.array(odds, dtype=float)
    inv = 1.0 / (odds - 1.0)
    w = inv / inv.sum()
    return np.round(total_stake * w, 2)

def each_way_ev(odds, win_p, places=3, terms_frac=0.25):
    odds = float(odds)
    p_win = float(win_p)
    # heuristic place prob
    place_p = min(1.0, p_win * (1 + 0.9*(places-1)))
    win_leg_ev   = p_win * (odds - 1) - (1 - p_win) * 1
    place_odds   = 1 + terms_frac * (odds - 1)
    place_leg_ev = place_p * (place_odds - 1) - (1 - place_p) * 1
    return 0.5*win_leg_ev + 0.5*place_leg_ev  # per £1 EW (0.5+0.5)

# ------------------ SIDEBAR NAV ------------------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet Pro",
        ["Overview", "Horse Racing", "Value Finder", "Dutching Lab", "EdgeBrain", "Debug"],
        icons=["house", "activity", "fire", "intersect", "robot", "bug"],
        default_index=0
    )
    mock_mode = st.toggle("🧪 Mock mode (no APIs)", value=False, help="Generate plausible data if feeds are down.")
    market_scope = st.multiselect("🌍 Market countries", options=["GB", "IE"], default=["GB"])
    # Alerts
    st.markdown("---")
    st.markdown("### Alerts")
    alert_ev = st.slider("🔔 Alert when EV ≥", -0.50, 1.0, 0.10, 0.01)
    alert_score = st.slider("🔔 And EdgeBrain ≥", 0, 100, 30, 1)
    alert_sound = st.toggle("🔊 Play beep", value=True)
    # Each-way controls (global)
    st.markdown("---")
    st.markdown("### Each‑Way")
    ew_places = st.selectbox("Places", [2,3,4,5], index=1)
    ew_terms = st.selectbox("Terms", ["1/5", "1/4"], index=0)
    terms_frac = 0.20 if ew_terms=="1/5" else 0.25

# ------------------ DATA FETCH ------------------
def get_df(day="today") -> pd.DataFrame:
    if mock_mode:
        data = {
            "racecards":[
                {"region":"GB","course":c,"off_time":f"2025-08-09T{h:02d}:{m:02d}:00Z",
                 "runners":[{"horse":f"{c} {h:02d}{m:02d} #{i+1}"} for i in range(n)]}
                for c,n in zip(["Ascot","Newbury","York","Ayr","Chepstow","Lingfield"], [8,10,12,8,9,11])
                for h in range(12,20) for m in (0,30)
            ]
        }
        df = build_live_df_racingapi(data)
    else:
        df = build_live_df_betfair(day, countries=market_scope)
        if df.empty:
            st.warning("Betfair feed not available. Falling back to Racing API.")
            ok, data, _, _ = fetch_racecards_basic(day)
            if ok:
                df = build_live_df_racingapi(data)
    return df

def _trigger_alerts(df: pd.DataFrame):
    hot = df[(df["EV"] >= alert_ev) & (df["EdgeBrain Score"] >= alert_score)]
    if hot.empty: return
    msg = (hot.head(3)["Horse"] + " @" + hot.head(3)["Odds"].astype(str) + " (" +
           hot.head(3)["Course"] + " " + hot.head(3)["Time"] + ")").tolist()
    js = f"""
    <script>
      if (window.edgebetNotify) {{
        window.edgebetNotify('EdgeBet Hot Picks', '{", ".join(msg)}');
      }}
    </script>
    """
    components.html(js, height=0)
    if alert_sound:
        # lightweight ping
        st.audio(b"https://actions.google.com/sounds/v1/alarms/beep_short.ogg")

# ------------------ CHART HELPERS (Altair with green max) ------------------
def bar_green_max(df: pd.DataFrame, label_col: str, value_col: str, title: str):
    if df.empty:
        return None
    d = df[[label_col, value_col]].copy()
    mx = d[value_col].max()
    d["is_max"] = (d[value_col] == mx)
    chart = alt.Chart(d).mark_bar().encode(
        x=alt.X(f"{label_col}:N", sort='-y', title=None),
        y=alt.Y(f"{value_col}:Q", title=title),
        color=alt.condition(
            alt.datum.is_max,
            alt.value("#16a34a"),  # green
            alt.value("#00bfa6")   # teal
        ),
        tooltip=[label_col, value_col]
    ).properties(height=320)
    return chart

# ------------------ PAGES ------------------
if sel == "Overview":
    st.header("📊 Overview – Today")
    df_live = get_df("today")
    if df_live.empty:
        st.error("No data.")
    else:
        df_live = edgebrain_enhance(df_live)
        df_live["EW EV"] = df_live.apply(lambda r: each_way_ev(r["Odds"], r["Fair p"], ew_places, terms_frac), axis=1).round(3)
        _toolbar(df_live, "overview")

        cols = st.columns(4)
        with cols[0]:
            st.metric("Races Today", int(df_live[["Course","Time"]].drop_duplicates().shape[0]))
        with cols[1]:
            st.metric("Runners", len(df_live))
        with cols[2]:
            hot = (df_live["EV"] >= 0.05).sum()
            st.metric("🔥 Value Picks (EV≥0.05)", int(hot))
        with cols[3]:
            avg_over = df_live["Overround"].mean() if "Overround" in df_live else np.nan
            st.metric("Avg Overround", f"{avg_over:.2f}" if avg_over==avg_over else "—")

        st.subheader("Top EdgeBrain (Live)")
        top = df_live.sort_values(["EdgeBrain Score","EV"], ascending=[False,False]).head(20).copy()
        top["Label"] = top["Horse"].str.slice(0, 22) + " (" + top["Course"] + " " + top["Time"] + ")"
        ch = bar_green_max(top, "Label", "EdgeBrain Score", "EdgeBrain Score")
        if ch: st.altair_chart(ch, use_container_width=True)
        _trigger_alerts(df_live)

elif sel == "Horse Racing":
    # RESTORED TAB ✅
    st.header("🏇 Horse Racing – Live")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = get_df(day)
    if df.empty:
        st.error("No data available.")
    else:
        _toolbar(df, f"horses_{day}")
        # Classic filters
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
        country = st.selectbox("Country", countries, index=0)
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
            # Legacy BetEdge Win % (compat)
            filt = filt.copy()
            filt["BetEdge Win %"] = ((filt["Predicted Win %"] * 0.6) + (filt["Win_Value"] * 0.4)).round(1)
            min_v = int(np.floor(filt["BetEdge Win %"].min()))
            max_v = int(np.ceil(filt["BetEdge Win %"].max()))
            edge_range = st.slider("🎯 Filter by BetEdge Win %", min_v, max_v, (min_v, max_v))
            filt = filt[filt["BetEdge Win %"].between(*edge_range)]

            # Chart (BetEdge) with green max
            st.subheader("Top 20 BetEdge")
            topb = filt.sort_values("BetEdge Win %", ascending=False).head(20).copy()
            topb["Label"] = topb["Horse"].str.slice(0, 22) + " (" + topb["Course"] + " " + topb["Time"] + ")"
            chb = bar_green_max(topb, "Label", "BetEdge Win %", "BetEdge Win %")
            if chb: st.altair_chart(chb, use_container_width=True)

            # Table
            st.dataframe(
                filt[["Country","Course","Time","Horse","Odds","SP (proj)","Fair Odds","Overround","Predicted Win %","BetEdge Win %","Source"]],
                use_container_width=True,
                column_config=_format_cols(filt)
            )

elif sel == "Value Finder":
    st.header("🔥 Value Finder – Live")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = get_df(day)
    if df.empty:
        st.error("No data.")
    else:
        df = edgebrain_enhance(df)
        df["EW EV"] = df.apply(lambda r: each_way_ev(r["Odds"], r["Fair p"], ew_places, terms_frac), axis=1).round(3)
        _toolbar(df, f"value_{day}")

        # Filters
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
        c1, c2, c3, c4 = st.columns([0.2,0.4,0.2,0.2])
        with c1: country = st.selectbox("Country", countries, index=0)
        with c2:
            courses = sorted(df["Course"].dropna().unique().tolist())
            course_filter = st.multiselect("Courses", courses, default=courses)
        with c3: min_ev = st.slider("Min EV", -0.50, 1.00, 0.05, 0.01)
        with c4: min_score = st.slider("Min EdgeBrain", 0, 100, 22, 1)

        filt = df.copy()
        if country != "All": filt = filt[filt["Country"] == country]
        if course_filter: filt = filt[filt["Course"].isin(course_filter)]
        filt = filt[(filt["EV"] >= min_ev) & (filt["EdgeBrain Score"] >= min_score)]

        # Display
        if filt.empty:
            st.info("No selections match your filters.")
        else:
            def sparkline_cell(key):
                series = _sparkline_series(key, 30)
                if not series: return "—"
                mini = "".join("▁▂▃▄▅▆▇█"[min(7, max(0, int((series[i]-min(series))/(max(series)-min(series)+1e-9)*7)))] for i in range(len(series)))
                return f"<code style='font-size:12px'>{mini}</code>"

            show = filt.copy()
            show["Key"] = show["MarketId"].astype(str) + ":" + show["SelectionId"].astype(str)
            show["Drift"] = show["Key"].apply(sparkline_cell)
            cols = ["Badges","Risk","Horse","Course","Time","Odds","Fair Odds","EV","EW EV","Kelly %","Stake £",
                    "EdgeBrain Score","Momentum","Pressure","SP (proj)","Drift","Back Size","Lay Size","Overround","Source"]
            show = show[cols].sort_values(["EV","EdgeBrain Score"], ascending=[False, False])

            st.markdown("**Tip:** EV is per £1 stake; Kelly% is capped by bankroll settings.")
            st.dataframe(
                show,
                use_container_width=True,
                column_config=_format_cols(show),
                height=600
            )
        _trigger_alerts(df)

elif sel == "Dutching Lab":
    st.header("🧪 Dutching Lab")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = get_df(day)
    if df.empty:
        st.info("No markets available.")
    else:
        df = edgebrain_enhance(df)
        _toolbar(df, f"dutch_{day}")

        races = sorted((df["Course"] + " | " + df["Time"]).unique().tolist())
        race = st.selectbox("Race", races)
        subset = df[(df["Course"] + " | " + df["Time"]) == race].copy().sort_values(["EdgeBrain Score","EV"], ascending=[False, False])
        st.write("Select runners to dutch:")
        options = (subset["Horse"] + " @ " + subset["Odds"].astype(str)).tolist()
        selected = st.multiselect("Runners", options, default=options[:3])
        if selected:
            picked = subset[subset["Horse"] + " @ " + subset["Odds"].astype(str) .isin(selected)]
            total_stake = st.number_input("Total Stake £", min_value=1.0, step=1.0, value=50.0)
            stakes = dutch_stakes(picked["Odds"].tolist(), total_stake, method="equal_return")
            picked = picked.copy()
            picked["Stake £"] = stakes
            picked["Return £"] = (picked["Stake £"] * picked["Odds"]).round(2)
            target_return = picked["Return £"].iloc[0] if len(picked)>0 else 0.0

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Selections", len(picked))
            with c2: st.metric("Total Stake", f"£{stakes.sum():.2f}")
            with c3: st.metric("Target Return", f"£{target_return:.2f}")

            st.dataframe(
                picked[["Horse","Odds","EdgeBrain Score","EV","Kelly %","Stake £","Return £"]],
                use_container_width=True,
                column_config=_format_cols(picked)
            )
        else:
            st.info("Pick at least one runner to dutch.")

elif sel == "EdgeBrain":
    st.header("🧠 EdgeBrain – Model View")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = get_df(day)
    if df.empty:
        st.info("No data available.")
    else:
        df = edgebrain_enhance(df)
        df["EW EV"] = df.apply(lambda r: each_way_ev(r["Odds"], r["Fair p"], ew_places, terms_frac), axis=1).round(3)
        _toolbar(df, f"edge_{day}")

        st.dataframe(
            df[[
                "Country","Course","Time","Horse","Odds","Fair Odds","EV","EW EV","Kelly %","Stake £",
                "EdgeBrain Score","Momentum","Pressure","SP (proj)","Base p %","Overround",
                "Back Size","Lay Size","Market Status","Last Seen (UTC)","Source"
            ]].sort_values(["EdgeBrain Score","EV"], ascending=[False, False]),
            use_container_width=True,
            column_config=_format_cols(df)
        )

        # Odds Ladder Inspector
        st.markdown("---")
        with st.expander("🔎 Odds Ladder Inspector"):
            mid = st.text_input("MarketId")
            selid = st.number_input("SelectionId", min_value=1, step=1)
            @st.cache_data(ttl=10, show_spinner=False)
            def bf_ladder(market_id: str):
                token = bf_get_session_token()
                if not token: return {}
                params = {"marketIds":[market_id], "priceProjection":{"priceData":["EX_ALL_OFFERS"], "virtualise":True}}
                try:
                    res = bf_call("listMarketBook", params, token)
                    return res[0] if res else {}
                except Exception:
                    return {}
            if st.button("Fetch ladder") and mid:
                book = bf_ladder(mid)
                ladder = {}
                for r in (book.get("runners") or []):
                    if int(r.get("selectionId", -1)) == int(selid):
                        ex = r.get("ex", {}) or {}
                        atb = ex.get("availableToBack", []) or []
                        atl = ex.get("availableToLay", []) or []
                        ladder = {
                            "Back": [(lvl["price"], lvl["size"]) for lvl in atb[:25]],
                            "Lay":  [(lvl["price"], lvl["size"]) for lvl in atl[:25]],
                        }
                        break
                if not ladder:
                    st.info("Runner not found.")
                else:
                    lt = pd.DataFrame({
                        "Back Price": [p for p,_ in ladder["Back"]],
                        "Back Size":  [s for _,s in ladder["Back"]],
                        "Lay Price":  [p for p,_ in ladder["Lay"]],
                        "Lay Size":   [s for _,s in ladder["Lay"]],
                    })
                    st.dataframe(lt, use_container_width=True)

else:
    st.header("🐞 Debug")
    token = bf_get_session_token()
    st.write("Has Betfair token:", bool(token))
    cat = bf_list_win_markets("today", countries=market_scope)
    st.write("Catalogue markets:", len(cat))
    if cat:
        st.json(cat[0])

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
