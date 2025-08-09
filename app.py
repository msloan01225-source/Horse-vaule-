# edgebet_app.py — EdgeBet Ultra (Matrix + EdgeBrain v3 + Compact Readable Charts)
# --------------------------------------------------------------------------------
# • Clean, compact, BB-style readable charts that fit the page:
#     - Bars show end-value labels, Top-N cap, larger fonts
#     - Scatter is zoomable; tidy tooltips
#     - Heatmap auto-wraps when many races
#     - Race Tape tightened + Top runners only
# • Green→Red scales across charts (best→worst)
# • Predictor: Top 3 + Pin to Tracker + Race Tape
# • EdgeBrain weights: live sliders with reset
# --------------------------------------------------------------------------------

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
import altair as alt

with suppress(Exception):
    from streamlit_autorefresh import st_autorefresh

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="EdgeBet Ultra – Horse Racing", layout="wide", page_icon="🏇")

# ------------------ SESSION STATE ------------------
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"
if "price_hist" not in st.session_state:
    st.session_state.price_hist = defaultdict(lambda: deque(maxlen=90))
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 1000.0
if "tracker" not in st.session_state:
    st.session_state.tracker = []  # list[dict]
if "ui" not in st.session_state:
    st.session_state.ui = {"compact": True, "font": 12, "max_bars": 12, "top_runners": 6}
if "ebw" not in st.session_state:
    st.session_state.ebw = {
        "w_base":0.50,"w_mom":0.18,"w_press":0.10,"w_sp":0.06,
        "w_value":0.08,"w_liq":0.05,"w_spread":-0.04,"w_vol":-0.03
    }

# ------------------ HEADER ------------------
def toggle_theme():
    st.session_state["theme_mode"] = "light" if st.session_state["theme_mode"] == "dark" else "dark"

col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("🏇 EdgeBet Ultra – Horse Racing")
with col2:
    st.button("🌗 Theme", on_click=toggle_theme, use_container_width=True)

# ------------------ MATRIX THEME TOGGLES (SIDEBAR) ------------------
matrix_mode = st.sidebar.toggle("🟩 Matrix Theme", value=True, help="Neon code colors on dark background")
matrix_rain = st.sidebar.toggle("🌧️ Matrix Rain (subtle)", value=False, help="Animated code rain backdrop (experimental)")

# ------------------ DESIGN SYSTEM ------------------
PALETTE = {
    "neon": "#00FF6A","neon_dim":"#00cc56","purple":"#7C3AED","white":"#EDEDED",
    "red":"#EF4444","amber":"#F59E0B","ink":"#0A0F0A","ink_2":"#0D1512","grid":"#134E4A",
    "teal":"#00BFA6","green":"#16A34A"
}

def apply_design_system():
    if matrix_mode:
        bg, sbg, card, text, axis = PALETTE["ink"], PALETTE["ink_2"], "#0F1C16", PALETTE["white"], PALETTE["grid"]
        primary, accent = PALETTE["neon"], PALETTE["purple"]
    else:
        bg, sbg, card, text, axis = "#0f2a26", "#09211d", "#143f37", "#f2f2f2", "#bfeee8"
        primary, accent = PALETTE["teal"], PALETTE["green"]
    st.markdown(f"""
    <style>
      :root {{
        --c-primary:{primary}; --c-accent:{accent}; --c-bg:{bg}; --c-sbg:{sbg}; --c-card:{card};
        --c-text:{text}; --c-axis:{axis}; --c-danger:{PALETTE["red"]}; --c-warn:{PALETTE["amber"]};
      }}
      .stApp {{ background:var(--c-bg); color:var(--c-text); }}
      section[data-testid="stSidebar"] {{ background:var(--c-sbg) !important; }}
      .stMetric,.stTable,.stDataFrame {{ background:var(--c-card) !important; border-radius:10px; }}
      h1,h2,h3,h4,h5,h6 {{ color:var(--c-primary); text-shadow:0 0 10px rgba(0,255,106,.25); letter-spacing:.3px; }}
      [data-testid="stMetricValue"]{{ color:var(--c-primary) !important; }}
      ::-webkit-scrollbar{{ width:8px; height:8px; }} ::-webkit-scrollbar-thumb{{ background:var(--c-primary); border-radius:4px; }}
      .badge{{ display:inline-block; padding:2px 8px; border-radius:12px; font-weight:600; font-size:12px; }}
      .badge-ev{{ background:rgba(0,255,106,.10); color:var(--c-primary); border:1px solid rgba(0,255,106,.35); }}
      .badge-hot{{ background:rgba(124,58,237,.12); color:var(--c-accent); border:1px solid rgba(124,58,237,.35); }}
      .badge-cold{{ background:rgba(255,255,255,.06); color:#9FB; border:1px solid rgba(255,255,255,.1); }}
      .badge-risk{{ background:rgba(239,68,68,.12); color:var(--c-danger); border:1px solid rgba(239,68,68,.35); }}
      .stDataFrame td,.stDataFrame th{{ font-variant-numeric: tabular-nums; }}
    </style>
    """, unsafe_allow_html=True)

apply_design_system()

# ------------------ Readability helpers ------------------
def _label(text: str, maxlen: int = 22) -> str:
    s = str(text)
    return s if len(s) <= maxlen else (s[:maxlen-1] + "…")

def _font_cfg():
    fs = int(st.session_state.ui.get("font", 12))
    return {
        "axis": {"labelFontSize": fs, "titleFontSize": fs, "labelAngle": 0},
        "legend": {"labelFontSize": fs, "titleFontSize": fs},
        "header": {"labelFontSize": fs, "titleFontSize": fs}
    }

# ------------------ ALTair Theme ------------------
def _altair_theme():
    sizes = _font_cfg()
    return {"config":{
        "background":"transparent",
        "view":{"stroke":"transparent"},
        "axis":{**{"labelColor":"var(--c-text)","titleColor":"var(--c-text)","domainColor":"var(--c-axis)","tickColor":"var(--c-axis)","gridColor":"rgba(0,255,106,0.08)"}, **sizes["axis"]},
        "legend":{**{"labelColor":"var(--c-text)","titleColor":"var(--c-text)"}, **sizes["legend"]},
        "header": sizes["header"],
        "title":{"color":"var(--c-text)","fontSize": sizes["axis"]["titleFontSize"]+1}
    }}
alt.themes.register("edgebet", _altair_theme); alt.themes.enable("edgebet")

# ------------------ DESKTOP NOTIFICATIONS ------------------
components.html("""
<script>
  async function notify(title, body){
    try{
      if(!('Notification' in window)) return;
      if(Notification.permission==='granted'){new Notification(title,{body});}
      else if(Notification.permission!=='denied'){
        let p=await Notification.requestPermission();
        if(p==='granted') new Notification(title,{body});
      }
    }catch(e){}
  }
  window.edgebetNotify = notify;
</script>
""", height=0)

# ------------------ MATRIX RAIN (optional) ------------------
if matrix_rain:
    components.html("""
    <canvas id="codeRain" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;opacity:0.12;pointer-events:none;"></canvas>
    <script>
      const c=document.getElementById('codeRain'),x=c.getContext('2d');
      function r(){c.width=innerWidth;c.height=innerHeight} addEventListener('resize',r); r();
      const g="01<>[]{}$#&@*/\\=+-%",fs=14; let d=Array(Math.floor(c.width/fs)).fill(1+Math.random()*20);
      (function draw(){
        x.fillStyle="rgba(10,15,10,0.08)"; x.fillRect(0,0,c.width,c.height);
        x.fillStyle="#00FF6A"; x.font=fs+"px monospace";
        for(let i=0;i<d.length;i++){ const t=g[Math.floor(Math.random()*g.length)]; x.fillText(t,i*fs,d[i]*fs);
          if(d[i]*fs>c.height && Math.random()>0.975) d[i]=0; d[i]++; }
        requestAnimationFrame(draw);
      })();
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

# ------------------ HELPERS ------------------
def _iso_range_for_day(day: str) -> tuple[str, str]:
    now_ldn = datetime.now(ZoneInfo("Europe/London"))
    base = now_ldn.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ldn = base if day == "today" else base + timedelta(days=1)
    end_ldn = start_ldn + timedelta(days=1)
    start_utc, end_utc = start_ldn.astimezone(timezone.utc), end_ldn.astimezone(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return start_utc.strftime(fmt), end_utc.strftime(fmt)

def _to_hhmm(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    if len(s)==5 and s[2]==":" and s[:2].isdigit() and s[3:].isdigit(): return s
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(dt): return ""
    return dt.tz_convert("Europe/London").strftime("%H:%M")

def _chunks(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i+n]

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
        if col == "Exp £": fmt[col] = "£{:.2f}"
        if col == "Win p %": fmt[col] = "{:.1f}%"
    return fmt

def _toolbar(df: pd.DataFrame, key_prefix: str = "tb"):
    c1, c2, c3, c4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with c1:
        refresh_sec = st.selectbox("Auto-refresh", ["Off", 5, 10, 20, 30, 60], index=0, key=f"{key_prefix}_rf")
        if isinstance(refresh_sec, int):
            with suppress(Exception):
                st_autorefresh(interval=refresh_sec * 1000, key=f"{key_prefix}_autorf_{refresh_sec}")
    with c2:
        st.download_button("⬇️ CSV", data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"edgebet_{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True)
    with c3:
        st.number_input("Bankroll £", min_value=0.0, step=50.0, value=float(st.session_state.bankroll),
            key=f"{key_prefix}_bk", on_change=lambda: st.session_state.update(bankroll=st.session_state[f"{key_prefix}_bk"]))
    with c4:
        st.write(f"Rows: **{len(df):,}**")

def _sparkline_series(key: str, window: int = 60):
    dq = st.session_state.price_hist.get(key, deque(maxlen=90))
    if not dq: return []
    arr = list(dq)[-window:]
    return [x[1] for x in arr]

def _update_hist(key: str, odds: float):
    if odds is None: return
    ts = int(time.time())
    dq = st.session_state.price_hist[key]
    if not dq or dq[-1][0] != ts:
        dq.append((ts, float(odds)))

def _odds_vol(series: list[float]) -> float:
    if len(series) < 5: return 0.0
    arr = np.array(series[-30:], dtype=float)
    return float(np.clip(np.std(arr) / (np.mean(arr)+1e-9) * 100.0, 0.0, 10.0))

# ------------------ Unified green→red colour scales ------------------
GREEN = "#16A34A"; LIME = "#84CC16"; AMBER = "#F59E0B"; RED = "#EF4444"

def metric_scale(field: str, invert: bool = False):
    rng = [GREEN, LIME, AMBER, RED]
    if invert: rng = list(reversed(rng))
    return alt.Color(f"{field}:Q", scale=alt.Scale(range=rng), legend=alt.Legend(title=field))

# ------------------ Compact Chart Helpers ------------------
def bar_gradient(df: pd.DataFrame, label_col: str, value_col: str, title: str):
    if df.empty: return None
    N = int(st.session_state.ui.get("max_bars", 12))
    d = df[[label_col, value_col]].copy().head(N)
    d[label_col] = d[label_col].apply(lambda x: _label(str(x), 26))
    height = 24 * len(d) + 40
    base = alt.Chart(d).properties(height=height)
    bars = base.mark_bar(size=18, opacity=0.9).encode(
        x=alt.X(f"{value_col}:Q", title=title),
        y=alt.Y(f"{label_col}:N", sort='-x', title=None),
        color=metric_scale(value_col, invert=False),
        tooltip=[label_col, alt.Tooltip(f"{value_col}:Q", format=".1f" if "%" in value_col else ".2f")],
    )
    text = base.mark_text(align="left", dx=6, color="#EDEDED").encode(
        x=alt.X(f"{value_col}:Q"),
        y=alt.Y(f"{label_col}:N", sort='-x'),
        text=alt.Text(f"{value_col}:Q", format=".1f" if "%" in value_col else ".2f")
    )
    return bars + text

def scatter_ev_vs_score(df: pd.DataFrame, color_by: str = "EV"):
    if df.empty: return None
    d = df.copy()
    d["liq"] = np.clip(((d.get("Back Size", 0) + d.get("Lay Size", 0)) / 50.0), 10, 500)
    zoom = alt.selection_interval(bind='scales')
    return (alt.Chart(d).mark_circle(opacity=0.85, stroke="white", strokeWidth=0.3)
            .encode(
                x=alt.X("EV:Q", title="EV (£/£)", scale=alt.Scale(nice=True)),
                y=alt.Y("EdgeBrain Score:Q", scale=alt.Scale(nice=True)),
                size=alt.Size("liq:Q", legend=None, title="Liquidity"),
                color=metric_scale(color_by, invert=False),
                tooltip=["Horse","Course","Time",
                         alt.Tooltip("Odds:Q", format=".2f"),
                         alt.Tooltip("EV:Q", format="+.2f"),
                         alt.Tooltip("EdgeBrain Score:Q", format=".1f")]
            )
            .add_selection(zoom)
            .properties(height=320))

def heat_rank(df: pd.DataFrame, metric: str = "Win p %"):
    if df.empty or metric not in df.columns: return None
    d = df.copy()
    d["Race"] = d["Course"] + " " + d["Time"]
    d = d.sort_values(["Race", metric], ascending=[True, False]).copy()
    d["Horse Short"] = d["Horse"].astype(str).apply(lambda x: _label(x, 16))
    races = d["Race"].nunique()
    cols = 3 if races > 9 else 2 if races > 4 else 1
    cell_h = 20
    heat = (alt.Chart(d)
            .mark_rect()
            .encode(
                x=alt.X("Horse Short:N", sort=None, title=None),
                y=alt.Y("Race:N", sort=None, title=None),
                color=metric_scale(metric, invert=False),
                tooltip=["Race","Horse",alt.Tooltip("Odds:Q", format=".2f"),
                         alt.Tooltip(f"{metric}:Q", format=".1f")]
            )
            .properties(height=max(cell_h * min(races, 12) + 40, 120)))
    if cols > 1:
        heat = heat.facet(row=alt.Row("Race:N", header=alt.Header(labelAngle=0)), columns=cols)
    return heat

def sparkline_chart(series: list[float], current_ev: float | None = None):
    if not series:
        return alt.Chart(pd.DataFrame({"x":[0], "y":[0]})).mark_line().encode(x="x", y="y")
    dfc = pd.DataFrame({"x": list(range(len(series))), "y": series})
    good_path = (len(series)>1 and series[-1] <= series[0])
    line_col = GREEN if good_path else RED
    base = alt.Chart(dfc).mark_line(color=line_col, strokeWidth=2).encode(x="x:Q", y="y:Q").properties(height=60)
    if current_ev is None:
        return base
    ev_col = GREEN if current_ev >= 0 else RED
    glow = alt.Chart(dfc).mark_area(opacity=0.12, color=ev_col).encode(x="x:Q", y="y:Q")
    return glow + base

# ========= Race Tape helpers (EV heatstrip + odds trend) =========
def _history_for_race(race_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in race_df.iterrows():
        key = f"{r['MarketId']}:{r['SelectionId']}"
        dq = st.session_state.price_hist.get(key, deque())
        if not dq:
            continue
        fair_p = float(r.get("Fair p", np.nan))
        for ts, odds in dq:
            odds = float(odds)
            ev_t = fair_p * odds - (1 - fair_p)
            rows.append({
                "ts": pd.to_datetime(ts, unit="s", utc=True),
                "t_rel_s": ts - dq[0][0],
                "Odds_t": odds,
                "EV_t": ev_t,
                "Horse": r["Horse"],
                "Course": r["Course"],
                "Time": r["Time"]
            })
    return pd.DataFrame(rows)

def chart_ev_heatstrip(race_df: pd.DataFrame, top_n: int | None = None):
    hist = _history_for_race(race_df)
    if hist.empty: return None
    metric = "Win p %" if "Win p %" in race_df.columns else "EdgeBrain Score"
    N = top_n or int(st.session_state.ui.get("top_runners", 6))
    keep = race_df.sort_values(metric, ascending=False).head(N)["Horse"].tolist()
    d = hist[hist["Horse"].isin(keep)].copy()
    d["t_bin"] = (d["t_rel_s"] // 5).astype(int)
    agg = d.groupby(["Horse","t_bin"], as_index=False).agg(EV_t=("EV_t","mean"))
    height = 22 * len(keep) + 40
    return (alt.Chart(agg).mark_rect()
            .encode(
                x=alt.X("t_bin:Q", title="Time (×5s)", axis=alt.Axis(values=[0,30,60,90,120,150])),
                y=alt.Y("Horse:N", sort=keep, title=None),
                color=metric_scale("EV_t", invert=False),
                tooltip=[alt.Tooltip("Horse:N"), alt.Tooltip("t_bin:Q", title="t (×5s)"),
                         alt.Tooltip("EV_t:Q", format="+.2f")]
            )
            .properties(height=height))

def chart_odds_trend(race_df: pd.DataFrame, top_n: int | None = None):
    hist = _history_for_race(race_df)
    if hist.empty: return None
    metric = "Win p %" if "Win p %" in race_df.columns else "EdgeBrain Score"
    N = top_n or int(st.session_state.ui.get("top_runners", 6))
    keep = race_df.sort_values(metric, ascending=False).head(N)["Horse"].tolist()
    d = hist[hist["Horse"].isin(keep)].copy()
    last_ev = d.sort_values("t_rel_s").groupby("Horse").tail(1)[["Horse","EV_t"]].set_index("Horse")["EV_t"]
    d = d.merge(last_ev.rename("EV_last"), left_on="Horse", right_index=True, how="left")
    d["EV_pos"] = np.where(d["EV_last"] >= 0, "Good", "Bad")
    color_scale = alt.Scale(domain=["Good","Bad"], range=[GREEN, RED])
    zoom = alt.selection_interval(bind='scales')
    return (alt.Chart(d).mark_line(strokeWidth=2)
            .encode(
                x=alt.X("t_rel_s:Q", title="Seconds"),
                y=alt.Y("Odds_t:Q", title="Odds", scale=alt.Scale(nice=True)),
                detail="Horse:N",
                color=alt.Color("EV_pos:N", scale=color_scale, legend=alt.Legend(title="Latest EV")),
                tooltip=["Horse","t_rel_s",alt.Tooltip("Odds_t:Q", format=".2f"),alt.Tooltip("EV_t:Q", format="+.2f")]
            )
            .add_selection(zoom)
            .properties(height=260))

# ------------------ BETFAIR API ------------------
@st.cache_resource
def bf_get_session_token():
    if not BF_APP_KEY:
        st.error("Betfair app_key missing in secrets."); return None
    if BF_SESSION: return BF_SESSION
    if BF_USER and BF_PASS:
        try:
            resp = requests.post(BF_IDENTITY_URL,
                headers={"X-Application": BF_APP_KEY, "Content-Type": "application/x-www-form-urlencoded"},
                data={"username": BF_USER, "password": BF_PASS}, timeout=12)
            resp.raise_for_status(); data = resp.json()
            return data.get("token") if data.get("status") == "SUCCESS" else None
        except Exception as e:
            st.error(f"Betfair login error: {e}"); return None
    st.warning("No Betfair session token and no username/password provided."); return None

def bf_headers(token): return {"X-Application": BF_APP_KEY, "X-Authentication": token, "Content-Type": "application/json"}

def _bf_post(method: str, params: dict, token: str):
    payload = {"jsonrpc":"2.0","method":f"SportsAPING/v1.0/{method}","params":params,"id":1}
    r = requests.post(BF_API_URL, headers=bf_headers(token), data=json.dumps(payload), timeout=12)
    r.raise_for_status(); return r.json()

def bf_call(method: str, params: dict, token: str):
    out = _bf_post(method, params, token)
    if "error" not in out: return out["result"]
    code = (out["error"].get("data") or {}).get("APINGException", {}).get("errorCode")
    if code in {"INVALID_SESSION_INFORMATION","NO_APP_KEY"}:
        bf_get_session_token.clear(); new_token = bf_get_session_token()
        if not new_token: raise RuntimeError(out["error"])
        out2 = _bf_post(method, params, new_token)
        if "error" in out2: raise RuntimeError(out2["error"])
        return out2["result"]
    raise RuntimeError(out["error"])

# ------------------ DATA BUILDERS ------------------
def _best_prices_sizes(rb: dict):
    ex = rb.get("ex", {}) or {}
    atb, atl = ex.get("availableToBack", []) or [], ex.get("availableToLay", []) or []
    best_back = atb[0]["price"] if atb else None
    best_lay  = atl[0]["price"] if atl else None
    back_sz = sum([lvl.get("size", 0) for lvl in atb[:3]])
    lay_sz  = sum([lvl.get("size", 0) for lvl in atl[:3]])
    sp = (rb.get("sp", {}) or {}).get("farPrice") or (rb.get("sp", {}) or {}).get("nearPrice")
    return best_back, best_lay, back_sz, lay_sz, sp

@st.cache_data(ttl=60, show_spinner=False)
def bf_list_win_markets(day="today", countries=None):
    countries = countries or ["GB"]
    token = bf_get_session_token()
    if not token: return []
    fr, to = _iso_range_for_day(day)
    params = {
        "filter": {"eventTypeIds":["7"], "marketCountries":countries, "marketTypeCodes":["WIN"], "marketStartTime":{"from":fr,"to":to}},
        "maxResults": 200, "marketProjection": ["RUNNER_DESCRIPTION","MARKET_START_TIME","EVENT","RUNNER_METADATA"]
    }
    try: return bf_call("listMarketCatalogue", params, token)
    except Exception: st.error("Betfair listMarketCatalogue error."); return []

@st.cache_data(ttl=30, show_spinner=False)
def bf_list_market_books(market_ids: list[str]):
    token = bf_get_session_token()
    if not token or not market_ids: return []
    books = []
    for chunk in _chunks(market_ids, 25):
        params = {"marketIds":chunk, "priceProjection":{"priceData":["EX_BEST_OFFERS","SP_PROJECTED","SP_TRADED"], "virtualise":True}}
        last_err = None
        for attempt in range(3):
            try: res = bf_call("listMarketBook", params, token); books.extend(res or []); last_err=None; break
            except Exception as e: last_err = e; time.sleep(0.25*(attempt+1))
        if last_err: st.warning(f"Betfair price chunk failed ({len(chunk)} IDs). Continuing.")
        time.sleep(0.05)
    return books

def build_live_df_betfair(day="today", countries=None) -> pd.DataFrame:
    cat = bf_list_win_markets(day, countries=countries)
    if not cat: return pd.DataFrame()
    ids = [m.get("marketId") for m in cat if m.get("marketId")]
    books = bf_list_market_books(ids) if ids else []

    price_map, status_map = {}, {}
    for mb in books:
        mid = mb.get("marketId"); status_map[mid] = mb.get("status", "UNKNOWN")
        sel_map = {}
        for rb in (mb.get("runners") or []):
            bb, bl, bsz, lsz, sp = _best_prices_sizes(rb)
            sel_map[rb.get("selectionId")] = (bb, bl, bsz, lsz, sp)
        price_map[mid] = sel_map

    rows = []; now_utc = datetime.utcnow()
    for m in cat:
        mid = m.get("marketId"); venue = (m.get("event", {}) or {}).get("venue", "Unknown")
        tstr = _to_hhmm(m.get("marketStartTime", "")); mstatus = status_map.get(mid, "UNKNOWN")
        for r in (m.get("runners", []) or []):
            name = r.get("runnerName"); sel = r.get("selectionId")
            bb, bl, bsz, lsz, sp = (price_map.get(mid, {}) or {}).get(sel, (None, None, 0.0, 0.0, None))
            if not name or bb is None: continue
            key = f"{mid}:{sel}"; _update_hist(key, bb)
            rows.append({
                "MarketId": mid, "SelectionId": sel, "Course": venue, "Time": tstr, "Horse": name,
                "Odds": round(float(bb), 2), "Best Lay": round(float(bl), 2) if bl else None,
                "Back Size": float(bsz or 0.0), "Lay Size": float(lsz or 0.0),
                "SP (proj)": round(float(sp), 2) if sp else None, "Market Status": mstatus,
                "Last Seen (UTC)": now_utc.strftime("%H:%M:%S"), "Source": "Betfair (live)"
            })
    return pd.DataFrame(rows)

# Racing API fallback (names + mock odds)
@st.cache_data(ttl=120, show_spinner=False)
def fetch_racecards_basic(day="today"):
    user = RA_USER; pwd = RA_PASS
    if not user or not pwd: return False, None, None, "No Racing API credentials in secrets."
    url = "https://api.theracingapi.com/v1/racecards"; params = {"day": day}
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(user, pwd), params=params, timeout=12)
        resp.raise_for_status(); return True, resp.json(), resp.status_code, resp.text
    except Exception as e:
        return False, None, None, str(e)

def build_live_df_racingapi(payload: dict) -> pd.DataFrame:
    if not payload or "racecards" not in payload: return pd.DataFrame()
    rows = []
    for race in payload.get("racecards", []) or []:
        course = race.get("course"); off = _to_hhmm(race.get("off_time", ""))
        for idx, rnr in enumerate(race.get("runners", []) or []):
            horse = rnr.get("horse")
            if not horse: continue
            if idx == 0: best_back = np.random.uniform(2.2, 3.8)
            elif idx <= 2: best_back = np.random.uniform(4.0, 7.0)
            elif idx <= 5: best_back = np.random.uniform(7.5, 15.0)
            else: best_back = np.random.uniform(16.0, 40.0)
            proj_sp = best_back * np.random.uniform(0.95, 1.05)
            key = f"MOCK:{course}:{off}:{horse}"; _update_hist(key, best_back)
            rows.append({
                "MarketId": f"MOCK:{course}:{off}", "SelectionId": idx+1, "Course": course, "Time": off, "Horse": horse,
                "Odds": round(float(best_back), 2), "Best Lay": round(float(best_back*1.02), 2),
                "Back Size": float(np.random.uniform(50, 1000)), "Lay Size": float(np.random.uniform(50, 1000)),
                "SP (proj)": round(float(proj_sp), 2), "Market Status": "OPEN",
                "Last Seen (UTC)": datetime.utcnow().strftime("%H:%M:%S"), "Source": "Racing API (names) + mock odds"
            })
    return pd.DataFrame(rows)

# ------------------ DE‑VIG & FAIR PROBS ------------------
def devig_power(odds: pd.Series) -> pd.Series:
    x = np.clip(1.0 / np.clip(odds.astype(float), 1.01, None), 1e-9, None)
    def sum_p(alpha: float) -> float: return float(np.sum(np.power(x, alpha)))
    lo, hi = 0.5, 2.0
    for _ in range(40):
        mid = 0.5*(lo+hi); s = sum_p(mid)
        if s > 1.0: lo = mid
        else: hi = mid
    alpha = 0.5*(lo+hi); p = np.power(x, alpha)
    return pd.Series(p / p.sum())

def add_fair_probs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy(); df["RaceKey"] = df["Course"].astype(str) + " | " + df["Time"].astype(str)
    fair_list = []
    for _, sub in df.groupby("RaceKey"):
        probs = devig_power(sub["Odds"]); fair_list.append(pd.DataFrame({"idx": sub.index, "Fair p": probs.values}))
    fair = pd.concat(fair_list, ignore_index=True) if fair_list else pd.DataFrame(columns=["idx","Fair p"])
    df = df.merge(fair.set_index("idx"), left_index=True, right_index=True, how="left")
    df["Fair Odds"] = (1.0 / df["Fair p"].clip(1e-6, None)).round(2)
    over = df.groupby("RaceKey")["Odds"].apply(lambda s: (1.0 / s.clip(lower=1.01)).sum()).rename("Overround")
    df = df.merge(over, on="RaceKey", how="left"); return df

# ------------------ EDGE BRAIN v3 (with live weights) ------------------
def _momentum(series_odds: list[float]) -> float:
    if len(series_odds) < 3: return 0.0
    y = np.array(series_odds[-12:]); x = np.arange(len(y))
    p = 1.0 / np.clip(y, 1.01, None)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, p, rcond=None)[0]
    return float(np.clip(slope * 100.0, -6.0, 6.0))

def _pressure(back_sz: float, lay_sz: float) -> float:
    total = max(back_sz + lay_sz, 1e-6)
    ratio = (back_sz - lay_sz) / total
    return float(np.clip(ratio * 10.0, -6.0, 6.0))

def _spread(best_back, best_lay) -> float:
    if not best_back or not best_lay: return 0.0
    mid = 0.5 * (best_back + best_lay)
    return float(np.clip((best_lay - best_back) / (mid + 1e-9) * 100.0, 0.0, 12.0))

def _liquidity(back_sz, lay_sz) -> float:
    depth = (back_sz or 0.0) + (lay_sz or 0.0)
    return float(np.clip(math.log10(max(depth, 1.0)), 0.0, 4.0))  # 0..4

def edgebrain_enhance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    keys = (out["MarketId"].astype(str) + ":" + out["SelectionId"].astype(str)).tolist()
    series = [ _sparkline_series(k, 60) for k in keys ]
    out["Momentum"] = [ _momentum(s) for s in series ]
    out["Vol %"]     = [ _odds_vol(s) for s in series ]
    out["Pressure"]  = out.apply(lambda r: _pressure(r["Back Size"], r["Lay Size"]), axis=1)
    out["Spread %"]  = out.apply(lambda r: _spread(r.get("Odds"), r.get("Best Lay")), axis=1)
    out["LiqScore"]  = out.apply(lambda r: _liquidity(r["Back Size"], r["Lay Size"]), axis=1)
    out["SP Deviation %"] = np.where(out["SP (proj)"].notna(),
                                     (out["SP (proj)"] - out["Odds"]) / out["Odds"] * 100.0, 0.0).round(1)
    out["Fair p"] = out["Fair p"].clip(0.001, 0.999)
    out["Base p %"] = (out["Fair p"] * 100.0).round(2)

    value_boost = ((1.0/np.clip(out["Odds"],1.01,None)) - out["Fair p"]) * 100.0
    w = st.session_state.get("ebw", {"w_base":0.50,"w_mom":0.18,"w_press":0.10,"w_sp":0.06,"w_value":0.08,"w_liq":0.05,"w_spread":-0.04,"w_vol":-0.03})
    out["EdgeBrain Score"] = (
        w["w_base"]   * out["Base p %"] +
        w["w_mom"]    * np.clip(out["Momentum"] + 1.5, -6.0, 6.0) +
        w["w_press"]  * out["Pressure"] +
        w["w_sp"]     * np.clip(-out["SP Deviation %"]/2.0, -6.0, 6.0) +
        w["w_value"]  * value_boost +
        w["w_liq"]    * out["LiqScore"] * 10.0 +
        w["w_spread"] * out["Spread %"] +
        w["w_vol"]    * out["Vol %"]
    ).clip(0, 100).round(1)

    p = out["Fair p"]; q = 1 - p; b = out["Odds"] - 1.0
    out["EV"] = (p * (out["Odds"]) - q).round(2)
    kelly = (b * p - q) / (b + 1e-9)
    bankroll = max(float(st.session_state.bankroll), 0.0); cap = bankroll * 0.02
    out["Kelly %"] = (np.clip(kelly, 0, 1.0) * 100.0).round(1)
    out["Stake £"] = np.minimum(cap, bankroll * np.clip(kelly, 0, 1.0)).round(2)

    out["Risk"] = np.select([out["EdgeBrain Score"] >= 38, out["EdgeBrain Score"] >= 24], ["✅", "⚠️"], default="❌")
    conf_raw = (out["Momentum"].clip(-3,3) + out["Pressure"].clip(-3,3) + (value_boost.clip(-6,6)) + (out["LiqScore"]-1.5)*2) / 4.0
    out["Confidence %"] = (50 + conf_raw * 10).clip(0, 100).round(0)

    out["Badges"] = out.apply(lambda r: "".join([
        f'<span class="badge badge-ev">EV {r["EV"]:+.2f}</span> ' if r["EV"] >= 0.05 else "",
        f'<span class="badge badge-hot">🔥</span> ' if r["Momentum"]>0.5 else "",
        f'<span class="badge badge-cold">🧊</span> ' if r["Momentum"]<-0.7 else "",
        f'<span class="badge badge-risk">Risk</span> ' if r["Risk"]=="❌" else ""
    ]), axis=1)

    return out

# ------------------ EW / ALERTS ------------------
def each_way_ev(odds, win_p, places=3, terms_frac=0.25):
    odds = float(odds); p_win = float(win_p)
    place_p = min(1.0, p_win * (1 + 0.9*(places-1)))
    win_leg_ev = p_win * (odds - 1) - (1 - p_win) * 1
    place_odds = 1 + terms_frac * (odds - 1)
    place_leg_ev = place_p * (place_odds - 1) - (1 - place_p) * 1
    return 0.5*win_leg_ev + 0.5*place_leg_ev

def trigger_alerts(df: pd.DataFrame, alert_ev: float, alert_score: int, play_sound=True):
    hot = df[(df["EV"] >= alert_ev) & (df["EdgeBrain Score"] >= alert_score)]
    if hot.empty: return
    msg = (hot.head(3)["Horse"] + " @" + hot.head(3)["Odds"].astype(str) + " (" +
           hot.head(3)["Course"] + " " + hot.head(3)["Time"] + ")").tolist()
    js = f"""<script> if (window.edgebetNotify) {{ window.edgebetNotify('EdgeBet Hot Picks', '{", ".join(msg)}'); }} </script>"""
    components.html(js, height=0)
    if play_sound:
        st.audio(b"https://actions.google.com/sounds/v1/alarms/beep_short.ogg")

# ------------------ PREDICTOR (Top 3) ------------------
def predict_top3(race_df: pd.DataFrame) -> pd.DataFrame:
    if race_df.empty: return race_df
    df = race_df.copy()
    score = (1.10 * df["Fair p"] + 0.015 * df["Momentum"] + 0.010 * df["Pressure"] +
             0.006 * (-df["SP Deviation %"]) + 0.002 * (df["LiqScore"] * 10.0) +
             -0.002 * df["Spread %"] + -0.0015 * df["Vol %"] +
             0.004 * np.clip((1.0/np.clip(df["Odds"],1.01,None)) - df["Fair p"], -0.2, 0.2) * 100.0)
    temp = 0.85; logits = score / max(temp, 1e-6)
    logits = logits - logits.max()
    winp = np.exp(np.clip(logits, -50, 50)); winp = winp / (winp.sum() + 1e-9)
    df["Win p %"] = (winp * 100.0).round(1)
    out = df.sort_values("Win p %", ascending=False).copy()
    out["Rank"] = range(1, len(out)+1)
    return out

def explain_row(r):
    return {
        "Fair p (base)": float(r["Fair p"]),
        "Momentum": float(r["Momentum"]),
        "Pressure": float(r["Pressure"]),
        "SP Δ%": float(r["SP Deviation %"]),
        "Liquidity": float(r["LiqScore"]*10.0),
        "Spread%": float(r["Spread %"]),
        "Vol%": float(r["Vol %"]),
        "Value Δ(pp)": float(((1.0/max(r["Odds"],1.01)) - r["Fair p"]) * 100.0)
    }

# ------------------ DATA FETCH (with fallbacks) ------------------
def get_df(day="today") -> pd.DataFrame:
    if st.session_state.get("mock_mode", False):
        data = {
            "racecards":[
                {"region":"GB","course":c,"off_time":f"2025-08-09T{h:02d}:{m:02d}:00Z",
                 "runners":[{"horse":f"{c} {h:02d}{m:02d} #{i+1}"} for i in range(n)]}
                for c,n in zip(["Ascot","Newbury","York","Ayr","Chepstow","Lingfield"], [8,10,12,8,9,11])
                for h in range(12,20) for m in (0,30)
            ]
        }
        base = build_live_df_racingapi(data)
    else:
        base = build_live_df_betfair(day, countries=st.session_state.get("market_scope", ["GB"]))
        if base.empty:
            st.warning("Betfair feed not available. Falling back to Racing API.")
            ok, payload, _, _ = fetch_racecards_basic(day)
            base = build_live_df_racingapi(payload) if ok else pd.DataFrame()
    if base.empty: return base

    key = (base["Course"].astype(str) + "|" + base["Time"].astype(str) + "|" + base["Horse"].astype(str))
    base["Win_Value"] = (5 + key.apply(_stable_rand01) * 25).round(1)
    base["Predicted Win %"] = (100.0 / base["Odds"].clip(lower=1.01)).round(1)
    base["BetEdge Win %"] = ((base["Predicted Win %"] * 0.6) + (base["Win_Value"] * 0.4)).round(1)

    base = add_fair_probs(base)
    base = edgebrain_enhance(base)
    places = st.session_state.get("ew_places", 3)
    terms_frac = st.session_state.get("ew_terms_frac", 0.25)
    base["EW EV"] = base.apply(lambda r: each_way_ev(r["Odds"], r["Fair p"], places, terms_frac), axis=1).round(3)
    return base

# ------------------ SIDEBAR NAV ------------------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet Ultra",
        ["Overview", "Horse Racing", "Value Finder", "Predictor", "EdgeBrain", "Bet Tracker", "Debug"],
        icons=["house", "activity", "fire", "trophy", "robot", "bookmark", "bug"],
        default_index=0
    )
    st.session_state.mock_mode = st.toggle("🧪 Mock mode (no APIs)", value=False)
    st.session_state.market_scope = st.multiselect("🌍 Market countries", options=["GB", "IE"], default=["GB"])
    st.markdown("---")
    st.markdown("### Alerts")
    s_alert_ev = st.slider("EV ≥", -0.50, 1.0, 0.10, 0.01)
    s_alert_score = st.slider("EdgeBrain ≥", 0, 100, 32, 1)
    s_alert_sound = st.toggle("🔊 Beep", value=True)

    st.markdown("---")
    st.markdown("### Each‑Way (global)")
    ew_places = st.selectbox("Places", [2,3,4,5], index=1)
    ew_terms = st.selectbox("Terms", ["1/5", "1/4"], index=0)
    st.session_state.ew_places = ew_places
    st.session_state.ew_terms_frac = 0.20 if ew_terms=="1/5" else 0.25

    st.markdown("---")
    st.markdown("### 🧠 EdgeBrain Weights (live)")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.ebw["w_base"]   = st.slider("Fair p (base)", 0.0, 1.0, st.session_state.ebw["w_base"], 0.01)
        st.session_state.ebw["w_mom"]    = st.slider("Momentum",      -1.0, 1.0, st.session_state.ebw["w_mom"], 0.01)
        st.session_state.ebw["w_press"]  = st.slider("Pressure",      -1.0, 1.0, st.session_state.ebw["w_press"], 0.01)
        st.session_state.ebw["w_sp"]     = st.slider("SP Δ (− good)", -1.0, 1.0, st.session_state.ebw["w_sp"], 0.01)
    with c2:
        st.session_state.ebw["w_value"]  = st.slider("Value Δ",       -1.0, 1.0, st.session_state.ebw["w_value"], 0.01)
        st.session_state.ebw["w_liq"]    = st.slider("Liquidity",     -1.0, 1.0, st.session_state.ebw["w_liq"], 0.01)
        st.session_state.ebw["w_spread"] = st.slider("Spread penalty",-1.0, 0.0, st.session_state.ebw["w_spread"], 0.01)
        st.session_state.ebw["w_vol"]    = st.slider("Vol penalty",   -1.0, 0.0, st.session_state.ebw["w_vol"], 0.01)
    if st.button("↺ Reset weights to defaults"):
        st.session_state.ebw = {"w_base":0.50,"w_mom":0.18,"w_press":0.10,"w_sp":0.06,"w_value":0.08,"w_liq":0.05,"w_spread":-0.04,"w_vol":-0.03}
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### 📐 Chart Readability")
    st.session_state.ui["compact"] = st.toggle("Compact charts", value=st.session_state.ui["compact"])
    st.session_state.ui["font"] = st.slider("Font size", 10, 16, st.session_state.ui["font"])
    st.session_state.ui["max_bars"] = st.slider("Max bars (Top N lists)", 6, 30, st.session_state.ui["max_bars"])
    st.session_state.ui["top_runners"] = st.slider("Top runners in Race Tape", 4, 10, st.session_state.ui["top_runners"])

# ------------------ PAGES ------------------
if sel == "Overview":
    st.header("📊 Overview – Today")
    df_live = get_df("today")
    if df_live.empty:
        st.error("No data.")
    else:
        _toolbar(df_live, "overview")
        cols = st.columns(4)
        with cols[0]: st.metric("Races Today", int(df_live[["Course","Time"]].drop_duplicates().shape[0]))
        with cols[1]: st.metric("Runners", len(df_live))
        with cols[2]: st.metric("🔥 Value (EV≥0.05)", int((df_live["EV"] >= 0.05).sum()))
        with cols[3]: st.metric("Avg Overround", f"{df_live['Overround'].mean():.2f}")

        st.subheader("Top EdgeBrain (Live)")
        top = df_live.sort_values(["EdgeBrain Score","EV"], ascending=[False,False]).copy()
        top["Label"] = (top["Horse"].str.slice(0, 24) + " (" + top["Course"] + " " + top["Time"] + ")")
        top = top.head(int(st.session_state.ui.get("max_bars", 12)))
        ch = bar_gradient(top, "Label", "EdgeBrain Score", "EdgeBrain Score")
        if ch: st.altair_chart(ch, use_container_width=True)

        # Compact Race Tape for the next race
        try:
            df_live["t_sort"] = pd.to_datetime(df_live["Time"], format="%H:%M", errors="coerce")
            next_race_key = (
                df_live.sort_values("t_sort")[["Course","Time"]]
                .dropna().head(1).astype(str).agg(" | ".join, axis=1).iloc[0]
            )
            mini = df_live[(df_live["Course"] + " | " + df_live["Time"]) == next_race_key]
            st.markdown(f"#### 📼 Next Race Tape — {next_race_key}")
            mini_heat = chart_ev_heatstrip(mini)
            if mini_heat: st.altair_chart(mini_heat, use_container_width=True)
        except Exception:
            pass

        trigger_alerts(df_live, s_alert_ev, s_alert_score, s_alert_sound)

elif sel == "Horse Racing":
    st.header("🏇 Horse Racing – Live (Classic)")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = get_df(day)
    if df.empty:
        st.error("No data available.")
    else:
        _toolbar(df, f"horses_{day}")
        countries = ["All"] + sorted(df.get("Country", pd.Series()).dropna().unique().tolist()) if "Country" in df else ["All"]
        country = st.selectbox("Country", countries, index=0)
        courses = sorted(df["Course"].dropna().unique().tolist())
        course_filter = st.multiselect("Courses", courses, default=courses)

        filt = df.copy()
        if country != "All" and "Country" in filt: filt = filt[filt["Country"] == country]
        if course_filter: filt = filt[filt["Course"].isin(course_filter)]

        if filt.empty:
            st.info("No selections match your filters.")
        else:
            st.subheader("Top 20 BetEdge")
            topb = filt.sort_values("BetEdge Win %", ascending=False).copy()
            topb["Label"] = (topb["Horse"].str.slice(0, 24) + " (" + topb["Course"] + " " + topb["Time"] + ")")
            topb = topb.head(int(st.session_state.ui.get("max_bars", 12)))
            chb = bar_gradient(topb, "Label", "BetEdge Win %", "BetEdge Win %")
            if chb: st.altair_chart(chb, use_container_width=True)

            min_v = int(np.floor(filt["BetEdge Win %"].min()))
            max_v = int(np.ceil(filt["BetEdge Win %"].max()))
            edge_range = st.slider("🎯 Filter by BetEdge Win %", min_v, max_v, (min_v, max_v))
            filt = filt[filt["BetEdge Win %"].between(*edge_range)]

            st.dataframe(
                filt[["Course","Time","Horse","Odds","SP (proj)","Fair Odds","Overround","Predicted Win %","BetEdge Win %","Source"]],
                use_container_width=True, column_config=_format_cols(filt), height=520
            )

elif sel == "Value Finder":
    st.header("🔥 Value Finder – Live")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = get_df(day)
    if df.empty:
        st.error("No data.")
    else:
        _toolbar(df, f"value_{day}")
        countries = ["All"] + sorted(df.get("Country", pd.Series()).dropna().unique().tolist()) if "Country" in df else ["All"]
        c1, c2, c3, c4 = st.columns([0.2,0.4,0.2,0.2])
        with c1: country = st.selectbox("Country", countries, index=0)
        with c2:
            courses = sorted(df["Course"].dropna().unique().tolist())
            course_filter = st.multiselect("Courses", courses, default=courses)
        with c3: min_ev = st.slider("Min EV", -0.50, 1.00, 0.05, 0.01)
        with c4: min_score = st.slider("Min EdgeBrain", 0, 100, 24, 1)

        filt = df.copy()
        if country != "All" and "Country" in filt: filt = filt[filt["Country"] == country]
        if course_filter: filt = filt[filt["Course"].isin(course_filter)]
        filt = filt[(filt["EV"] >= min_ev) & (filt["EdgeBrain Score"] >= min_score)]

        if filt.empty:
            st.info("No selections match your filters.")
        else:
            show = filt.copy()
            show["Key"] = show["MarketId"].astype(str) + ":" + show["SelectionId"].astype(str)
            cols = ["Badges","Risk","Horse","Course","Time","Odds","Fair Odds","EV","EW EV","Kelly %","Stake £",
                    "EdgeBrain Score","Momentum","Pressure","SP (proj)","Back Size","Lay Size","Spread %","Vol %","Overround","Source"]
            show = show[cols].sort_values(["EV","EdgeBrain Score"], ascending=[False, False])
            st.dataframe(show, use_container_width=True, column_config=_format_cols(show), height=560)

            sc = scatter_ev_vs_score(filt, color_by="EV")
            if sc: st.altair_chart(sc, use_container_width=True)

        trigger_alerts(df, s_alert_ev, s_alert_score, s_alert_sound)

elif sel == "Predictor":
    st.header("🏆 Predictor — Top 3 by Race")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True, key="pred_day")
    df = get_df(day)
    if df.empty:
        st.info("No markets available.")
    else:
        _toolbar(df, f"predictor_{day}")
        hm = heat_rank(df, metric="Win p %") if "Win p %" in df.columns else None
        if hm: st.altair_chart(hm, use_container_width=True)

        races = sorted((df["Course"] + " | " + df["Time"]).unique().tolist())
        q = st.text_input("🔎 Search race (type course/time)", placeholder="e.g., Ascot 14:30 or York")
        choices = [r for r in races if (q or "").strip().lower() in r.lower()] if q else races
        race = st.selectbox("Race", choices, index=0 if choices else None)
        if not race:
            st.warning("No matching races.")
        else:
            rd = df[(df["Course"] + " | " + df["Time"]) == race].copy()
            picks = predict_top3(rd); top3 = picks.head(3).copy()

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Runners", len(rd))
            with c2: st.metric("Overround", f"{rd['Overround'].mean():.2f}")
            with c3: st.metric("Top Win p%", f"{top3['Win p %'].iloc[0]:.1f}%")
            with c4: st.metric("Top EV", f"{rd['EV'].max():+.2f}")

            st.subheader("Top 3 — Predicted Win Probability")
            display = top3[["Horse","Odds","Fair Odds","Win p %","EdgeBrain Score","EV","Kelly %","Stake £"]].copy()
            st.dataframe(display, use_container_width=True, column_config=_format_cols(display))

            # Pin Top 3 to Tracker
            c_add1, c_add2 = st.columns([0.7, 0.3])
            with c_add1:
                st.caption("Add these to your Bet Tracker with a single click.")
            with c_add2:
                stake_default = float(np.clip(st.session_state.bankroll * 0.01, 1.0, 100.0))
                pin_stake = st.number_input("Stake £", min_value=0.5, step=0.5, value=stake_default, key="pin_top3_stake")
            if st.button("📌 Pin Top 3 to Tracker"):
                for _, r in top3.iterrows():
                    entry = {
                        "Course": r["Course"], "Time": r["Time"], "Horse": r["Horse"],
                        "Odds": float(r["Odds"]), "Fair p": float(r["Fair p"]), "Fair Odds": float(r["Fair Odds"]),
                        "EV": float(r["EV"]), "Kelly %": float(r["Kelly %"]), "Stake £": float(pin_stake),
                        "EdgeBrain Score": float(r["EdgeBrain Score"]),
                        "Added": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.tracker.append(entry)
                st.success("Pinned Top 3 to Tracker ✅")

            # Race Tape
            with st.expander("📼 Race Tape — EV heat‑ribbons & odds trend"):
                st.caption("Green = positive EV; red = negative EV. Time axis uses 5s bins for clarity.")
                heat = chart_ev_heatstrip(rd)
                if heat: st.altair_chart(heat, use_container_width=True)
                trend = chart_odds_trend(rd)
                if trend: st.altair_chart(trend, use_container_width=True)

            # Explainability
            st.markdown("### Explain")
            for _, r in top3.iterrows():
                with st.expander(f"{r['Horse']} — Win {r['Win p %']:.1f}% | Odds {r['Odds']:.2f} | EB {r['EdgeBrain Score']:.1f} | EV {r['EV']:+.2f}"):
                    k = f"{r['MarketId']}:{r['SelectionId']}"
                    s = _sparkline_series(k, 60)
                    st.altair_chart(sparkline_chart(s, current_ev=float(r["EV"])), use_container_width=True)
                    expl = explain_row(r)
                    e1,e2,e3,e4 = st.columns(4)
                    e1.metric("Fair p", f"{expl['Fair p (base)']*100:.1f}%")
                    e2.metric("Momentum", f"{expl['Momentum']:+.2f}")
                    e3.metric("Pressure", f"{expl['Pressure']:+.2f}")
                    e4.metric("SP Δ%", f"{expl['SP Δ%']:+.1f}%")
                    f1,f2,f3,f4 = st.columns(4)
                    f1.metric("Liquidity", f"{expl['Liquidity']:.0f}")
                    f2.metric("Spread%", f"{expl['Spread%']:.1f}%")
                    f3.metric("Vol%", f"{expl['Vol%']:.1f}%")
                    f4.metric("Value Δ", f"{expl['Value Δ(pp)']:+.1f} pp")

elif sel == "EdgeBrain":
    st.header("🧠 EdgeBrain – Model View")
    st.caption(
        "Weights — Base:{w_base:.2f}, Mom:{w_mom:.2f}, Press:{w_press:.2f}, SPΔ:{w_sp:.2f}, "
        "Value:{w_value:.2f}, Liq:{w_liq:.2f}, Spread:{w_spread:.2f}, Vol:{w_vol:.2f}"
        .format(**st.session_state.ebw)
    )
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    df = get_df(day)
    if df.empty:
        st.info("No data available.")
    else:
        _toolbar(df, f"edge_{day}")
        st.dataframe(
            df[[
                "Course","Time","Horse","Odds","Fair Odds","EV","EW EV","Kelly %","Stake £",
                "EdgeBrain Score","Momentum","Pressure","SP (proj)","Base p %","Spread %","Vol %","LiqScore","Overround",
                "Back Size","Lay Size","Market Status","Last Seen (UTC)","Source"
            ]].sort_values(["EdgeBrain Score","EV"], ascending=[False, False]),
            use_container_width=True, column_config=_format_cols(df), height=560
        )
        st.markdown("---")
        with st.expander("🔎 Odds Ladder Inspector"):
            mid = st.text_input("MarketId")
            selid = st.number_input("SelectionId", min_value=1, step=1)
            @st.cache_data(ttl=10, show_spinner=False)
            def bf_ladder(market_id: str):
                token = bf_get_session_token()
                if not token: return {}
                params = {"marketIds":[market_id], "priceProjection":{"priceData":["EX_ALL_OFFERS"], "virtualise":True}}
                try: res = bf_call("listMarketBook", params, token); return res[0] if res else {}
                except Exception: return {}
            if st.button("Fetch ladder") and mid:
                book = bf_ladder(mid); ladder = {}
                for r in (book.get("runners") or []):
                    if int(r.get("selectionId", -1)) == int(selid):
                        ex = r.get("ex", {}) or {}
                        atb = ex.get("availableToBack", []) or []
                        atl = ex.get("availableToLay", []) or []
                        ladder = {"Back": [(lvl["price"], lvl["size"]) for lvl in atb[:25]],
                                  "Lay":  [(lvl["price"], lvl["size"]) for lvl in atl[:25]],}
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

elif sel == "Bet Tracker":
    st.header("🔖 Bet Tracker")
    st.caption("Add picks from any page: select rows and click **Add to Tracker** (see below).")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True, key="trk_day")
    df = get_df(day)
    if not df.empty:
        _toolbar(df, f"track_{day}")
        st.write("Select picks to add:")
        to_show = df[["Course","Time","Horse","Odds","Fair p","Fair Odds","EV","Kelly %","Stake £","EdgeBrain Score"]].copy()
        chosen = st.multiselect("Rows", (to_show["Horse"] + " | " + to_show["Course"] + " " + to_show["Time"]).tolist())
        stake_default = float(np.clip(st.session_state.bankroll * 0.01, 1.0, 100.0))
        stake = st.number_input("Stake per selection £", min_value=0.5, step=0.5, value=stake_default)
        if st.button("➕ Add to Tracker") and chosen:
            for lbl in chosen:
                r = to_show[(to_show["Horse"] + " | " + to_show["Course"] + " " + to_show["Time"]) == lbl].iloc[0].to_dict()
                r["Stake £"] = stake; r["Added"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.tracker.append(r)
            st.success(f"Added {len(chosen)} selections.")
    if st.session_state.tracker:
        tdf = pd.DataFrame(st.session_state.tracker)
        tdf["Exp £"] = (tdf["EV"] * tdf["Stake £"]).round(2)
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Bets", len(tdf))
        with c2: st.metric("Staked", f"£{tdf['Stake £'].sum():.2f}")
        with c3: st.metric("Expected P/L", f"£{tdf['Exp £'].sum():.2f}")
        st.dataframe(tdf, use_container_width=True, column_config=_format_cols(tdf))
        if st.button("🗑️ Clear Tracker"):
            st.session_state.tracker = []; st.experimental_rerun()
    else:
        st.info("No tracked bets yet.")

else:
    st.header("🐞 Debug")
    token = bf_get_session_token()
    st.write("Has Betfair token:", bool(token))
    cat = bf_list_win_markets("today", countries=st.session_state.get("market_scope", ["GB"]))
    st.write("Catalogue markets:", len(cat))
    if cat: st.json(cat[0])

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
