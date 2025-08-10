# edgebet_app.py  — Streamlit UI (front-end only) that talks to the FastAPI backend (Point 7)
# ==========================================================================================
# This version removes all direct Betfair calls and uses your backend endpoints instead:
#   GET  /health, /status, /config, /presets
#   GET  /markets?day=today|tomorrow
#   GET  /races?day=...
#   GET  /race?course=...&time_hhmm=...&day=...
#   GET  /race/top3?course=...&time_hhmm=...&day=...
#   GET  /race/history?course=...&time_hhmm=...&day=...
#   GET  /alerts?day=...
#
# Configure your backend URL in Streamlit Secrets (recommended):
#   [api]
#   base = "https://YOUR-EDGEBRAIN-API-URL"
#
# Or via environment variable: EDGEBRAIN_API_BASE

import os
import json
import time
import hashlib
import random
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from streamlit_option_menu import option_menu

# Optional plotting – we try Altair; if unavailable we fallback to Streamlit charts
try:
    import altair as alt
    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="EdgeBet – Live Racing", layout="wide", page_icon="🏇")

# ------------------ API BASE ------------------
API_BASE = st.secrets.get("api", {}).get("base", os.getenv("EDGEBRAIN_API_BASE", "")).rstrip("/")
if not API_BASE:
    st.warning("API base URL not set. Add [api].base in Streamlit Secrets or EDGEBRAIN_API_BASE env var.", icon="⚠️")

# ------------------ THEME TOGGLE ------------------
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

def toggle_theme():
    st.session_state["theme_mode"] = "light" if st.session_state["theme_mode"] == "dark" else "dark"

# ------------------ HEADER ------------------
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("🏇 EdgeBet – Live Racing")
with col2:
    st.button("🌗 Toggle Theme", on_click=toggle_theme, use_container_width=True)

# ------------------ MATRIX THEME CSS ------------------
def apply_theme():
    is_dark = st.session_state["theme_mode"] == "dark"
    # Matrixy palette
    neon_green = "#00FF6A"
    purple = "#7C3AED"
    red = "#EF4444"
    if is_dark:
        bg = "#0A0F0A"      # near-black with green tint
        sbg = "#0F1C16"     # panel background
        text = "#EDEDED"
    else:
        bg = "#FFFFFF"
        sbg = "#F2F2F2"
        text = "#111111"

    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}
    section[data-testid="stSidebar"] {{ background-color: {sbg} !important; }}
    h1, h2, h3, h4, h5, h6 {{
        color: {neon_green};
        text-shadow: 0px 0px 8px rgba(0, 255, 106, 0.25);
    }}
    [data-testid="stMetricLabel"] {{ color: {text} !important; }}
    [data-testid="stMetricValue"] {{ color: {neon_green} !important; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background-color: {neon_green}; border-radius: 4px; }}
    ::-webkit-scrollbar-track {{ background-color: {bg}; }}
    .eb-badge {{
        display:inline-block; padding:2px 6px; border-radius:6px; font-weight:600; font-size:12px;
        color:#0A0F0A; background:{neon_green};
    }}
    .ev-badge-pos {{ color:#0A0F0A; background:{neon_green}; padding:2px 6px; border-radius:6px; font-weight:600; }}
    .ev-badge-neg {{ color:#FFF; background:{red}; padding:2px 6px; border-radius:6px; font-weight:600; }}
    .card {{
        background:{sbg}; border:1px solid rgba(0,255,106,0.1); border-radius:10px; padding:12px;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ------------------ HELPERS ------------------
def _get_json(path: str, params: dict | None = None, timeout: int = 12):
    if not API_BASE:
        return None, "API base URL not set"
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        # pass through status info for diagnostics
        if not r.ok:
            return None, f"HTTP {r.status_code}: {(r.text or '')[:200]}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")

def _traffic_light(val: float, lo: float, hi: float) -> str:
    """Return hex color from green (best) → red (worst)."""
    if hi <= lo:
        return "#16A34A"  # default green
    t = (val - lo) / (hi - lo + 1e-9)
    # green (#16A34A) → lime (#84CC16) → amber (#F59E0B) → red (#EF4444)
    if t < 0.33:  return "#16A34A"
    if t < 0.66:  return "#84CC16"
    if t < 0.85:  return "#F59E0B"
    return "#EF4444"

def _style_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Color EdgeBrain Score (green best→red worst), EV (green pos / red neg)."""
    def color_eb(s: pd.Series):
        lo, hi = float(s.min()), float(s.max())
        return [f"background-color: {_traffic_light(v, lo, hi)}; color: #0A0F0A; font-weight:700"
                for v in s]
    def color_ev(s: pd.Series):
        out = []
        for v in s:
            if pd.isna(v):
                out.append("")
            else:
                out.append("background-color:#16A34A; color:#0A0F0A; font-weight:700" if v >= 0
                           else "background-color:#EF4444; color:#FFF; font-weight:700")
        return out
    styler = df.style
    if "EdgeBrain Score" in df.columns:
        styler = styler.apply(color_eb, subset=["EdgeBrain Score"])
    if "EV" in df.columns:
        styler = styler.apply(color_ev, subset=["EV"])
    return styler.format(precision=2)

def _safe_df(rows) -> pd.DataFrame:
    try:
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ------------------ BACKEND CLIENT (CACHED) ------------------
@st.cache_data(ttl=30, show_spinner=False)
def api_config():
    return _get_json("/config")[0] or {}

@st.cache_data(ttl=15, show_spinner=False)
def api_markets(day="today") -> tuple[pd.DataFrame, str | None]:
    data, err = _get_json("/markets", {"day": day})
    if err: return pd.DataFrame(), err
    return _safe_df(data), None

@st.cache_data(ttl=15, show_spinner=False)
def api_races(day="today") -> tuple[list[str], str | None]:
    data, err = _get_json("/races", {"day": day})
    if err: return [], err
    return data or [], None

@st.cache_data(ttl=15, show_spinner=False)
def api_race(course: str, time_hhmm: str, day="today") -> tuple[pd.DataFrame, str | None]:
    data, err = _get_json("/race", {"course": course, "time_hhmm": time_hhmm, "day": day})
    if err: return pd.DataFrame(), err
    return _safe_df(data), None

@st.cache_data(ttl=15, show_spinner=False)
def api_top3(course: str, time_hhmm: str, day="today") -> tuple[pd.DataFrame, str | None]:
    data, err = _get_json("/race/top3", {"course": course, "time_hhmm": time_hhmm, "day": day})
    if err: return pd.DataFrame(), err
    arr = (data or {}).get("top3", [])
    return _safe_df(arr), None

@st.cache_data(ttl=30, show_spinner=False)
def api_alerts(day="today"):
    return _get_json("/alerts", {"day": day})[0] or {"items": []}

@st.cache_data(ttl=20, show_spinner=False)
def api_health():
    ok, _ = _get_json("/health")
    return ok or {}

@st.cache_data(ttl=20, show_spinner=False)
def api_status():
    ok, _ = _get_json("/status")
    return ok or {}

# ------------------ SIDEBAR NAV ------------------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Debug"],
        icons=["house", "trophy", "robot", "wrench"],
        default_index=0
    )
    st.markdown("---")
    st.caption("Front-end connected to EdgeBrain API")

# ------------------ PAGES ------------------
if sel == "Overview":
    st.header("📊 Overview – Today's Racing")
    df_live, err = api_markets("today")
    if err:
        st.error(f"API error: {err}")

    races = int(df_live["Course"].nunique()) if not df_live.empty else 0
    runners = int(len(df_live)) if not df_live.empty else 0
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Races (today)", races)
    with c2: st.metric("Runners (today)", runners)
    with c3:
        cfg = api_config()
        st.metric("Alerts (EV/EB min)", f"{cfg.get('alerts',{}).get('ev_min',0.1):.2f} / {cfg.get('alerts',{}).get('eb_min',32)}")

    # Alerts preview
    alerts = api_alerts("today") or {}
    items = (alerts or {}).get("items", [])[:3]
    if items:
        with st.container():
            st.subheader("🔥 Latest Hot Picks")
            for a in items:
                badge = f"<span class='ev-badge-pos'>EV {a.get('ev',0):+.2f}</span>" if a.get("ev",0) >= 0 else f"<span class='ev-badge-neg'>EV {a.get('ev',0):+.2f}</span>"
                st.markdown(
                    f"<div class='card'>"
                    f"<b>{a.get('horse','')}</b> — {a.get('course','')} {a.get('time','')} &nbsp; "
                    f"Odds {a.get('odds',0):.2f} &nbsp; "
                    f"{badge} &nbsp; "
                    f"<span class='eb-badge'>EB {a.get('eb',0):.1f}</span>"
                    f"</div>", unsafe_allow_html=True
                )

    # Top 20 by EdgeBrain Score
    if not df_live.empty:
        st.subheader("Top EdgeBrain (Live)")
        top = df_live.sort_values("EdgeBrain Score", ascending=False).head(12).copy()
        top["Label"] = top["Horse"].astype(str) + " (" + top["Course"].astype(str) + " " + top["Time"].astype(str) + ")"
        if HAS_ALTAIR:
            maxv = float(top["EdgeBrain Score"].max())
            minv = float(top["EdgeBrain Score"].min())
            top["color"] = top["EdgeBrain Score"].apply(lambda v: _traffic_light(v, minv, maxv))
            chart = (
                alt.Chart(top)
                .mark_bar()
                .encode(
                    x=alt.X("Label:N", sort="-y", axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y("EdgeBrain Score:Q"),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=["Horse","Course","Time","Odds","Fair Odds","EdgeBrain Score","EV"]
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.bar_chart(top.set_index("Label")[["EdgeBrain Score"]])

elif sel == "Horse Racing":
    st.header("🏇 Horse Racing – Live")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)

    races, err = api_races(day)
    if err:
        st.error(f"API error: {err}")
        races = []

    if not races:
        st.info("No races available.")
    else:
        race = st.selectbox("Select Race", races, index=0)
        if race:
            course, tstr = race.split(" | ", 1)
            df_race, err_race = api_race(course, tstr, day)
            df_top3, err_top = api_top3(course, tstr, day)
            if err_race:
                st.error(f"Race error: {err_race}")
            if err_top:
                st.error(f"Top3 error: {err_top}")

            colA, colB = st.columns([0.6, 0.4])
            with colA:
                st.subheader(f"Runners — {course} {tstr}")
                if df_race.empty:
                    st.info("No runners found.")
                else:
                    # keep key columns readable first
                    cols = ["Horse","Odds","Fair Odds","EdgeBrain Score","EV","Best Lay","Back Size","Lay Size","SP (proj)"]
                    show = [c for c in cols if c in df_race.columns]
                    show = ["Horse"] + [c for c in show if c != "Horse"]
                    st.dataframe(_style_df(df_race[show]), use_container_width=True)

            with colB:
                st.subheader("Top 3 Predictor")
                if df_top3.empty:
                    st.info("No picks.")
                else:
                    for i, r in df_top3.reset_index(drop=True).iterrows():
                        ev_badge = f"<span class='ev-badge-pos'>EV {r['EV']:+.2f}</span>" if r["EV"] >= 0 else f"<span class='ev-badge-neg'>EV {r['EV']:+.2f}</span>"
                        st.markdown(
                            f"<div class='card'><b>{i+1}. {r['Horse']}</b><br>"
                            f"Odds {r['Odds']:.2f} &nbsp; Fair {r['Fair Odds']:.2f} &nbsp; "
                            f"Win {r['Win p %']:.1f}% &nbsp; "
                            f"{ev_badge} &nbsp; <span class='eb-badge'>EB {r['EdgeBrain Score']:.1f}</span></div>",
                            unsafe_allow_html=True
                        )

elif sel == "EdgeBrain":
    st.header("🧠 EdgeBrain — Full Board")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True, key="eb_day")
    df, err = api_markets(day)
    if err:
        st.error(f"API error: {err}")
    if df.empty:
        st.info("No data available.")
    else:
        # Filters
        min_eb = float(np.floor(df["EdgeBrain Score"].min())) if "EdgeBrain Score" in df.columns else 0.0
        max_eb = float(np.ceil(df["EdgeBrain Score"].max())) if "EdgeBrain Score" in df.columns else 100.0
        eb_range = st.slider("🎯 Filter by EdgeBrain Score", min_value=min_eb, max_value=max_eb, value=(min_eb, max_eb))
        df_f = df[(df["EdgeBrain Score"] >= eb_range[0]) & (df["EdgeBrain Score"] <= eb_range[1])].copy()

        # Country/Course filters if present
        if "Course" in df_f.columns:
            courses = sorted(df_f["Course"].dropna().unique().tolist())
            chosen = st.multiselect("Courses", courses, default=courses)
            if chosen:
                df_f = df_f[df_f["Course"].isin(chosen)]

        # Table
        cols = ["Course","Time","Horse","Odds","Fair Odds","EdgeBrain Score","EV"]
        show = [c for c in cols if c in df_f.columns]
        st.dataframe(_style_df(df_f[show].sort_values(["EdgeBrain Score","EV"], ascending=[False, False])), use_container_width=True, height=520)

else:
    st.header("🐞 Debug")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Health")
        st.json(api_health())
    with c2:
        st.subheader("Status")
        st.json(api_status())

    st.markdown("---")
    st.code(f"API_BASE = {API_BASE or '(not set)'}")
    st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
