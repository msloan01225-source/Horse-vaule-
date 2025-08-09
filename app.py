import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Optional
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="EdgeBet", layout="wide")

# ---------------------------
# Controls (sidebar)
# ---------------------------
with st.sidebar:
    st.title("⚙️ Live Controls")
    src_pref = st.selectbox(
        "Source preference",
        ["API → RP fallback", "API only", "RP only"],
        index=0,
        help="Use TheRacingAPI basic endpoint first, then Racing Post if needed; or force one source."
    )
    day = st.selectbox("Day", ["today", "tomorrow"], index=0)
    debug = st.toggle("Debug mode", value=False)
    if st.button("Force refresh (clear cache)"):
        st.cache_data.clear()
        st.rerun()

st.title("🏇 EdgeBet — Live Racing Tracker")

# ---------------------------
# Helpers
# ---------------------------
def _api_creds():
    user = st.secrets.get("RACING_API_USER", "")
    pw   = st.secrets.get("RACING_API_PASS", "")
    return user, pw

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# ---------------------------
# Live: TheRacingAPI BASIC
# ---------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_api_basic(day_str, user, pw):
    url = "https://api.theracingapi.com/v1/racecards/basic"
    params = {"day": day_str.lower()}  # "today" or "tomorrow"
    headers = {"User-Agent": "EdgeBet/1.0"}
    try:
        r = requests.get(url, params=params, auth=HTTPBasicAuth(user, pw), headers=headers, timeout=12)
        status = f"{r.status_code}"
        text = r.text[:500]
        r.raise_for_status()
        data = r.json()
        # Parse flexibly
        rows = []
        meetings = data.get("meetings") or data.get("racecards") or []
        for m in meetings:
            # course
            course = None
            if isinstance(m.get("course"), dict):
                course = m["course"].get("name") or m["course"].get("id")
            if not course:
                course = m.get("course") or m.get("course_name") or m.get("meeting") or "Unknown"
            races = m.get("races", [])
            for race in races:
                off = race.get("off") or race.get("time") or race.get("race_time") or ""
                if isinstance(off, str):
                    off = off[:5]
                runners = race.get("runners") or race.get("horses") or []
                for rnr in runners:
                    horse = rnr.get("horse")
                    if isinstance(horse, dict):
                        horse = horse.get("name") or horse.get("id")
                    if not horse:
                        horse = rnr.get("name") or rnr.get("runner") or "Unknown"
                    odds = (
                        rnr.get("sp_dec") or rnr.get("spDec") or
                        rnr.get("spDecimal") or rnr.get("oddsDecimal") or
                        rnr.get("decimal_odds")
                    )
                    rows.append({
                        "Horse": str(horse),
                        "Course": str(course),
                        "Time": off,
                        "Odds": _safe_float(odds),
                    })
        df = pd.DataFrame(rows)
        return df, {"ok": True, "status": status, "url": r.url, "sample": meetings[:1]}
    except requests.HTTPError as e:
        return pd.DataFrame(), {"ok": False, "status": f"HTTP {e.response.status_code}", "url": e.response.url, "body": e.response.text[:500]}
    except Exception as e:
        return pd.DataFrame(), {"ok": False, "status": "EXC", "url": url, "body": str(e)}

# ---------------------------
# Live: Racing Post scrape
# ---------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_rp(day_str):
    target = datetime.utcnow().date()
    if day_str.lower() == "tomorrow":
        target += timedelta(days=1)
    urls = [
        f"https://www.racingpost.com/racecards/time-order/{target.isoformat()}",
        "https://www.racingpost.com/racecards/time-order/",
    ]
    headers = {"User-Agent": "EdgeBet/1.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            status = r.status_code
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            rows = []
            # sections for each race
            sections = soup.select("section.race-time") or soup.select("section")
            for sec in sections:
                time_el = sec.select_one(".rc-race-time")
                course_el = sec.select_one(".rc-race-meeting__course a")
                if not time_el or not course_el:
                    continue
                off = time_el.get_text(strip=True)
                course = course_el.get_text(strip=True)
                runners = sec.select(".runner-wrap") or sec.select("[data-test-selector='runner']")
                for runner in runners:
                    name_el = runner.select_one(".runner-runner__name") or runner.select_one("[data-test-selector='runner-name']")
                    sp_el   = runner.select_one(".runner-sp__price") or runner.select_one("[data-test-selector='runner-odds']")
                    if not name_el or not sp_el:
                        continue
                    horse = name_el.get_text(strip=True)
                    odds  = _safe_float(sp_el.get_text(strip=True).replace(" ", ""))
                    rows.append({"Horse": horse, "Course": course, "Time": off, "Odds": odds})
            if rows:
                return pd.DataFrame(rows), {"ok": True, "status": status, "url": url, "sample": rows[:3]}
        except Exception as e:
            last_err = {"ok": False, "status": "EXC", "url": url, "body": str(e)}
    return pd.DataFrame(), last_err if "last_err" in locals() else {"ok": False, "status": "NO_DATA", "url": urls[-1], "body": "No runners parsed"}

# ---------------------------
# Enrich & charts
# ---------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["Odds"] = pd.to_numeric(d["Odds"], errors="coerce")
    d["Implied %"] = (100.0 / d["Odds"].replace(0, np.nan)).round(1)
    # provisional value signals (replace with your model later)
    d["Win_Value"] = np.random.uniform(6, 24, len(d)).round(1)
    d["Place_Value"] = (d["Win_Value"] * 0.6).round(1)
    d["Predicted Win %"] = d["Implied %"]
    d["Predicted Place %"] = (d["Predicted Win %"] * 0.6).round(1)
    d["BetEdge Win %"] = ((d["Predicted Win %"] * 0.6) + (d["Win_Value"] * 0.4)).round(1)
    d["BetEdge Place %"] = ((d["Predicted Place %"] * 0.6) + (d["Place_Value"] * 0.4)).round(1)
    d["Edge Δ %"] = (d["BetEdge Win %"] - d["Implied %"]).round(1)
    d["Risk"] = np.where(d["BetEdge Win %"] > 25, "✅", np.where(d["BetEdge Win %"] > 15, "⚠️", "❌"))
    # basic UK/USA tagging for visual filtering
    uk_courses = {"Ascot","York","Cheltenham","Newmarket","Goodwood","Aintree","Epsom","Doncaster","Sandown"}
    d["Country"] = d["Course"].astype(str).apply(lambda c: "UK" if c in uk_courses else "USA")
    return d

def plot_value_vs_odds(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("No data to chart.")
        return
    d = df.sort_values("BetEdge Win %", ascending=False).head(30)
    if not PLOTLY_OK:
        st.bar_chart(d.set_index("Horse")["BetEdge Win %"])
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=d["Horse"], y=d["BetEdge Win %"], name="BetEdge Win %",
        hovertemplate="<b>%{x}</b><br>BetEdge: %{y:.1f}%<br>Implied: %{customdata[0]:.1f}%<br>Odds: %{customdata[1]:.2f}<br>Value: %{customdata[2]:.1f}",
        customdata=np.stack([d["Implied %"], d["Odds"], d["Win_Value"]], axis=-1),
    )
    fig.add_scatter(
        x=d["Horse"], y=d["Odds"], name="Odds (dec)", mode="lines+markers", yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Odds: %{y:.2f}",
    )
    fig.update_layout(title=title, height=470, legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(title_text="BetEdge Win %", secondary_y=False)
    fig.update_yaxes(title_text="Odds (dec)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Build live according to preference
# ---------------------------
def build_live(day_str, src_pref):
    dbg = {}
    if src_pref in ("API → RP fallback", "API only"):
        user, pw = _api_creds()
        df_api, info = fetch_api_basic(day_str, user, pw)
        dbg["api"] = info
        if not df_api.empty:
            return enrich(df_api), "TheRacingAPI", dbg
        if src_pref == "API only":
            return pd.DataFrame(), "API only (no data)", dbg
    if src_pref in ("API → RP fallback", "RP only"):
        df_rp, info = fetch_rp(day_str)
        dbg["rp"] = info
        if not df_rp.empty:
            return enrich(df_rp), "Racing Post", dbg
        if src_pref == "RP only":
            return pd.DataFrame(), "RP only (no data)", dbg
    return pd.DataFrame(), "No source returned data", dbg

# ---------------------------
# Load live
# ---------------------------
df, source, debug_info = build_live(day, src_pref)

if debug:
    st.subheader("🔎 Debug")
    st.write("Source preference:", src_pref, "| Day:", day)
    st.json(debug_info)

if df.empty:
    st.error("No live runners returned. Check secrets (API) or try switching source preference.")
    st.stop()

# ---------------------------
# UI
# ---------------------------
st.caption(f"Source: {source}")
c1, c2, c3 = st.columns(3)
c1.metric("Meetings", int(df["Course"].nunique()))
c2.metric("Total Runners", len(df))
c3.metric("Top BetEdge", f"{pd.to_numeric(df['BetEdge Win %'], errors='coerce').max():.1f}%")

# Filters
fc1, fc2, fc3 = st.columns(3)
with fc1:
    country = st.selectbox("Country", ["All","UK","USA"], index=0)
with fc2:
    courses = sorted(df["Course"].dropna().unique().tolist())
    course_filter = st.multiselect("Courses", courses, default=courses)
with fc3:
    be = pd.to_numeric(df["BetEdge Win %"], errors="coerce")
    minv = int(np.nanmin(be)) if be.notna().any() else 0
    maxv = int(np.nanmax(be)) if be.notna().any() else 100
    edge_range = st.slider("BetEdge Win % range", minv, maxv, (minv, maxv))

f = df.copy()
if country != "All":
    f = f[f["Country"] == country]
f = f[f["Course"].isin(course_filter)]
f = f[pd.to_numeric(f["BetEdge Win %"], errors="coerce").between(edge_range[0], edge_range[1])]

tab1, tab2 = st.tabs(["Charts", "Table"])

with tab1:
    topn = st.slider("Chart Top N", 10, 60, 20, step=5)
    plot_value_vs_odds(f, "Value vs Odds (Top by BetEdge)")

with tab2:
    cols = ["Horse","Course","Time","Odds","Implied %","Win_Value","BetEdge Win %","Edge Δ %","Risk"]
    st.dataframe(f.sort_values("BetEdge Win %", ascending=False)[cols], use_container_width=True)

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
