# app.py
import os
import re
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from datetime import datetime
from streamlit_option_menu import option_menu

# ---------- CONFIG ----------
st.set_page_config(page_title="EdgeBet – Live Racing (Basic API + Odds)", layout="wide")

# Try to load creds from secrets; fall back to env constants
USERNAME = st.secrets.get("racing_api", {}).get("username", os.getenv("RACING_API_USERNAME", "YOUR_USERNAME_HERE"))
PASSWORD = st.secrets.get("racing_api", {}).get("password", os.getenv("RACING_API_PASSWORD", "YOUR_PASSWORD_HERE"))

# ---------- THEME ----------
st.markdown("""
<style>
body { background-color: #111111; color: #f2f2f2; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.block-container { padding-top: 1.5rem; }
[data-testid="stMetricValue"] { color: #00ffcc !important; }
</style>
""", unsafe_allow_html=True)

# ---------- UTILITIES ----------
def normalize_text(s: str) -> str:
    """Uppercase, strip accents/punct/extra spaces for fuzzy matching."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()

def fractional_to_decimal(txt: str) -> float:
    """
    Convert fractional odds like '4/1', '7-2', 'Evs', 'EVS' to decimal.
    Returns np.nan if it cannot parse.
    """
    if not isinstance(txt, str):
        return np.nan
    t = txt.strip().upper().replace(" ", "")
    if t in ("EVS", "EVENS"):
        return 2.0
    # 7/2 or 7-2 style
    m = re.match(r"^(\d+)\s*[/\-]\s*(\d+)$", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return np.nan
        return round(a / b + 1.0, 2)
    # already decimal?
    try:
        return float(t)
    except Exception:
        return np.nan

# ---------- API: list racecards (BASIC) ----------
@st.cache_data(ttl=120)
def fetch_racecards(day: str = "today"):
    """
    BASIC endpoint:
      GET https://api.theracingapi.com/v1/racecards?day=today|tomorrow
    Returns (ok, json, status_code, text) so we can debug easily in the UI.
    """
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"day": day.lower()}
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), params=params, timeout=12)
        status = resp.status_code
        text = resp.text
        resp.raise_for_status()
        return True, resp.json(), status, text
    except requests.HTTPError as e:
        return False, None, getattr(e.response, "status_code", None), getattr(e.response, "text", str(e))
    except Exception as e:
        return False, None, None, str(e)

def map_region_to_country(region: str) -> str:
    if not isinstance(region, str):
        return "Unknown"
    r = region.upper()
    if r == "GB":
        return "UK"
    if r in ("US", "USA"):
        return "USA"
    if r == "IE":
        return "Ireland"
    return r

def build_live_df_from_api(payload: dict) -> pd.DataFrame:
    """
    Convert BASIC racecards JSON into runners table.
    BASIC does not include odds.
    """
    if not payload or "racecards" not in payload:
        return pd.DataFrame()

    rows = []
    for race in payload.get("racecards", []):
        region = race.get("region")
        course = race.get("course")
        off    = race.get("off_time")  # "HH:MM"
        for rnr in race.get("runners", []):
            horse = rnr.get("horse")
            if not horse:
                continue
            rows.append({
                "Country": map_region_to_country(region),
                "Course": course,
                "Time": off,
                "Horse": horse,
            })

    df = pd.DataFrame(rows)
    return df

# ---------- Racing Post SP odds scrape (time-order) ----------
@st.cache_data(ttl=90)
def fetch_rp_odds(day: str = "today") -> pd.DataFrame:
    """
    Scrape Racing Post time-order page to get runner SP odds.
    Returns dataframe with Course, Time, Horse, Odds (decimal if parsed).
    """
    # RP uses local date on path; we try both 'schedule' and plain time-order
    today_iso = datetime.utcnow().date().isoformat()
    url_candidates = [
        f"https://www.racingpost.com/racecards/time-order/{today_iso}",
        "https://www.racingpost.com/racecards/time-order/",
    ]
    headers = {"User-Agent": "EdgeBet/1.0"}
    out_rows = []

    for url in url_candidates:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            # Loop through race containers
            sections = soup.select("section.race-time") or soup.select("section")
            for sec in sections:
                time_el = sec.select_one(".rc-race-time")
                course_el = sec.select_one(".rc-race-meeting__course a")
                if not time_el or not course_el:
                    continue
                off = time_el.get_text(strip=True)
                course = course_el.get_text(strip=True)

                # runners list
                runners = sec.select(".runner-wrap") or sec.select("[data-test-selector='runner']")
                for runner in runners:
                    name_el = runner.select_one(".runner-runner__name") or runner.select_one("[data-test-selector='runner-name']")
                    sp_el   = runner.select_one(".runner-sp__price") or runner.select_one("[data-test-selector='runner-odds']")
                    if not name_el or not sp_el:
                        continue
                    horse = name_el.get_text(strip=True)
                    sp_txt = sp_el.get_text(strip=True)
                    dec = fractional_to_decimal(sp_txt)
                    out_rows.append({
                        "Course": course,
                        "Time": off[:5] if isinstance(off, str) else off,
                        "Horse": horse,
                        "Odds": dec,            # may be NaN if parse failed
                        "Odds_Source": sp_txt,  # raw text
                    })
            if out_rows:
                break
        except Exception:
            continue

    return pd.DataFrame(out_rows)

def compute_value_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # If Odds is missing, keep NaN; we'll fill later if needed
    d["Predicted Win %"] = (100.0 / pd.to_numeric(d["Odds"], errors="coerce")).round(1)
    d.loc[~np.isfinite(d["Predicted Win %"]), "Predicted Win %"] = np.nan

    # Simple placeholder value signals while we wire live books
    # If Odds exists -> value noise around 10–22; else default 12–20 so we still rank
    rng = np.random.default_rng(42)
    d["Win_Value"] = np.where(
        d["Odds"].notna(), rng.uniform(10, 22, len(d)), rng.uniform(12, 20, len(d))
    ).round(1)
    d["Place_Value"] = (d["Win_Value"] * 0.6).round(1)

    # If no odds, fallback predicted win% from a notional 5.0 decimal
    d["Predicted Win %"] = d["Predicted Win %"].fillna(round(100.0 / 5.0, 1))
    d["Predicted Place %"] = (d["Predicted Win %"] * 0.6).round(1)

    d["BetEdge Win %"] = ((d["Predicted Win %"] * 0.6) + (d["Win_Value"] * 0.4)).round(1)
    d["BetEdge Place %"] = ((d["Predicted Place %"] * 0.6) + (d["Place_Value"] * 0.4)).round(1)
    return d

def merge_live_with_odds(api_df: pd.DataFrame, rp_df: pd.DataFrame, use_mock_if_missing: bool = True) -> pd.DataFrame:
    """
    Left-join API runners to RP odds by normalized Course/Time/Horse.
    If RP has no odds, optionally fill with mock odds.
    """
    if api_df.empty:
        return api_df

    a = api_df.copy()
    a["Course_key"] = a["Course"].astype(str).map(normalize_text)
    a["Horse_key"]  = a["Horse"].astype(str).map(normalize_text)
    a["Time_key"]   = a["Time"].astype(str).str[:5]

    if rp_df.empty:
        out = a.copy()
        if use_mock_if_missing:
            np.random.seed(42)
            out["Odds"] = np.random.uniform(2, 10, len(out)).round(2)
            out["Source"] = "LIVE (names/times) + Mock Odds"
        else:
            out["Odds"] = np.nan
            out["Source"] = "LIVE (names/times) – Odds missing"
        out = compute_value_cols(out)
        return out.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

    b = rp_df.copy()
    b["Course_key"] = b["Course"].astype(str).map(normalize_text)
    b["Horse_key"]  = b["Horse"].astype(str).map(normalize_text)
    b["Time_key"]   = b["Time"].astype(str).str[:5]

    merged = pd.merge(
        a, b[["Course_key","Time_key","Horse_key","Odds","Odds_Source"]],
        on=["Course_key","Time_key","Horse_key"], how="left"
    )

    # Fill odds if missing
    if use_mock_if_missing:
        need = merged["Odds"].isna()
        np.random.seed(42)
        merged.loc[need, "Odds"] = np.random.uniform(2, 10, need.sum()).round(2)
        merged.loc[need, "Odds_Source"] = "Mock"
        merged["Source"] = np.where(need, "LIVE + Mock Odds", "LIVE + RP SP Odds")
    else:
        merged["Source"] = np.where(merged["Odds"].notna(), "LIVE + RP SP Odds", "LIVE (no odds)")

    merged = compute_value_cols(merged)
    keep_cols = [
        "Country","Course","Time","Horse","Odds","Odds_Source",
        "Predicted Win %","Predicted Place %","Win_Value","Place_Value",
        "BetEdge Win %","BetEdge Place %","Source"
    ]
    return merged[keep_cols].sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# ---------- EdgeBrain (stub until model is present) ----------
def edgebrain_score_stub(df: pd.DataFrame) -> pd.Series:
    return (0.5 * df["BetEdge Win %"] + 0.5 * df["Predicted Win %"]).round(1)

# ---------- SIDEBAR ----------
with st.sidebar:
    sel = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Debug"],
        icons=["house", "activity", "robot", "bug"],
        default_index=0
    )

# ---------- OVERVIEW ----------
if sel == "Overview":
    st.title("📊 EdgeBet – Live Racing (Basic API + RP Odds)")
    ok, data, status, text = fetch_racecards("today")
    if not ok:
        st.error(f"API error (today): HTTP {status}\n{text[:300]}")
        st.info("Basic endpoint is GET /v1/racecards?day=today|tomorrow (no 'date').")
        st.stop()

    base_df = build_live_df_from_api(data)
    rp_df = fetch_rp_odds("today")
    live_df = merge_live_with_odds(base_df, rp_df, use_mock_if_missing=True)

    st.metric("Meetings (today)", len({(rc.get('region'), rc.get('course')) for rc in data.get('racecards', [])}))
    st.metric("Races (today)", len(data.get("racecards", [])))
    st.metric("Runners (today)", len(live_df))

    if live_df.empty:
        st.warning("No runners returned by the API for today. Try 'Horse Racing' tab and switch to tomorrow.")
    else:
        st.subheader("Top 20 BetEdge (today)")
        st.bar_chart(live_df.head(20).set_index("Horse")["BetEdge Win %"])

# ---------- HORSE RACING ----------
elif sel == "Horse Racing":
    st.title("🏇 Horse Racing — Live (API + RP Odds)")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    odds_mode = st.radio(
        "Odds Mode",
        ["Try RP Odds (fallback to mock)", "Mock Odds only"],
        horizontal=True
    )

    ok, data, status, text = fetch_racecards(day)
    if not ok:
        st.error(f"API error ({day}): HTTP {status}\n{text[:300]}")
        st.stop()

    api_df = build_live_df_from_api(data)
    if api_df.empty:
        st.warning("No races from API. Try switching the day.")
        st.stop()

    if odds_mode.startswith("Try RP"):
        rp_df = fetch_rp_odds(day)
        df = merge_live_with_odds(api_df, rp_df, use_mock_if_missing=True)
    else:
        df = merge_live_with_odds(api_df, pd.DataFrame(), use_mock_if_missing=True)

    # Filters & controls
    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country", ["All"] + sorted(df["Country"].dropna().unique().tolist()))
    with col2:
        bookie = st.selectbox("Bookmaker", ["All", "SkyBet", "Bet365", "Betfair"])
    with col3:
        courses = sorted(df["Course"].dropna().unique().tolist())
        course_filter = st.multiselect("Courses", courses, default=courses)

    # Apply filters
    filt = df.copy()
    if country != "All":
        filt = filt[filt["Country"] == country]
    if course_filter:
        filt = filt[filt["Course"].isin(course_filter)]

    # Sort & view mode
    min_val = int(filt["BetEdge Win %"].min()) if not filt.empty else 0
    max_val = int(filt["BetEdge Win %"].max()) if not filt.empty else 0
    edge_range = st.slider("🎯 Filter by BetEdge Win %", min_val, max_val, (min_val, max_val))
    filt = filt[filt["BetEdge Win %"].between(*edge_range)]
    view = st.radio("View", ["📊 Charts", "📋 Tables"], horizontal=True)

    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        if view == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(filt.sort_values("BetEdge Win %", ascending=False)
                             .head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(filt.sort_values("BetEdge Place %", ascending=False)
                             .head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("Full Runner Table")
            st.dataframe(
                filt[["Country","Course","Time","Horse","Odds","Odds_Source","BetEdge Win %","BetEdge Place %","Source"]],
                use_container_width=True
            )

# ---------- EDGEBRAIN ----------
elif sel == "EdgeBrain":
    st.title("🧠 EdgeBrain — Live Names + Odds Merge")
    ok, data, status, text = fetch_racecards("today")
    if not ok:
        st.error(f"API error (today): HTTP {status}\n{text[:300]}")
        st.stop()
    base_df = build_live_df_from_api(data)
    rp_df = fetch_rp_odds("today")
    df = merge_live_with_odds(base_df, rp_df, use_mock_if_missing=True)
    if df.empty:
        st.warning("No runners returned by API today.")
    else:
        df["EdgeBrain Win %"] = edgebrain_score_stub(df)
        st.subheader("Top 20 EdgeBrain Win %")
        st.bar_chart(df.sort_values("EdgeBrain Win %", ascending=False)
                       .head(20).set_index("Horse")["EdgeBrain Win %"])
        st.dataframe(
            df[["Country","Course","Time","Horse","Odds","Odds_Source","EdgeBrain Win %","BetEdge Win %","Source"]],
            use_container_width=True
        )

# ---------- DEBUG ----------
else:
    st.title("🐞 API / Odds Debug")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    ok, data, status, text = fetch_racecards(day)
    st.write("Request:", "`GET /v1/racecards`", "params:", {"day": day})
    st.write("HTTP Status:", status)
    if ok:
        st.success("API OK")
        st.write("Top-level keys:", list(data.keys()))
        st.write("racecards count:", len(data.get("racecards", [])))
        sample = data.get("racecards", [])[:2]
        st.json(sample)
        api_df = build_live_df_from_api(data)
        st.write("API DataFrame shape:", api_df.shape)
        st.dataframe(api_df.head(20), use_container_width=True)

        st.subheader("Racing Post odds scrape")
        rp_df = fetch_rp_odds(day)
        st.write("RP DataFrame shape:", rp_df.shape)
        st.dataframe(rp_df.head(20), use_container_width=True)

        merged = merge_live_with_odds(api_df, rp_df, use_mock_if_missing=True)
        st.subheader("Merged sample")
        st.dataframe(merged.head(20), use_container_width=True)
    else:
        st.error("API error:")
        st.code((text or "")[:1000])

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
