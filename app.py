import os
import json
import math
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ------------------ CONFIG ------------------
st.set_page_config(page_title="EdgeBet – Live", layout="wide")

def _get_creds():
    u = ""
    p = ""
    try:
        u = st.secrets.get("racingapi", {}).get("username", "")
        p = st.secrets.get("racingapi", {}).get("password", "")
    except Exception:
        pass
    # env fallbacks (optional)
    u = u or os.getenv("RACING_API_USERNAME", "")
    p = p or os.getenv("RACING_API_PASSWORD", "")
    return u, p

API_USER, API_PASS = _get_creds()

# If no secrets, show inputs (so you can test quickly)
with st.sidebar:
    if not API_USER or not API_PASS:
        st.info("Add TheRacingAPI creds to .streamlit/secrets.toml\n(racingapi.username / racingapi.password)")
        API_USER = st.text_input("TheRacingAPI Username", type="password", key="api_user")
        API_PASS = st.text_input("TheRacingAPI Password", type="password", key="api_pass")

# ------------------ FETCH (BASIC) ------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_racecards_basic(day: str = "today"):
    """
    TheRacingAPI: Racecards (Basic)
    GET /v1/racecards/basic?day=today|tomorrow  (HTTP Basic Auth)
    """
    if day not in ("today", "tomorrow"):
        day = "today"

    url = "https://api.theracingapi.com/v1/racecards/basic"
    params = {"day": day}
    auth = HTTPBasicAuth(API_USER, API_PASS) if API_USER and API_PASS else None

    if not auth:
        # No credentials -> return an empty object so caller can fallback
        return {"_error": "missing_credentials"}

    try:
        r = requests.get(url, auth=auth, params=params, timeout=12)
        r.raise_for_status()
        return {"_ok": True, "_status": r.status_code, "_payload": r.json()}
    except requests.HTTPError as e:
        return {
            "_error": "http_error",
            "_status": getattr(e.response, "status_code", None),
            "_body": getattr(e.response, "text", ""),
        }
    except Exception as e:
        return {"_error": f"exception: {e.__class__.__name__}", "_msg": str(e)}

# ------------------ PARSE → DATAFRAME ------------------
def safe_get(obj, *keys, default=None):
    """Try several keys/paths, return first hit."""
    for k in keys:
        if isinstance(k, (list, tuple)):
            # path walk
            cur = obj
            ok = True
            for kk in k:
                if isinstance(cur, dict) and kk in cur:
                    cur = cur[kk]
                else:
                    ok = False
                    break
            if ok:
                return cur
        else:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
    return default

def parse_basic_payload(payload: dict) -> pd.DataFrame:
    """
    Accepts the JSON from /v1/racecards/basic and tries to extract:
    Course, Off time, Runner (Horse), Odds (if present).
    """
    rows = []

    # The top-level container may be one of these:
    meetings = (
        payload.get("meetings")
        or payload.get("racecards")
        or payload.get("items")
        or []
    )

    # If the api returns a bare list, accept that too
    if isinstance(meetings, dict):
        # sometimes under "data" or similar
        meetings = meetings.get("data", [])

    if not isinstance(meetings, list):
        meetings = []

    for m in meetings:
        course = (
            safe_get(m, ["course", "name"])
            or m.get("course_name")
            or m.get("venue")
            or m.get("meeting_name")
            or "Unknown"
        )
        races = m.get("races") or m.get("items") or []
        if isinstance(races, dict):
            races = races.get("data", [])
        if not isinstance(races, list):
            races = []

        for race in races:
            off = (
                race.get("off")
                or race.get("scheduled_off")
                or race.get("time")
                or race.get("off_time")
                or ""
            )
            # runners may be present on basic (names), odds may not be
            runners = race.get("runners") or race.get("entries") or []
            if isinstance(runners, dict):
                runners = runners.get("data", [])
            if not isinstance(runners, list):
                runners = []

            if runners:
                for rnr in runners:
                    horse = (
                        rnr.get("horse")
                        or rnr.get("horse_name")
                        or rnr.get("name")
                        or str(rnr.get("horse_id", "Unknown"))
                    )
                    # A few common places odds might appear (if available on your plan)
                    odds = (
                        rnr.get("sp_dec")
                        or rnr.get("odds_dec")
                        or safe_get(rnr, ["odds", "dec"])
                        or safe_get(rnr, ["prices", "decimal"])
                        or np.nan
                    )
                    try:
                        odds = float(odds)
                    except Exception:
                        odds = np.nan

                    rows.append(
                        {"Course": course, "Time": off, "Horse": horse, "Odds": odds}
                    )
            else:
                # No runners on Basic? keep at least one row for this race
                rows.append({"Course": course, "Time": off, "Horse": "", "Odds": np.nan})

    df = pd.DataFrame(rows)
    # Drop empty rows if nothing meaningful
    if df.empty:
        return df

    # If no horse names at all, drop those rows (just to keep table tidy)
    df = df[~(df["Horse"].astype(str).str.strip() == "")]
    return df.reset_index(drop=True)

# ------------------ ENRICH / BETEDGE ------------------
def enrich_with_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # If no Odds, leave NaN and later guard calculations
    # Create simple value scaffolding
    rng = np.random.default_rng(42)
    if "Win_Value" not in out.columns:
        out["Win_Value"] = rng.uniform(5, 30, len(out)).round(1)
    if "Place_Value" not in out.columns:
        out["Place_Value"] = (out["Win_Value"] * 0.6).round(1)

    # Predicted percentages from odds (when present)
    with np.errstate(divide="ignore", invalid="ignore"):
        pred_win = (1.0 / out["Odds"] * 100.0).astype(float)
        pred_win = pred_win.replace([np.inf, -np.inf], np.nan)
        out["Predicted Win %"] = np.where(np.isnan(pred_win), np.nan, pred_win).round(1)
    out["Predicted Place %"] = (out["Predicted Win %"] * 0.6).round(1)

    # Where Predicted Win % is NaN (no odds on Basic), seed a conservative baseline (optional)
    mask_nan = out["Predicted Win %"].isna()
    if mask_nan.any():
        out.loc[mask_nan, "Predicted Win %"] = 10.0  # baseline
        out.loc[mask_nan, "Predicted Place %"] = 6.0

    # BetEdge blend
    out["BetEdge Win %"] = (
        (out["Predicted Win %"] * 0.6) + (out["Win_Value"] * 0.4)
    ).round(1)
    out["BetEdge Place %"] = (
        (out["Predicted Place %"] * 0.6) + (out["Place_Value"] * 0.4)
    ).round(1)

    # Country inference (basic heuristic)
    out["Country"] = out["Course"].apply(
        lambda c: "UK" if c in ("Ascot", "York", "Cheltenham", "Sandown Park",
                                "Newmarket", "Goodwood", "Ayr", "Windsor") else "USA"
    )

    # Placeholder bookmaker (so your filters work visually)
    out["Bookie"] = "All"

    # Risk flag
    out["Risk"] = np.where(
        out["BetEdge Win %"] > 25, "✅",
        np.where(out["BetEdge Win %"] > 15, "⚠️", "❌")
    )
    return out

# ------------------ UI: NAV ------------------
with st.sidebar:
    page = option_menu(
        "🏇 EdgeBet",
        ["Overview", "Horse Racing", "EdgeBrain", "Performance", "How It Works"],
        icons=["house", "activity", "robot", "bar-chart", "info-circle"],
        default_index=0,
    )

# ------------------ OVERVIEW ------------------
if page == "Overview":
    st.title("📊 EdgeBet – Live Racecards (Basic)")
    colA, colB = st.columns(2)
    with colA:
        day = st.selectbox("Day", ["today", "tomorrow"], index=0)
    resp = fetch_racecards_basic(day)
    if "_ok" in resp:
        st.success(f"Connected to TheRacingAPI (HTTP {resp['_status']})")
    else:
        st.error(f"API error: {resp.get('_error')} {resp.get('_status', '')}")
        if resp.get("_body"):
            with st.expander("Server response"):
                st.code(resp["_body"][:2000])
    payload = resp.get("_payload") if "_payload" in resp else {}
    df_live = enrich_with_values(parse_basic_payload(payload)) if payload else pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    c1.metric("Meetings (approx)", payload.get("meetingCount", len(payload)) if isinstance(payload, dict) else (0 if df_live.empty else df_live["Course"].nunique()))
    c2.metric("Races (visible)", len(df_live["Time"].unique()) if not df_live.empty else 0)
    c3.metric("Runners (parsed)", len(df_live))

    st.subheader("Sample (Top 10 by BetEdge)")
    if df_live.empty:
        st.info("No rows parsed yet (possibly no runners on Basic).")
    else:
        st.dataframe(df_live.sort_values("BetEdge Win %", ascending=False).head(10), use_container_width=True)

    with st.expander("Debug: raw JSON (first 2000 chars)"):
        st.code(json.dumps(payload, indent=2)[:2000])

# ------------------ HORSE RACING ------------------
elif page == "Horse Racing":
    st.title("🏇 Horse Racing – Live (Basic)")

    colA, colB = st.columns([1, 2])
    with colA:
        day = st.radio("Day", ["today", "tomorrow"], horizontal=True)

    resp = fetch_racecards_basic(day)
    payload = resp.get("_payload") if "_payload" in resp else {}
    df = enrich_with_values(parse_basic_payload(payload)) if payload else pd.DataFrame()

    if df.empty:
        st.warning("No runners found on Basic for this day. (Odds may not be included on the Basic plan.)")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            country = st.selectbox("Country", ["All", "UK", "USA"])
        with col2:
            courses = sorted(df["Course"].dropna().unique().tolist())
            default_courses = courses[:]  # select all by default
            course_filter = st.multiselect("Courses", courses, default=default_courses)
        with col3:
            # placeholder bookie filter
            bookie = st.selectbox("Bookmaker", ["All"])

        min_edge = int(max(0, math.floor(df["BetEdge Win %"].min())))
        max_edge = int(math.ceil(df["BetEdge Win %"].max()))
        edge_range = st.slider("Filter BetEdge Win %", min_edge, max_edge, (min_edge, max_edge))

        # Apply filters
        filtered = df[
            ((df["Country"] == country) | (country == "All"))
            & (df["Course"].isin(course_filter))
            & (df["BetEdge Win %"].between(*edge_range))
        ].copy()

        # Sort highest value first (like BookieBashing)
        filtered = filtered.sort_values(["BetEdge Win %", "Predicted Win %"], ascending=False)

        view = st.radio("View", ["Charts", "Tables"], horizontal=True)
        if filtered.empty:
            st.info("No rows match your filters.")
        else:
            if view == "Charts":
                st.subheader("Top 20 BetEdge Win %")
                st.bar_chart(filtered.head(20).set_index("Horse")["BetEdge Win %"])
                st.subheader("Top 20 BetEdge Place %")
                st.bar_chart(filtered.head(20).set_index("Horse")["BetEdge Place %"])
            else:
                st.subheader("Win Rankings (sorted by BetEdge)")
                st.dataframe(
                    filtered[["Course", "Time", "Horse", "Odds", "Predicted Win %", "Win_Value", "BetEdge Win %", "Risk"]],
                    use_container_width=True
                )

# ------------------ EDGEBRAIN (stub for now) ------------------
elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain – Model Output (Stub)")
    day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
    resp = fetch_racecards_basic(day)
    payload = resp.get("_payload") if "_payload" in resp else {}
    df = enrich_with_values(parse_basic_payload(payload)) if payload else pd.DataFrame()
    if df.empty:
        st.info("No runners to score yet. When odds are available, EdgeBrain will use them.")
    else:
        # If you have a trained model, replace this with real proba
        df["EdgeBrain Win %"] = df["BetEdge Win %"]
        st.dataframe(df.sort_values("EdgeBrain Win %", ascending=False)[
            ["Course", "Time", "Horse", "Odds", "EdgeBrain Win %", "Risk"]
        ], use_container_width=True)

# ------------------ PERFORMANCE (upload) ------------------
elif page == "Performance":
    st.title("📈 Historical Performance")
    uploaded = st.file_uploader("Upload historical.csv", type="csv")
    if uploaded:
        hist = pd.read_csv(uploaded)
        # normalise likely columns
        for col in ("Stake", "Return"):
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0.0)
        if "Outcome" in hist.columns:
            hist["Outcome"] = hist["Outcome"].astype(str)

        total_bets  = len(hist)
        total_stake = float(hist["Stake"].sum()) if "Stake" in hist.columns else 0.0
        total_ret   = float(hist["Return"].sum()) if "Return" in hist.columns else 0.0
        profit      = total_ret - total_stake
        roi         = (profit / total_stake * 100.0) if total_stake else 0.0
        sr          = (
            (hist["Outcome"].str.lower() == "win").mean() * 100.0
            if "Outcome" in hist.columns and total_bets else 0.0
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Bets", total_bets)
        c2.metric("Stake", f"£{total_stake:,.2f}")
        c3.metric("Return", f"£{total_ret:,.2f}")
        c4.metric("P/L", f"£{profit:,.2f}")
        c5.metric("ROI", f"{roi:.1f}%")
        st.metric("Strike Rate", f"{sr:.1f}%")
        st.subheader("Sample rows")
        st.dataframe(hist.head(50), use_container_width=True)
    else:
        st.info("Drop in your historical.csv to see ROI, strike rate and P&L.")

# ------------------ HOW IT WORKS ------------------
else:
    st.title("ℹ️ How EdgeBet Works")
    st.markdown(
        "- Pull **live racecards** from TheRacingAPI (Basic).\n"
        "- When decimal odds are available, compute **Predicted Win%** and **BetEdge** value.\n"
        "- Filter by **country/course** and view **Charts** or **Tables**.\n"
        "- EdgeBrain tab shows model output (stubbed until we switch to a trained model).\n"
    )

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
