# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Optional: ML model
try:
    import joblib
except Exception:
    joblib = None

# Optional: Plotly / option menu (fallback to Streamlit widgets if missing)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    from streamlit_option_menu import option_menu
    MENU_OK = True
except Exception:
    MENU_OK = False

st.set_page_config(page_title="EdgeBet", layout="wide")

# ---------------------------
# EdgeBrain model (optional)
# ---------------------------
@st.cache_resource
def load_edgebrain_model():
    if joblib is None:
        return None
    try:
        return joblib.load("edgebrain_model.pkl")
    except Exception:
        return None

EDGE_MODEL = load_edgebrain_model()

def predict_edgebrain(df: pd.DataFrame) -> pd.Series:
    if EDGE_MODEL is not None:
        cols = ["Odds", "Win_Value", "Place_Value"]
        X = df.reindex(columns=cols, fill_value=np.nan)
        try:
            return (EDGE_MODEL.predict_proba(X)[:, 1] * 100).round(1)
        except Exception:
            pass
    # Stub fallback
    return df.get("BetEdge Win %", pd.Series([np.nan] * len(df)))

# ---------------------------
# API (TheRacingAPI) – primary live source
# ---------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_api_basic(day: str, user: str, pw: str):
    """
    Calls: https://api.theracingapi.com/v1/racecards/basic?day=today|tomorrow
    Returns (DataFrame, source_label, error_text)
    """
    try:
        if not user or not pw:
            return pd.DataFrame(), "NO CREDS", "Missing API credentials."
        url = "https://api.theracingapi.com/v1/racecards/basic"
        params = {"day": day.lower()}  # 'today' or 'tomorrow'
        r = requests.get(url, params=params, auth=HTTPBasicAuth(user, pw), timeout=10)
        r.raise_for_status()
        data = r.json()
        # Parse robustly
        rows = []
        meetings = data.get("meetings", data.get("racecards", []))  # support both keys
        for m in meetings or []:
            course = (
                (m.get("course") or {}).get("name")
                if isinstance(m.get("course"), dict)
                else m.get("course", "Unknown")
            ) or "Unknown"
            races = m.get("races", [])
            for race in races or []:
                off = race.get("off") or race.get("time") or ""
                if isinstance(off, str):
                    off = off[:5]
                runners = race.get("runners", [])
                for rnr in runners or []:
                    # Odds may appear in different fields; try a few
                    odds = (
                        rnr.get("sp_dec", None)
                        or rnr.get("spDec", None)
                        or rnr.get("spDecimal", None)
                        or rnr.get("oddsDecimal", None)
                    )
                    # Name can be in 'horse' or nested
                    horse = rnr.get("horse")
                    if isinstance(horse, dict):
                        horse = horse.get("name") or horse.get("id") or "Unknown"
                    if not isinstance(horse, str):
                        horse = str(horse) if horse is not None else "Unknown"
                    try:
                        odds = float(odds) if odds is not None else np.nan
                    except Exception:
                        odds = np.nan
                    rows.append(
                        {
                            "Horse": horse,
                            "Course": course,
                            "Time": off if off else "",
                            "Odds": odds,
                        }
                    )
        df = pd.DataFrame(rows)
        return df, "TheRacingAPI (basic)", ""
    except requests.HTTPError as e:
        return pd.DataFrame(), "API ERROR", f"HTTP {e.response.status_code} {e.response.text}"
    except Exception as e:
        return pd.DataFrame(), "API ERROR", str(e)

# ---------------------------
# Racing Post scrape – secondary fallback
# ---------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_rp(day: str):
    try:
        target = datetime.utcnow().date()
        if day.lower() == "tomorrow":
            target += timedelta(days=1)
        urls = [
            f"https://www.racingpost.com/racecards/time-order/{target.isoformat()}",
            "https://www.racingpost.com/racecards/time-order/",
        ]
        for url in urls:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            rows = []
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
                    sp_el = runner.select_one(".runner-sp__price") or runner.select_one("[data-test-selector='runner-odds']")
                    if not name_el or not sp_el:
                        continue
                    name = name_el.get_text(strip=True)
                    sp_txt = sp_el.get_text(strip=True).replace(" ", "")
                    try:
                        odds = float(sp_txt)
                    except Exception:
                        odds = np.nan
                    rows.append({"Horse": name, "Course": course, "Time": off, "Odds": odds})
            if rows:
                return pd.DataFrame(rows), "RacingPost (scrape)"
        return pd.DataFrame(), "RacingPost (scrape)"
    except Exception:
        return pd.DataFrame(), "RacingPost (scrape)"

# ---------------------------
# Mock fallback – tertiary
# ---------------------------
def generate_mock(n: int = 60) -> pd.DataFrame:
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot", "York", "Cheltenham", "Newmarket", "Goodwood", "Churchill Downs", "Santa Anita"],
        n,
    )
    times = np.random.choice([f"{h:02d}:{m:02d}" for h in range(12, 21) for m in (0, 30)], n)
    odds = np.random.uniform(2.0, 12.0, n).round(2)
    return pd.DataFrame({"Horse": horses, "Course": courses, "Time": times, "Odds": odds})

# ---------------------------
# Value engine + helpers
# ---------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ["Odds", "Horse", "Course", "Time"]:
        if col not in d.columns:
            d[col] = np.nan

    d["Win_Value"] = np.random.uniform(5, 30, len(d)).round(1)
    d["Place_Value"] = (d["Win_Value"] * 0.6).round(1)

    # Avoid div-by-zero
    odds_num = pd.to_numeric(d["Odds"], errors="coerce").replace(0, np.nan)
    d["Implied %"] = (100.0 / odds_num).round(1)
    d["Predicted Win %"] = d["Implied %"]  # simple baseline
    d["Predicted Place %"] = (d["Predicted Win %"] * 0.6).round(1)

    d["BetEdge Win %"] = ((d["Predicted Win %"] * 0.6) + (d["Win_Value"] * 0.4)).round(1)
    d["BetEdge Place %"] = ((d["Predicted Place %"] * 0.6) + (d["Place_Value"] * 0.4)).round(1)
    d["EdgeBrain %"] = predict_edgebrain(d)

    d["Edge Δ %"] = (d["BetEdge Win %"] - d["Implied %"]).round(1)
    d["Risk"] = np.where(d["BetEdge Win %"] > 25, "✅", np.where(d["BetEdge Win %"] > 15, "⚠️", "❌"))

    uk_courses = {"Ascot", "York", "Cheltenham", "Newmarket", "Goodwood", "Aintree", "Epsom"}
    d["Country"] = d["Course"].apply(lambda c: "UK" if str(c) in uk_courses else "USA")
    return d

def build_live(day: str, user: str, pw: str):
    # 1) Primary: API
    df_api, src, err = fetch_api_basic(day, user, pw)
    if not df_api.empty:
        return enrich(df_api), src, err
    # 2) Fallback: RacingPost
    df_rp, src2 = fetch_rp(day)
    if not df_rp.empty:
        return enrich(df_rp), src2, err or ""
    # 3) Mock
    return enrich(generate_mock()), "Mock (no live)", err or ""

# ---------------------------
# Charts
# ---------------------------
def plot_value_vs_odds(df: pd.DataFrame, title: str):
    d = df.copy()
    if d.empty:
        st.info("No data to chart.")
        return
    if not PLOTLY_OK:
        st.bar_chart(d.set_index("Horse")["BetEdge Win %"])
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=d["Horse"], y=d["BetEdge Win %"], name="BetEdge Win %",
        hovertemplate="<b>%{x}</b><br>BetEdge: %{y:.1f}%<br>Implied: %{customdata[0]:.1f}%<br>Odds: %{customdata[1]:.2f}<br>Value: %{customdata[2]:.1f}%",
        customdata=np.stack([d["Implied %"], d["Odds"], d["Win_Value"]], axis=-1),
    )
    fig.add_scatter(
        x=d["Horse"], y=d["Odds"], name="Odds (dec)", mode="lines+markers",
        hovertemplate="<b>%{x}</b><br>Odds: %{y:.2f}",
        yaxis="y2",
    )
    fig.update_layout(
        title=title, bargap=0.2, height=460,
        margin=dict(l=20, r=20, t=50, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(title_text="BetEdge Win %", secondary_y=False)
    fig.update_yaxes(title_text="Odds (decimal)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_edge_map(df: pd.DataFrame, title: str):
    d = df.copy()
    if d.empty:
        st.info("No data to chart.")
        return
    if not PLOTLY_OK:
        st.scatter_chart(d[["Implied %", "BetEdge Win %"]])
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["Implied %"], y=d["BetEdge Win %"], mode="markers", name="Runners",
            marker=dict(
                size=np.clip((d["Win_Value"].fillna(0) * 0.9) + 6, 6, 26),
                color=(d["BetEdge Win %"] - d["Implied %"]),
                colorscale="Turbo", showscale=True, colorbar=dict(title="Edge Δ %"),
            ),
            text=d["Horse"],
            hovertemplate="<b>%{text}</b><br>Implied: %{x:.1f}%<br>BetEdge: %{y:.1f}%<br>Odds: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.stack([d["Odds"]], axis=-1),
        )
    )
    # parity line
    x0 = float(np.nanmin(d["Implied %"])) if len(d) else 0.0
    x1 = float(np.nanmax(d["Implied %"])) if len(d) else 100.0
    fig.add_trace(go.Scatter(x=[x0, x1], y=[x0, x1], mode="lines", name="Parity", line=dict(dash="dash")))
    fig.update_layout(
        title=title, xaxis_title="Implied Probability (%)", yaxis_title="BetEdge Win (%)",
        height=480, margin=dict(l=20, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Simple accent styling
# ---------------------------
st.markdown(
    """
<style>
:root { --edge-accent: #00ffcc; }
h1, h2, h3, h4, h5, h6 { color: var(--edge-accent); }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Sidebar: menu + API creds
# ---------------------------
with st.sidebar:
    if MENU_OK:
        page = option_menu(
            "🏇 EdgeBet",
            ["Overview", "Horse Racing", "EdgeBrain", "How It Works"],
            icons=["house", "bar-chart", "robot", "info-circle"],
            default_index=0,
        )
    else:
        page = st.selectbox("Menu", ["Overview", "Horse Racing", "EdgeBrain", "How It Works"])
    st.markdown("---")
    st.subheader("Live API (TheRacingAPI)")
    default_user = st.secrets.get("RACING_API_USER", "")
    default_pass = st.secrets.get("RACING_API_PASS", "")
    API_USER = st.text_input("Username", value=default_user)
    API_PASS = st.text_input("Password", value=default_pass, type="password")

# ---------------------------
# Pages
# ---------------------------
if page == "Overview":
    st.title("📊 EdgeBet Live Tracker")
    df0, src0, err0 = build_live("today", API_USER, API_PASS)
    c1, c2, c3 = st.columns(3)
    c1.metric("Races Found", int(df0["Course"].nunique()))
    c2.metric("Total Runners", len(df0))
    c3.metric("Top BetEdge Value", f"{pd.to_numeric(df0['BetEdge Win %'], errors='coerce').max():.1f}%")
    st.caption(f"Source: {src0}" + (f" | Note: {err0}" if err0 else ""))

elif page == "Horse Racing":
    st.title("🏇 Horse Racing — Live UK/USA")
    day = st.selectbox("Select Day", ["today", "tomorrow"], index=0)
    df, src, err = build_live(day, API_USER, API_PASS)
    st.caption(f"Source: {src}" + (f" | Note: {err}" if err else ""))

    tab_all, tab_by_race = st.tabs(["All Runners", "By Race"])

    with tab_all:
        col1, col2, col3 = st.columns(3)
        with col1:
            country = st.selectbox("Country", ["All", "UK", "USA"], index=0)
        with col2:
            bookie = st.selectbox("Bookmaker", ["All", "SkyBet", "Bet365", "Betfair"], index=0)
        with col3:
            courses_list = sorted(df["Course"].dropna().unique().tolist())
            course_filter = st.multiselect("Courses", courses_list, default=courses_list)

        betedge = pd.to_numeric(df["BetEdge Win %"], errors="coerce")
        min_val = int(np.nanmin(betedge)) if betedge.notna().any() else 0
        max_val = int(np.nanmax(betedge)) if betedge.notna().any() else 100
        edge_range = st.slider("Filter by BetEdge Win %", min_val, max_val, (min_val, max_val))

        filtered = df.copy()
        if country != "All":
            filtered = filtered[filtered["Country"] == country]
        filtered = filtered[filtered["Course"].isin(course_filter)]
        filtered = filtered[pd.to_numeric(filtered["BetEdge Win %"], errors="coerce").between(edge_range[0], edge_range[1])]

        view_mode = st.radio("View Mode", ["📊 Charts", "📋 Tables"], horizontal=True)

        if filtered.empty:
            st.warning("No horses match your filters.")
        else:
            if view_mode == "📊 Charts":
                top_n = st.slider("How many horses to chart?", 10, 60, 20, step=5)
                plot_value_vs_odds(
                    filtered.sort_values("BetEdge Win %", ascending=False).head(top_n),
                    title="Value vs Odds (Top N by BetEdge)",
                )
                st.markdown("---")
                plot_edge_map(filtered, title="Edge Map: Implied vs BetEdge")
            else:
                df_win = filtered.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
                df_place = filtered.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)
                st.subheader("🏆 Win Rankings")
                cols_win = ["Horse", "Course", "Time", "Odds", "Implied %", "Win_Value", "BetEdge Win %", "Edge Δ %", "Risk"]
                cols_win = [c for c in cols_win if c in df_win.columns]
                st.dataframe(df_win[cols_win], use_container_width=True)

                st.subheader("🥈 Place Rankings")
                cols_place = ["Horse", "Course", "Time", "Odds", "Place_Value", "BetEdge Place %", "Risk"]
                cols_place = [c for c in cols_place if c in df_place.columns]
                st.dataframe(df_place[cols_place], use_container_width=True)

    with tab_by_race:
        if df.empty:
            st.info("No runners available.")
        else:
            meetings = sorted(df["Course"].dropna().unique().tolist())
            sel_course = st.selectbox("Select Meeting (Course)", meetings, index=0)
            df_meet = df[df["Course"] == sel_course].copy()

            race_times = sorted(df_meet["Time"].dropna().unique().tolist())
            sel_time = st.selectbox("Select Race Time", race_times, index=0) if race_times else ""
            df_race = df_meet[df_meet["Time"] == sel_time].copy() if sel_time else df_meet.copy()

            if df_race.empty:
                st.warning("No runners for this race.")
            else:
                plot_value_vs_odds(
                    df_race.sort_values("BetEdge Win %", ascending=False),
                    title=f"{sel_course} {sel_time} — Value vs Odds",
                )
                cols = ["Horse", "Odds", "Implied %", "Win_Value", "BetEdge Win %", "Edge Δ %", "Risk"]
                cols = [c for c in cols if c in df_race.columns]
                st.dataframe(df_race.sort_values("BetEdge Win %", ascending=False)[cols], use_container_width=True)

elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    df, src, err = build_live("today", API_USER, API_PASS)
    st.caption(f"Source: {src}" + (f" | Note: {err}" if err else ""))
    if not df.empty:
        d2 = df.sort_values("EdgeBrain %", ascending=False).head(20)
        if PLOTLY_OK:
            fig = go.Figure()
            fig.add_bar(x=d2["Horse"], y=d2["EdgeBrain %"], name="EdgeBrain Win %")
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=80))
            fig.update_xaxes(tickangle=-45, automargin=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(d2.set_index("Horse")["EdgeBrain %"])
    cols = ["Horse", "Course", "Time", "Odds", "EdgeBrain %", "BetEdge Win %", "Risk"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df.sort_values("EdgeBrain %", ascending=False)[cols], use_container_width=True)

else:
    st.title("How EdgeBet Works")
    st.markdown(
        """
Pipeline:
- Live odds from TheRacingAPI (primary), Racing Post scrape (fallback), or mock if blocked
- Compute BetEdge value metrics (Win/Place) + EdgeBrain scoring (if model present)
- Filter by course/country, view charts or tables, inspect per-race views
"""
    )
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
