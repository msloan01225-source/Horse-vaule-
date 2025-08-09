# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Optional: EdgeBrain model (safe if file missing)
try:
    import joblib
except Exception:
    joblib = None

# Optional Plotly (safe fallback)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Optional menu (safe fallback)
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
    # Stub fallback = echo BetEdge Win %
    return df.get("BetEdge Win %", pd.Series([np.nan] * len(df)))

# ---------------------------
# Live SP scraping (RacingPost) with safe fallback
# ---------------------------
@st.cache_data(ttl=60)
def fetch_live_sp(day: str = "today") -> pd.DataFrame:
    """
    Try to scrape Racing Post time-order page for the given day.
    Returns empty DataFrame on any error (so we can mock fallback).
    """
    try:
        target = datetime.utcnow().date()
        if day.lower() == "tomorrow":
            target += timedelta(days=1)

        # Try dated URL first, then undated as a fallback
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

            # Primary selectors (may change; robustly skip if not found)
            sections = soup.select("section.race-time")
            if not sections:
                # Try a looser selector as a backstop
                sections = soup.select("section")

            for sec in sections:
                time_el = sec.select_one(".rc-race-time")
                course_el = sec.select_one(".rc-race-meeting__course a")
                if not time_el or not course_el:
                    continue

                off = time_el.get_text(strip=True)
                course = course_el.get_text(strip=True)

                runners = sec.select(".runner-wrap")
                if not runners:
                    # Try some fallbacks
                    runners = sec.select("[data-test-selector='runner']")
                for runner in runners:
                    name_el = runner.select_one(".runner-runner__name")
                    sp_el = runner.select_one(".runner-sp__price")
                    if not name_el or not sp_el:
                        # Try alternative labels
                        name_el = name_el or runner.select_one("[data-test-selector='runner-name']")
                        sp_el = sp_el or runner.select_one("[data-test-selector='runner-odds']")
                        if not name_el or not sp_el:
                            continue
                    name = name_el.get_text(strip=True)
                    sp_txt = sp_el.get_text(strip=True).replace(" ", "")
                    try:
                        odds = float(sp_txt)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "Horse": name,
                            "Course": course,
                            "Time": off,
                            "Odds": odds,
                        }
                    )

            if rows:
                return pd.DataFrame(rows)

        return pd.DataFrame()

    except Exception:
        return pd.DataFrame()

# ---------------------------
# Mock fallback + value engine
# ---------------------------
def generate_mock_data(n: int = 60) -> pd.DataFrame:
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot", "York", "Cheltenham", "Newmarket", "Goodwood", "Churchill Downs", "Santa Anita"],
        n,
    )
    times = np.random.choice([f"{h:02d}:{m:02d}" for h in range(12, 21) for m in (0, 30)], n)
    odds = np.random.uniform(2.0, 12.0, n).round(2)
    df = pd.DataFrame({"Horse": horses, "Course": courses, "Time": times, "Odds": odds})
    return df

def enrich_with_value_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Ensure required columns exist
    for col in ["Odds", "Horse", "Course", "Time"]:
        if col not in d.columns:
            d[col] = np.nan

    # Value metrics (placeholder logic; replace as you integrate true inputs)
    d["Win_Value"] = np.random.uniform(5, 30, len(d)).round(1)
    d["Place_Value"] = (d["Win_Value"] * 0.6).round(1)
    d["Predicted Win %"] = (100.0 / d["Odds"]).round(1)
    d["Predicted Place %"] = (d["Predicted Win %"] * 0.6).round(1)
    d["BetEdge Win %"] = ((d["Predicted Win %"] * 0.6) + (d["Win_Value"] * 0.4)).round(1)
    d["BetEdge Place %"] = ((d["Predicted Place %"] * 0.6) + (d["Place_Value"] * 0.4)).round(1)
    d["Risk"] = np.where(d["BetEdge Win %"] > 25, "✅", np.where(d["BetEdge Win %"] > 15, "⚠️", "❌"))

    # Country inference from course label (rough)
    uk_courses = {"Ascot", "York", "Cheltenham", "Newmarket", "Goodwood", "Aintree", "Epsom"}
    d["Country"] = d["Course"].apply(lambda c: "UK" if str(c) in uk_courses else "USA")

    # EdgeBrain (model or stub)
    d["EdgeBrain %"] = predict_edgebrain(d)

    return d

def build_live_df(day: str = "today") -> pd.DataFrame:
    live = fetch_live_sp(day)
    if live.empty:
        live = generate_mock_data()
    return enrich_with_value_cols(live)

# ---------------------------
# Plotly chart helpers (with fallback)
# ---------------------------
def _prep_for_charts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ["Odds", "BetEdge Win %", "Win_Value"]:
        if col not in d.columns:
            d[col] = np.nan
    d["Implied %"] = (100.0 / d["Odds"]).round(1)
    d["Edge Δ %"] = (d["BetEdge Win %"] - d["Implied %"]).round(1)
    d["Label"] = d.get("Horse", "").astype(str)
    if "Course" in d.columns and "Time" in d.columns:
        d["Label"] = d["Label"] + " — " + d["Course"].astype(str) + " " + d["Time"].astype(str)
    return d

def plot_value_vs_odds(df: pd.DataFrame, top_n: int = 20, title: str = "Value vs Odds (Top N)"):
    d = _prep_for_charts(df).sort_values("BetEdge Win %", ascending=False).head(top_n)
    if len(d) == 0:
        st.info("No data to chart.")
        return

    if not PLOTLY_OK:
        st.bar_chart(d.set_index("Horse")["BetEdge Win %"])
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=d["Horse"], y=d["BetEdge Win %"], name="BetEdge Win %",
            hovertemplate="<b>%{x}</b><br>BetEdge: %{y:.1f}%<br>Implied: %{customdata[0]:.1f}%<br>Odds: %{customdata[1]:.2f}<br>Value: %{customdata[2]:.1f}%",
            customdata=np.stack([d["Implied %"], d["Odds"], d["Win_Value"]], axis=-1),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=d["Horse"], y=d["Odds"], name="Odds (dec)", mode="lines+markers",
            hovertemplate="<b>%{x}</b><br>Odds: %{y:.2f}",
        ),
        secondary_y=True,
    )
    fig.update_layout(title=title, bargap=0.2, height=460, margin=dict(l=20, r=20, t=50, b=80))
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(title_text="BetEdge Win %", secondary_y=False)
    fig.update_yaxes(title_text="Odds (decimal)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_edge_map(df: pd.DataFrame, max_points: int = 300, title: str = "Edge Map: Implied vs BetEdge"):
    d = _prep_for_charts(df).sort_values("Edge Δ %", ascending=False).head(max_points)
    if len(d) == 0:
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
                color=d["Edge Δ %"], colorscale="Turbo", showscale=True,
                colorbar=dict(title="Edge Δ %"),
            ),
            text=d["Label"],
            hovertemplate="<b>%{text}</b><br>Implied: %{x:.1f}%<br>BetEdge: %{y:.1f}%<br>Edge Δ: %{marker.color:.1f}%<br>Odds: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.stack([d["Odds"]], axis=-1),
        )
    )
    x0 = float(np.nanmin(d["Implied %"])) if len(d) else 0.0
    x1 = float(np.nanmax(d["Implied %"])) if len(d) else 100.0
    fig.add_trace(go.Scatter(x=[x0, x1], y=[x0, x1], mode="lines", name="Parity", line=dict(dash="dash")))
    fig.update_layout(
        title=title, xaxis_title="Implied Probability (%)", yaxis_title="BetEdge Win (%)",
        height=480, margin=dict(l=20, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# UI
# ---------------------------
# Simple dark-ish accent (safe CSS)
st.markdown(
    """
<style>
:root { --edge-accent: #00ffcc; }
h1, h2, h3, h4, h5, h6 { color: var(--edge-accent); }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar menu
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

# Pages
if page == "Overview":
    st.title("📊 EdgeBet Live Tracker")
    df0 = build_live_df("today")
    c1, c2, c3 = st.columns(3)
    c1.metric("Races Found", int(df0["Course"].nunique()))
    c2.metric("Total Runners", len(df0))
    c3.metric("Top BetEdge Value", f"{pd.to_numeric(df0['BetEdge Win %'], errors='coerce').max():.1f}%")

elif page == "Horse Racing":
    st.title("🏇 Horse Racing — Live UK/USA")
    day = st.selectbox("Select Day", ["today", "tomorrow"], index=0)
    df = build_live_df(day)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country", ["All", "UK", "USA"], index=0)
    with col2:
        # Placeholder bookie filter (data not sourced yet)
        bookie = st.selectbox("Bookmaker", ["All", "SkyBet", "Bet365", "Betfair"], index=0)
    with col3:
        courses_list = sorted(df["Course"].dropna().unique().tolist())
        course_filter = st.multiselect("Courses", courses_list, default=courses_list)

    # Numeric slider bounds
    betedge = pd.to_numeric(df["BetEdge Win %"], errors="coerce")
    min_val = int(np.nanmin(betedge)) if betedge.notna().any() else 0
    max_val = int(np.nanmax(betedge)) if betedge.notna().any() else 100
    edge_range = st.slider("Filter by BetEdge Win %", min_val, max_val, (min_val, max_val))

    # Apply filters
    filtered = df.copy()
    if country != "All":
        filtered = filtered[filtered["Country"] == country]
    filtered = filtered[filtered["Course"].isin(course_filter)]
    filtered = filtered[pd.to_numeric(filtered["BetEdge Win %"], errors="coerce").between(edge_range[0], edge_range[1])]

    # View toggle
    view_mode = st.radio("View Mode", ["📊 Charts", "📋 Tables"], horizontal=True)

    if filtered.empty:
        st.warning("No horses match your filters.")
    else:
        if view_mode == "📊 Charts":
            top_n = st.slider("How many horses to chart?", 10, 60, 20, step=5)
            plot_value_vs_odds(filtered, top_n=top_n, title="Value vs Odds (Top N by BetEdge)")
            st.markdown("---")
            plot_edge_map(filtered, max_points=200, title="Edge Map: Implied vs BetEdge")
        else:
            # Tables
            df_win = filtered.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
            df_place = filtered.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

            st.subheader("🏆 Win Rankings")
            st.dataframe(df_win[["Horse", "Course", "Time", "Odds", "BetEdge Win %", "Win_Value", "Implied %"]]
                         if "Implied %" in df_win.columns else
                         df_win[["Horse", "Course", "Time", "Odds", "BetEdge Win %", "Win_Value"]],
                         use_container_width=True)

            st.subheader("🥈 Place Rankings")
            st.dataframe(df_place[["Horse", "Course", "Time", "Odds", "BetEdge Place %", "Place_Value"]],
                         use_container_width=True)

elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain Predictions")
    df = build_live_df("today")
    # Chart
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
    # Table
    cols = ["Horse", "Course", "Time", "Odds", "EdgeBrain %", "BetEdge Win %", "Risk"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df.sort_values("EdgeBrain %", ascending=False)[cols], use_container_width=True)

else:
    st.title("ℹ️ How EdgeBet Works")
    st.markdown(
        """
**Pipeline**
- Scrape live SP odds (or mock fallback if blocked)
- Compute real-time BetEdge value metrics (Win/Place)
- Optional EdgeBrain ML scoring (if model present)
- Filter by course, country, and show charts/tables
"""
    )
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
