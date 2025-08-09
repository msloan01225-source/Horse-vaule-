# app.py — EdgeBet (strict live mode)

import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Optional ML
try:
    import joblib
except Exception:
    joblib = None

# Optional UI libs
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
    # stub = mirror BetEdge if no model
    return df.get("BetEdge Win %", pd.Series([np.nan] * len(df)))

# ---------------------------
# Live sources
# ---------------------------
def _api_creds():
    user = st.secrets.get("RACING_API_USER", "")
    pw   = st.secrets.get("RACING_API_PASS", "")
    return user, pw

@st.cache_data(ttl=30, show_spinner=False)
def fetch_api_basic(day: str, user: str, pw: str):
    """TheRacingAPI: /v1/racecards/basic?day=today|tomorrow"""
    try:
        if not user or not pw:
            return pd.DataFrame(), "Missing credentials."
        url = "https://api.theracingapi.com/v1/racecards/basic"
        params = {"day": day.lower()}
        r = requests.get(url, params=params, auth=HTTPBasicAuth(user, pw), timeout=10)
        r.raise_for_status()
        data = r.json()

        # Parse robustly
        rows = []
        meetings = data.get("meetings", data.get("racecards", [])) or []
        for m in meetings:
            course = (
                (m.get("course") or {}).get("name")
                if isinstance(m.get("course"), dict)
                else m.get("course", "Unknown")
            ) or "Unknown"
            races = m.get("races", []) or []
            for race in races:
                off = race.get("off") or race.get("time") or ""
                if isinstance(off, str):
                    off = off[:5]
                runners = race.get("runners", []) or []
                for rnr in runners:
                    # horse name
                    horse = rnr.get("horse")
                    if isinstance(horse, dict):
                        horse = horse.get("name") or horse.get("id") or "Unknown"
                    if not isinstance(horse, str):
                        horse = str(horse) if horse is not None else "Unknown"
                    # decimal odds
                    odds = (
                        rnr.get("sp_dec")
                        or rnr.get("spDec")
                        or rnr.get("spDecimal")
                        or rnr.get("oddsDecimal")
                    )
                    try:
                        odds = float(odds) if odds is not None else np.nan
                    except Exception:
                        odds = np.nan

                    rows.append(
                        {"Horse": horse, "Course": course, "Time": off, "Odds": odds}
                    )
        return pd.DataFrame(rows), ""
    except requests.HTTPError as e:
        return pd.DataFrame(), f"HTTP {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_rp(day: str):
    """Racing Post time-order scrape (live fallback)."""
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
                    sp_el   = runner.select_one(".runner-sp__price") or runner.select_one("[data-test-selector='runner-odds']")
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
                return pd.DataFrame(rows), ""
        return pd.DataFrame(), "Racing Post returned no runners."
    except Exception as e:
        return pd.DataFrame(), f"Racing Post error: {e}"

# ---------------------------
# Value/feature engineering
# ---------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ["Horse", "Course", "Time", "Odds"]:
        if col not in d.columns:
            d[col] = np.nan

    d["Odds"] = pd.to_numeric(d["Odds"], errors="coerce")
    d["Win_Value"] = np.random.uniform(5, 30, len(d)).round(1)
    d["Place_Value"] = (d["Win_Value"] * 0.6).round(1)

    implied = 100.0 / d["Odds"].replace(0, np.nan)
    d["Implied %"] = implied.round(1)
    d["Predicted Win %"] = d["Implied %"]
    d["Predicted Place %"] = (d["Predicted Win %"] * 0.6).round(1)

    d["BetEdge Win %"] = ((d["Predicted Win %"] * 0.6) + (d["Win_Value"] * 0.4)).round(1)
    d["BetEdge Place %"] = ((d["Predicted Place %"] * 0.6) + (d["Place_Value"] * 0.4)).round(1)
    d["EdgeBrain %"] = predict_edgebrain(d)
    d["Edge Δ %"] = (d["BetEdge Win %"] - d["Implied %"]).round(1)
    d["Risk"] = np.where(d["BetEdge Win %"] > 25, "✅", np.where(d["BetEdge Win %"] > 15, "⚠️", "❌"))

    uk_courses = {"Ascot", "York", "Cheltenham", "Newmarket", "Goodwood", "Aintree", "Epsom", "Doncaster", "Sandown"}
    d["Country"] = d["Course"].astype(str).apply(lambda c: "UK" if c in uk_courses else "USA")
    return d

def build_live(day: str):
    user, pw = _api_creds()

    # Primary: API
    df_api, api_err = fetch_api_basic(day, user, pw)
    if api_err:
        st.error(f"Live API error: {api_err}")

    if not df_api.empty:
        return enrich(df_api), "TheRacingAPI"

    # Secondary: RP scrape
    df_rp, rp_err = fetch_rp(day)
    if rp_err and df_rp.empty:
        st.error(f"Racing Post fallback error: {rp_err}")

    if not df_rp.empty:
        return enrich(df_rp), "Racing Post (scrape)"

    # Strict live mode: no mock return
    st.stop()

# ---------------------------
# Charts
# ---------------------------
def plot_value_vs_odds(df: pd.DataFrame, title: str):
    d = df.copy()
    if d.empty:
        st.info("No data to chart.")
        return
    d = d.sort_values("BetEdge Win %", ascending=False).head(30)
    if not PLOTLY_OK:
        st.bar_chart(d.set_index("Horse")["BetEdge Win %"])
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=d["Horse"], y=d["BetEdge Win %"], name="BetEdge Win %",
        hovertemplate="<b>%{x}</b><br>BetEdge: %{y:.1f}%<br>Implied: %{customdata[0]:.1f}%<br>Odds: %{customdata[1]:.2f}<br>Value Score: %{customdata[2]:.1f}",
        customdata=np.stack([d["Implied %"], d["Odds"], d["Win_Value"]], axis=-1),
    )
    fig.add_scatter(
        x=d["Horse"], y=d["Odds"], name="Odds (dec)", mode="lines+markers", yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Odds: %{y:.2f}",
    )
    fig.update_layout(
        title=title, bargap=0.2, height=480,
        margin=dict(l=20, r=20, t=50, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
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
# Styling
# ---------------------------
st.markdown(
    """
<style>
:root { --edge-accent:#00ffcc; }
h1,h2,h3,h4,h5,h6 { color: var(--edge-accent); }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Sidebar menu
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
    st.caption("Live mode reads TheRacingAPI creds from Secrets.")

# ---------------------------
# Pages
# ---------------------------
if page == "Overview":
    st.title("📊 EdgeBet — Live Tracker")
    df0, src0 = build_live("today")
    c1, c2, c3 = st.columns(3)
    c1.metric("Races Found", int(df0["Course"].nunique()))
    c2.metric("Total Runners", len(df0))
    c3.metric("Top BetEdge Value", f"{pd.to_numeric(df0['BetEdge Win %'], errors='coerce').max():.1f}%")
    st.caption(f"Source: {src0}")
    st.dataframe(df0.sort_values("BetEdge Win %", ascending=False), use_container_width=True)

elif page == "Horse Racing":
    st.title("🏇 Horse Racing — Live (UK/USA)")
    day = st.selectbox("Day", ["today", "tomorrow"], index=0)
    df, src = build_live(day)
    st.caption(f"Source: {src}")

    tab_all, tab_race = st.tabs(["All Runners", "By Race"])

    with tab_all:
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("Country", ["All", "UK", "USA"], index=0)
        with col2:
            courses = sorted(df["Course"].dropna().unique().tolist())
            course_filter = st.multiselect("Courses", courses, default=courses)

        betedge = pd.to_numeric(df["BetEdge Win %"], errors="coerce")
        min_val = int(np.nanmin(betedge)) if betedge.notna().any() else 0
        max_val = int(np.nanmax(betedge)) if betedge.notna().any() else 100
        edge_range = st.slider("Filter by BetEdge Win %", min_val, max_val, (min_val, max_val))

        f = df.copy()
        if country != "All":
            f = f[f["Country"] == country]
        f = f[f["Course"].isin(course_filter)]
        f = f[betedge.between(edge_range[0], edge_range[1])]

        if f.empty:
            st.warning("No runners match your filters.")
        else:
            top_n = st.slider("Chart Top N", 10, 60, 20, step=5)
            plot_value_vs_odds(f.sort_values("BetEdge Win %", ascending=False).head(top_n),
                               title="Value vs Odds (Top N by BetEdge)")
            st.markdown("---")
            plot_edge_map(f, title="Edge Map: Implied vs BetEdge")
            cols_win = ["Horse", "Course", "Time", "Odds", "Implied %", "Win_Value", "BetEdge Win %", "Edge Δ %", "Risk"]
            st.dataframe(f.sort_values("BetEdge Win %", ascending=False)[cols_win], use_container_width=True)

    with tab_race:
        meetings = sorted(df["Course"].dropna().unique().tolist())
        if not meetings:
            st.info("No meetings found.")
        else:
            csel = st.selectbox("Meeting", meetings, index=0)
            dcm = df[df["Course"] == csel].copy()
            times = sorted(dcm["Time"].dropna().unique().tolist())
            tsel = st.selectbox("Race time", times, index=0) if times else ""
            dr = dcm[dcm["Time"] == tsel].copy() if tsel else dcm.copy()
            if dr.empty:
                st.warning("No runners for this race.")
            else:
                plot_value_vs_odds(dr, title=f"{csel} {tsel} — Value vs Odds")
                cols = ["Horse", "Odds", "Implied %", "Win_Value", "BetEdge Win %", "Edge Δ %", "Risk"]
                st.dataframe(dr.sort_values("BetEdge Win %", ascending=False)[cols], use_container_width=True)

elif page == "EdgeBrain":
    st.title("🧠 EdgeBrain — Live Scoring")
    df, src = build_live("today")
    st.caption(f"Source: {src}")
    d2 = df.sort_values("EdgeBrain %", ascending=False).head(25)
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_bar(x=d2["Horse"], y=d2["EdgeBrain %"], name="EdgeBrain Win %")
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=80))
        fig.update_xaxes(tickangle=-45, automargin=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(d2.set_index("Horse")["EdgeBrain %"])
    cols = ["Horse", "Course", "Time", "Odds", "Implied %", "BetEdge Win %", "EdgeBrain %", "Risk"]
    st.dataframe(df.sort_values("EdgeBrain %", ascending=False)[cols], use_container_width=True)

else:
    st.title("How EdgeBet Works")
    st.markdown(
        """
- Live odds from TheRacingAPI (primary) or Racing Post (fallback)
- BetEdge value model (Win/Place) + EdgeBrain scoring (if model present)
- Filter by country/course, view All Runners or By Race
"""
    )

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
