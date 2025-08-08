import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# ——— Auth & Config ———
USERNAME, PASSWORD = "…", "…"
st.set_page_config(page_title="EdgeBet", layout="wide")

# ——— Fetch Live ———
@st.cache_data(ttl=300)
def fetch_live_data(day="today"):
    d = datetime.utcnow().date() + (timedelta(days=1) if day=="tomorrow" else timedelta())
    url = "https://api.theracingapi.com/v1/racecards"
    params = {"date": d.isoformat(), "region": "GB"}  # adjust per API
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME,PASSWORD), params=params, timeout=10)
    r.raise_for_status(); return r.json()

def live_to_df(raw):
    rows=[]
    for m in raw.get("meetings",[]):
        c = m["course"]["name"]
        for race in m["races"]:
            t = race["off"][:5]
            for rnr in race["runners"]:
                odds = float(rnr.get("sp_dec",np.nan))
                if np.isnan(odds): continue
                val  = max((1/odds*100) - rnr.get("implied_prob",0)*100,0)
                rows.append({"Horse":rnr["horse"],"Course":c,"Time":t,
                             "Odds":odds,"Win_Value":val,"Place_Value":val*0.6})
    df = pd.DataFrame(rows)
    df["Pred Win %"] = (1/df["Odds"]*100).round(1)
    df["Pred Place %"] = (df["Pred Win %"]*0.6).round(1)
    df["BetEdge Win %"] = ((df["Pred Win %"]*0.6)+(df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"] = ((df["Pred Place %"]*0.6)+(df["Place_Value"]*0.4)).round(1)
    return df

# ——— Load History ———
@st.cache_data(ttl=600)
def load_history(): return pd.read_csv("historical.csv")
hist = load_history()
total_stake  = hist["Stake"].sum()
total_return = hist["Return"].sum()
profit       = total_return - total_stake
roi          = (profit/total_stake*100).round(1)
strike_rate  = (hist["Result"]=="Win").mean()*100

# ——— Build UI ———
with st.sidebar:
    tab = option_menu("Main Menu",
          ["Overview","Horse Racing","Performance","EdgeBrain","How It Works"],
          icons=["house","activity","bar-chart","robot","book"], default_index=0)

if tab=="Overview":
    st.title("Welcome to EdgeBet")
    st.metric("Total Runners", len(live_to_df(fetch_live_data())))
    st.metric("Top Value %", f"{live_to_df(fetch_live_data())['BetEdge Win %'].max():.1f}%")

elif tab=="Horse Racing":
    df = live_to_df(fetch_live_data())
    # … your existing filters & charts …

elif tab=="Performance":
    st.title("Back-Test Performance")
    st.metric("Stake",  f"£{total_stake:,.0f}")
    st.metric("Return", f"£{total_return:,.0f}")
    st.metric("Profit", f"£{profit:,.0f}")
    st.metric("ROI",    f"{roi}%")
    st.metric("Strike Rate", f"{strike_rate:.1f}%")
    st.dataframe(hist, use_container_width=True)

elif tab=="EdgeBrain":
    df = live_to_df(fetch_live_data())
    st.title("EdgeBrain Predictions")
    st.dataframe(df[["Horse","Course","BetEdge Win %","Risk"]])

elif tab=="How It Works":
    st.title("How It Works")
    st.video("https://youtube.com/…")    

st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide", initial_sidebar_state="auto")

# ---- DARK THEME STYLING ----
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# ---- MOCK EDGE BRAIN DATA ----
def generate_mock_data(n=40):
    horses   = [f"Horse {i+1}" for i in range(n)]
    courses  = np.random.choice(
        ["Ascot", "York", "Cheltenham", "Churchill Downs", "Santa Anita"], n
    )
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" for c in courses]
    bookies  = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds     = np.random.uniform(2,10,n).round(2)
    win_val  = np.random.uniform(5,30,n).round(1)

    df = pd.DataFrame({
        "Horse":        horses,
        "Course":       courses,
        "Country":      countries,
        "Bookie":       bookies,
        "Odds":         odds,
        "Win_Value":    win_val,
        "Place_Value":  (win_val * 0.6).round(1)
    })
    df["Predicted Win %"]    = (1/df["Odds"] * 100).round(1)
    df["Predicted Place %"]  = (df["Predicted Win %"] * 0.6).round(1)
    df["BetEdge Win %"]      = ((df["Predicted Win %"]*0.6) + (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"]    = ((df["Predicted Place %"]*0.6) + (df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"] > 25, "✅",
        np.where(df["BetEdge Win %"] > 15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ---- MAIN MENU ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview", "Horse Racing", "Football", "EdgeBrain", "Performance", "How It Works"],
        icons=['house','activity','soccer','robot','bar-chart-line','book'],
        menu_icon="cast", default_index=0
    )

# ---- OVERVIEW PAGE ----
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive value models.")
    st.metric("Active Sports",       2)
    st.metric("Total Runners",       len(df))
    st.metric("Top Edge Value",     f"{df['BetEdge Win %'].max()}%")

# ---- HORSE RACING PAGE ----
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")

    col1, col2, col3 = st.columns(3)
    with col1:
        country       = st.selectbox("Select Country",    ["All","UK","USA"])
    with col2:
        bookie        = st.selectbox("Select Bookmaker",  ["All","SkyBet","Bet365","Betfair"])
    with col3:
        courses_list  = df["Course"].unique().tolist()
        course_filter = st.multiselect("Select Courses", courses_list, default=courses_list)

    min_val = int(df["BetEdge Win %"].min())
    max_val = int(df["BetEdge Win %"].max())
    edge_range = st.slider("🎯 Filter by BetEdge Win %", min_val, max_val, (min_val, max_val))

    view_mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    # apply filters
    filtered = df[
        ((df["Country"]==country) | (country=="All")) &
        ((df["Bookie"]==bookie)   | (bookie=="All")) &
        (df["Course"].isin(course_filter)) &
        (df["BetEdge Win %"].between(*edge_range))
    ]

    df_win   = filtered.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
    df_place = filtered.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

    if filtered.empty:
        st.warning("No horses match your filters.")
    else:
        if view_mode == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(df_win,   use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(df_place, use_container_width=True)

# ---- FOOTBALL PAGE ----
elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data – coming soon with EdgeBrain integration.")

# ---- EDGEBRAIN PAGE ----
elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("AI-enhanced simulated predictions using EdgeBrain logic.")
    st.dataframe(df[["Horse","Course","Odds","BetEdge Win %","Risk"]], use_container_width=True)

# ---- PERFORMANCE PAGE ----
elif selected == "Performance":
    st.title("📈 Historical Performance")
    uploaded = st.file_uploader("Upload your historical.csv", type="csv")
    if not uploaded:
        st.info("Drop in your historical.csv to see ROI, strike rate, P&L, etc.")
    else:
        hist = pd.read_csv(uploaded)
        hist["Stake"]  = pd.to_numeric(hist["Stake"], errors="coerce").fillna(0)
        hist["Return"] = pd.to_numeric(hist["Return"],errors="coerce").fillna(0)
        hist["Outcome"]= hist["Outcome"].astype(str)

        total_bets   = len(hist)
        total_stake  = hist["Stake"].sum()
        total_ret    = hist["Return"].sum()
        profit       = total_ret - total_stake
        roi          = (profit/total_stake*100) if total_stake else 0
        wins         = hist[hist["Outcome"].str.lower()=="win"]
        strike_rate  = (len(wins)/total_bets*100) if total_bets else 0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Bets",       f"{total_bets}")
        c2.metric("Total Stake",      f"£{total_stake:.2f}")
        c3.metric("Total Profit/Loss",f"£{profit:.2f}")
        c4.metric("ROI",              f"{roi:.1f}%")
        st.metric("Strike Rate",      f"{strike_rate:.1f}%")

        st.subheader("Historical Bets (sample)")
        st.dataframe(hist.head(20), use_container_width=True)

# ---- HOW IT WORKS PAGE ----
elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet uses a smart hybrid model combining:

    - ✅ Implied Probability from Odds  
    - 📊 Win & Place Value Metrics  
    - 🧠 Proprietary EdgeBrain Simulation  
    - 🔎 Risk Indicators and Smart Filtering
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")

# ---- FOOTER ----
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
