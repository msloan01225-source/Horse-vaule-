import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from streamlit_option_menu import option_menu

# --- Load trained EdgeBrain model (cached) ---
@st.cache_resource
def load_edgebrain_model():
    return joblib.load("edgebrain_model.pkl")

edge_model = load_edgebrain_model()

# --- Generate/mock live horse‐racing data ---
def generate_mock_data(n=40):
    horses = [f"Horse {i+1}" for i in range(n)]
    courses = np.random.choice(
        ["Ascot","York","Cheltenham","Churchill Downs","Santa Anita"], n
    )
    countries = ["UK" if c in ["Ascot","York","Cheltenham"] else "USA" for c in courses]
    bookies = np.random.choice(["SkyBet","Bet365","Betfair"], n)
    odds = np.random.uniform(2,10,n).round(2)
    win_val = np.random.uniform(5,30,n).round(1)

    df = pd.DataFrame({
        "Horse":       horses,
        "Course":      courses,
        "Country":     countries,
        "Bookie":      bookies,
        "Odds":        odds,
        "Win_Value":   win_val,
        "Place_Value": (win_val*0.6).round(1),
    })
    df["Pred Win %"]     = (1/df["Odds"] * 100).round(1)
    df["Pred Place %"]   = (df["Pred Win %"] * 0.6).round(1)
    df["BetEdge Win %"]   = ((df["Pred Win %"]*0.6) + (df["Win_Value"]*0.4)).round(1)
    df["BetEdge Place %"] = ((df["Pred Place %"]*0.6) + (df["Place_Value"]*0.4)).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"]>25, "✅",
        np.where(df["BetEdge Win %"]>15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

# --- Build & score live DataFrame ---
df_live = generate_mock_data()
X_edge = df_live[["Odds","Win_Value","Place_Value"]]
df_live["EdgeBrain Win %"] = (edge_model.predict_proba(X_edge)[:,1] * 100).round(1)

# --- Streamlit page config & dark theme ---
st.set_page_config(page_title="EdgeBet – Phase 3", layout="wide")
st.markdown("""
<style>
body { background-color: #111111; color: white; }
h1,h2,h3,h4,h5,h6 { color: #00ffcc; }
.css-1d391kg { background-color: #222 !important; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar navigation ---
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview","Horse Racing","Football","EdgeBrain","Performance","How It Works"],
        icons=['house','activity','soccer','robot','bar-chart-line','book'],
        menu_icon="cast", default_index=0
    )

# --- Overview ---
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive value models.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners", len(df_live))
    st.metric("Top Edge Value", f"{df_live['BetEdge Win %'].max()}%")

# --- Horse Racing page ---
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", ["All","UK","USA"])
    with c2:
        bookie = st.selectbox("Bookmaker", ["All","SkyBet","Bet365","Betfair"])
    with c3:
        courses = df_live["Course"].unique().tolist()
        course_filter = st.multiselect("Courses", courses, default=courses)

    min_v, max_v = int(df_live["BetEdge Win %"].min()), int(df_live["BetEdge Win %"].max())
    edge_range = st.slider("🎯 BetEdge Win % range", min_v, max_v, (min_v, max_v))

    view_mode = st.radio("View Mode", ["📊 Charts","📋 Tables"], horizontal=True)

    filt = df_live[
        ((df_live["Country"]==country)|(country=="All")) &
        ((df_live["Bookie"]==bookie)|(bookie=="All")) &
        (df_live["Course"].isin(course_filter)) &
        df_live["BetEdge Win %"].between(*edge_range)
    ]

    if filt.empty:
        st.warning("No horses match your filters.")
    else:
        win = filt.sort_values("BetEdge Win %", ascending=False)
        place = filt.sort_values("BetEdge Place %", ascending=False)

        if view_mode == "📊 Charts":
            st.subheader("Top 20 BetEdge Win %")
            st.bar_chart(win.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 BetEdge Place %")
            st.bar_chart(place.head(20).set_index("Horse")["BetEdge Place %"])
        else:
            st.subheader("🏆 Win Rankings")
            st.dataframe(win, use_container_width=True)
            st.subheader("🥈 Place Rankings")
            st.dataframe(place, use_container_width=True)

# --- Football stub ---
elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Mock data – coming soon with EdgeBrain integration.")

# --- EdgeBrain page ---
elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("AI-enhanced Win% from our RandomForest model")
    st.dataframe(
        df_live[["Horse","Course","Odds","EdgeBrain Win %","Risk"]]
        .sort_values("EdgeBrain Win %", ascending=False),
        use_container_width=True
    )

# --- Performance / historical upload ---
elif selected == "Performance":
    st.title("📈 Historical Performance")
    uploaded = st.file_uploader("Upload historical.csv", type="csv")
    if not uploaded:
        st.info("Drop in your historical.csv to see ROI, strike rate & P&L.")
    else:
        hist = pd.read_csv(uploaded)
        hist["Stake"]  = pd.to_numeric(hist.get("Stake",0),errors="coerce").fillna(0)
        hist["Return"] = pd.to_numeric(hist.get("Return",0),errors="coerce").fillna(0)
        hist["Outcome"]= hist.get("Outcome","").astype(str)
        total_bets   = len(hist)
        total_stake  = hist["Stake"].sum()
        total_ret    = hist["Return"].sum()
        profit       = total_ret - total_stake
        roi          = (profit/total_stake*100) if total_stake else 0
        strike_rate  = (hist["Outcome"].str.lower()=="win").mean()*100
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Bets",       total_bets)
        c2.metric("Total Stake",      f"£{total_stake:.2f}")
        c3.metric("Profit/Loss",      f"£{profit:.2f}")
        c4.metric("ROI",              f"{roi:.1f}%")
        st.metric("Strike Rate",      f"{strike_rate:.1f}%")
        st.subheader("Sample Bets")
        st.dataframe(hist.head(20), use_container_width=True)

# --- How It Works ---
elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet uses a smart hybrid model combining:
    - ✅ Implied Probability from Odds  
    - 📊 Win & Place Value Metrics  
    - 🧠 Proprietary EdgeBrain Simulation  
    - 🔎 Risk Indicators & Smart Filtering  
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # swap in your final intro

# --- Footer ---
st.caption(f"Last updated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
