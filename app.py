import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="EdgeBet – Phase 3",
    layout="wide",
    initial_sidebar_state="auto"
)

# ---- THEMING ----
st.markdown(
    """
    <style>
    body { background-color: #111111; color: white; }
    h1, h2, h3, h4, h5, h6 { color: #00ffcc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- MOCK EDGE BRAIN DATA ----
def generate_mock_data(n=20):
    horses = [f"Horse {i+1}" for i in range(n)]
    odds = np.random.uniform(2, 10, n).round(2)
    win_val = np.random.uniform(5, 30, n).round(1)
    df = pd.DataFrame({
        "Horse": horses,
        "Odds": odds,
        "Predicted Win %": (1 / odds * 100).round(1),
        "Win_Value": win_val,
        "Place_Value": (win_val * 0.6).round(1),
    })
    df["BetEdge Win %"] = (
        df["Predicted Win %"] * 0.6 + df["Win_Value"] * 0.4
    ).round(1)
    df["BetEdge Place %"] = (
        df["Predicted Win %"] * 0.6 + df["Place_Value"] * 0.4
    ).round(1)
    df["Risk"] = np.where(
        df["BetEdge Win %"] > 25, "✅",
        np.where(df["BetEdge Win %"] > 15, "⚠️", "❌")
    )
    return df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)

df = generate_mock_data()

# ---- MAIN NAVIGATION ----
with st.sidebar:
    selected = option_menu(
        "🏠 Main Menu",
        ["Overview", "Horse Racing", "Football", "EdgeBrain", "How It Works"],
        icons=['house', 'activity', 'soccer', 'robot', 'book'],
        menu_icon="cast",
        default_index=0,
    )

# ---- PAGES ----
if selected == "Overview":
    st.title("📊 Welcome to EdgeBet")
    st.write("Get the edge over the bookmakers with cutting-edge predictive value models.")
    st.metric("Active Sports", 2)
    st.metric("Total Runners", len(df))
    st.metric("Top Edge Value", f"{df['BetEdge Win %'].max()}%")

elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – UK & USA")
    country = st.selectbox("Country", ["All", "UK", "USA"])
    bookie = st.selectbox("Bookmaker", ["All", "SkyBet", "Bet365", "Betfair"])
    course_filter = st.multiselect(
        "Courses",
        ["Ascot", "York", "Cheltenham", "Churchill Downs", "Santa Anita"],
        default=["Ascot", "York"],
    )
    show_all = st.checkbox("Show All Horses", value=True)

    st.subheader("Top Edge Picks")
    display_df = df if show_all else df.head(10)
    st.dataframe(display_df, use_container_width=True)

elif selected == "Football":
    st.title("⚽ Football Value Picks (Mock Data)")
    st.info("Coming soon with EdgeBrain integration...")

elif selected == "EdgeBrain":
    st.title("🧠 EdgeBrain – Smart Predictions")
    st.write("AI-enhanced simulated predictions using EdgeBrain model.")
    st.dataframe(
        df[["Horse", "Odds", "BetEdge Win %", "Risk"]],
        use_container_width=True
    )

elif selected == "How It Works":
    st.title("📚 How EdgeBet Works")
    st.markdown("""
    EdgeBet uses a smart hybrid model combining:
    - Implied Probability from Odds  
    - Value Metrics (Win & Place)  
    - Proprietary Simulation Logic  
    - Risk Indicators 🧠  

    This gives you a percentage-based **BetEdge score** and smart filters to guide your betting decisions.
    """)
    st.video("https://www.youtube.com/watch?v=2Xc9gXyf2G4")  # placeholder

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
