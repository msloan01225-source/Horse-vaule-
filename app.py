
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

# --- Page Config ---
st.set_page_config(page_title="BetEdge App", layout="wide")
st.markdown("<style>body { background-color: #0e1117; color: #ffffff; }</style>", unsafe_allow_html=True)

# --- Main Menu ---
selected = option_menu(
    menu_title=None,
    options=["Home", "Horse Racing", "Football", "Edge Brain"],
    icons=["house", "trophy", "soccer", "cpu"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# --- Home Page ---
if selected == "Home":
    st.title("🏇 Welcome to BetEdge")
    st.markdown("#### Your smart betting companion powered by data and machine learning.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Horse_racing_in_Japan.jpg/640px-Horse_racing_in_Japan.jpg", use_column_width=True)

    st.markdown("**Why BetEdge?**")
    st.markdown("- 📊 Data-Driven Value Picks")
    st.markdown("- 🧠 Proprietary Edge Brain Algorithms")
    st.markdown("- ⚽ Multi-Sport Support (Horse Racing, Football, More Coming Soon)")
    st.markdown("- 📈 Visualized Insights + Value Indicators")
    st.markdown("---")
    st.markdown("**Explore sections using the navigation bar above.**")

# --- Horse Racing Page ---
elif selected == "Horse Racing":
    st.title("🏇 Horse Racing – Value Tracker")

    # Mock data
    data = {
        "Horse": [f"Horse {i}" for i in range(1, 21)],
        "BetEdge Win %": np.random.uniform(5, 35, 20).round(1),
        "Best Odds": np.random.uniform(2.0, 8.0, 20).round(2)
    }
    df = pd.DataFrame(data).sort_values(by="BetEdge Win %", ascending=False)

    tab1, tab2 = st.tabs(["📊 Charts", "📋 Tables"])

    with tab1:
        st.subheader("Top 10 Horses by BetEdge Win %")
        st.bar_chart(df.head(10).set_index("Horse")["BetEdge Win %"])

    with tab2:
        st.subheader("Full Rankings")
        st.dataframe(df.reset_index(drop=True))

# --- Football Page ---
elif selected == "Football":
    st.title("⚽ Football Value Picks")
    st.info("Coming soon — Smart football picks and betting edges powered by the Edge Brain.")

# --- Edge Brain Page ---
elif selected == "Edge Brain":
    st.title("🧠 The Edge Brain")
    st.markdown("#### Behind every value pick is the intelligence of our proprietary algorithm.")
    st.markdown("The Edge Brain combines:")
    st.markdown("- ✔ Historical Data")
    st.markdown("- ✔ Real-Time Market Odds")
    st.markdown("- ✔ Predictive Modeling")
    st.markdown("---")
    st.markdown("Coming soon: Edge Scores, Confidence Metrics, and Interactive AI Picks.")
    st.image("https://cdn.pixabay.com/photo/2020/02/20/16/43/artificial-intelligence-4868872_1280.jpg", use_column_width=True)
