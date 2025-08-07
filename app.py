import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random

# --- Page Config ---
st.set_page_config(page_title="BetEdge Sports Value Tracker", layout="wide")

# --- Sidebar Menu ---
st.sidebar.title("⚙️ Navigation")
sport_tab = st.sidebar.selectbox("Choose a Sport", ["Horse Racing", "Football"])

# --- Shared Styling ---
def color_val(v):
    if v > 20: return 'background-color:#58D68D;color:black'
    if v > 10: return 'background-color:#F9E79F;color:black'
    return 'background-color:#F5B7B1;color:black'

# --- Horse Racing Section ---
if sport_tab == "Horse Racing":
    st.title("🏇 BetEdge – Horse Racing Value Tracker")

    country = st.selectbox("Select Country", ["UK", "USA"])
    filter_value = st.slider("Minimum BetEdge Win %", 0, 100, 10)

    # Generate mock data
    def generate_horse_data(country):
        horses = []
        for i in range(50):
            name = f"Horse {i+1}"
            course = random.choice(["Ascot", "Cheltenham", "York", "Sandown", "Newmarket"]) if country == "UK" else random.choice(["Belmont", "Gulfstream", "Saratoga", "Del Mar"])
            odds = round(np.random.uniform(2, 8), 2)
            win_val = round(np.random.uniform(5, 25), 1)
            place_val = round(win_val * 0.6, 1)
            pred_win = round(100 / odds, 1)
            pred_place = round(pred_win * 0.6, 1)
            bedge_win = round(pred_win * 0.6 + win_val * 0.4, 1)
            bedge_place = round(pred_place * 0.6 + place_val * 0.4, 1)
            horses.append({
                "Horse": name,
                "Course": course,
                "Best Odds": odds,
                "Win_Value": win_val,
                "Place_Value": place_val,
                "Predicted Win %": pred_win,
                "Predicted Place %": pred_place,
                "BetEdge Win %": bedge_win,
                "BetEdge Place %": bedge_place
            })
        return pd.DataFrame(horses)

    df = generate_horse_data(country)
    df = df[df["BetEdge Win %"] >= filter_value]

    if df.empty:
        st.warning("No races match your criteria.")
    else:
        df_win = df.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
        df_place = df.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

        tab1, tab2 = st.tabs(["📊 Charts", "📋 Tables"])

        with tab1:
            st.subheader("Top 20 Horses – BetEdge Win %")
            st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])
            st.subheader("Top 20 Horses – BetEdge Place %")
            st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])

        with tab2:
            st.subheader("🏆 Win Rankings")
            st.dataframe(df_win.style.applymap(color_val, subset=["BetEdge Win %"]))
            st.subheader("🥈 Place Rankings")
            st.dataframe(df_place.style.applymap(color_val, subset=["BetEdge Place %"]))

# --- Football Section ---
elif sport_tab == "Football":
    st.title("⚽ BetEdge – Football Value Tracker")

    league = st.selectbox("Select League", ["Premier League", "Championship", "La Liga", "MLS"])
    min_edge = st.slider("Minimum Value Edge %", 0, 100, 10)

    def generate_football_data():
        matches = []
        for i in range(30):
            home = f"Team {chr(65+i)}"
            away = f"Team {chr(90-i)}"
            odds = round(np.random.uniform(1.5, 3.5), 2)
            pred_win = round(100 / odds, 1)
            edge = round(pred_win * 0.7 + np.random.uniform(0, 15), 1)
            matches.append({
                "Match": f"{home} vs {away}",
                "Odds": odds,
                "Predicted Win %": pred_win,
                "Value Edge %": edge
            })
        return pd.DataFrame(matches)

    df = generate_football_data()
    df = df[df["Value Edge %"] >= min_edge]

    if df.empty:
        st.warning("No matches with enough edge.")
    else:
        df_sorted = df.sort_values("Value Edge %", ascending=False).reset_index(drop=True)

        tab1, tab2 = st.tabs(["📊 Charts", "📋 Tables"])

        with tab1:
            st.subheader("Top 20 Matches – Value Edge %")
            st.bar_chart(df_sorted.head(20).set_index("Match")["Value Edge %"])

        with tab2:
            st.subheader("📋 Match Rankings")
            st.dataframe(df_sorted.style.applymap(color_val, subset=["Value Edge %"]))
