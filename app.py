import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="BetEdge Sports Tracker", layout="wide")
st.title("🎯 BetEdge – Smart Sports Value Tracker")

# --- Top Menu (Sport Selection) ---
sport = st.sidebar.selectbox("Select Sport", ["🏇 Horse Racing", "⚽ Football"])

# ========== HORSE RACING SECTION ==========
if sport == "🏇 Horse Racing":
    st.header("🏇 Horse Racing – BetEdge Insights")

    @st.cache_data(ttl=600)
    def generate_horse_data(n=60):
        np.random.seed(42)
        courses = ["Ascot", "York", "Newmarket", "Cheltenham", "Aintree", "Sandown"]
        bookies = ["Bet365", "SkyBet", "William Hill", "PaddyPower"]
        track_types = ["Flat", "Jump"]
        race_classes = ["Class 1", "Class 2", "Class 3", "Class 4"]

        data = []
        for i in range(n):
            course = np.random.choice(courses)
            horse = f"Horse {i+1}"
            bookie = np.random.choice(bookies)
            track = np.random.choice(track_types)
            race_class = np.random.choice(race_classes)
            form = "".join(np.random.choice(list("1234567890-"), size=5))
            best_odds = round(np.random.uniform(2, 15), 2)
            win_val = round(np.random.uniform(0, 25), 1)
            place_val = round(win_val * np.random.uniform(0.5, 0.8), 1)
            pred_win = round(100 / best_odds, 1)
            pred_place = round(pred_win * 0.6, 1)
            betedge_win = round((pred_win * 0.6 + win_val * 0.4), 1)
            betedge_place = round((pred_place * 0.6 + place_val * 0.4), 1)

            data.append({
                "Time": (datetime.now() + timedelta(minutes=30 * i)).strftime("%H:%M"),
                "Course": course,
                "Horse": horse,
                "Form": form,
                "Bookie": bookie,
                "Track": track,
                "Race Class": race_class,
                "Best Odds": best_odds,
                "Win_Value": win_val,
                "Place_Value": place_val,
                "Predicted Win %": pred_win,
                "Predicted Place %": pred_place,
                "BetEdge Win %": betedge_win,
                "BetEdge Place %": betedge_place
            })

        return pd.DataFrame(data)

    df = generate_horse_data()

    # --- Filters ---
    st.sidebar.header("🎯 Filters (Horse Racing)")
    selected_course = st.sidebar.multiselect("Course", sorted(df["Course"].unique()), default=df["Course"].unique())
    selected_bookie = st.sidebar.multiselect("Bookie", sorted(df["Bookie"].unique()), default=df["Bookie"].unique())
    selected_track = st.sidebar.multiselect("Track Type", sorted(df["Track"].unique()), default=df["Track"].unique())
    selected_class = st.sidebar.multiselect("Race Class", sorted(df["Race Class"].unique()), default=df["Race Class"].unique())

    df_filtered = df[
        df["Course"].isin(selected_course) &
        df["Bookie"].isin(selected_bookie) &
        df["Track"].isin(selected_track) &
        df["Race Class"].isin(selected_class)
    ]

    # --- Output ---
    if df_filtered.empty:
        st.warning("No races match your filters.")
    else:
        view = st.radio("Choose View Mode", ["📈 Charts", "📋 Tables"], horizontal=True)
        st.caption(f"Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

        df_win = df_filtered.sort_values("BetEdge Win %", ascending=False).reset_index(drop=True)
        df_place = df_filtered.sort_values("BetEdge Place %", ascending=False).reset_index(drop=True)

        if view == "📋 Tables":
            st.subheader("🏆 Win Rankings")
            st.dataframe(df_win)

            st.subheader("🥈 Place Rankings")
            st.dataframe(df_place)

        else:
            st.subheader("📊 Top 20 BetEdge Win %")
            st.bar_chart(df_win.head(20).set_index("Horse")["BetEdge Win %"])

            st.subheader("📊 Top 20 BetEdge Place %")
            st.bar_chart(df_place.head(20).set_index("Horse")["BetEdge Place %"])

# ========== FOOTBALL SECTION ==========
elif sport == "⚽ Football":
    st.header("⚽ Football – Match Value Tracker (Mock Data)")
    
    @st.cache_data(ttl=600)
    def generate_football_data():
        leagues = ["Premier League", "Championship", "La Liga", "Bundesliga"]
        teams = [("Arsenal", "Chelsea"), ("Man Utd", "Liverpool"), ("Leeds", "Leicester"), ("Barcelona", "Real Madrid"), ("Dortmund", "Bayern")]

        data = []
        for i, (home, away) in enumerate(teams):
            league = np.random.choice(leagues)
            odds_home = round(np.random.uniform(1.5, 3.5), 2)
            odds_draw = round(np.random.uniform(2.5, 4.0), 2)
            odds_away = round(np.random.uniform(2.0, 4.0), 2)
            edge_home = round((100 / odds_home) - 100 + np.random.uniform(-5, 5), 1)
            edge_away = round((100 / odds_away) - 100 + np.random.uniform(-5, 5), 1)

            data.append({
                "League": league,
                "Home Team": home,
                "Away Team": away,
                "Odds (H)": odds_home,
                "Odds (D)": odds_draw,
                "Odds (A)": odds_away,
                "Value Edge Home %": edge_home,
                "Value Edge Away %": edge_away
            })

        return pd.DataFrame(data)

    df_foot = generate_football_data()
    st.dataframe(df_foot)

    st.subheader("📊 Top Value Opportunities")
    st.write(df_foot.sort_values("Value Edge Home %", ascending=False).head(5))
