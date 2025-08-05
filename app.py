import streamlit as st
import pandas as pd
import os
from merge_odds import merge_best_prices

CACHE_FILE = "odds_cache.csv"

# --- PAGE CONFIG ---
st.set_page_config(page_title="BetEdge – Global Racing Value Finder", layout="wide")
st.title("🌍 BetEdge – Global Horse Racing Value Finder")
st.markdown("_Live UK 🇬🇧, AUS 🇦🇺, and USA 🇺🇸 horse racing odds with Best Price EV calculations._")

# --- SIDEBAR FILTERS ---
country_filter = st.sidebar.multiselect(
    "🌐 Select Countries",
    ["UK", "AUS", "USA"],
    default=["UK", "AUS", "USA"]
)

ev_threshold = st.sidebar.slider("📈 Minimum Value Score (EV)", 0.0, 0.2, 0.05, 0.01)
min_odds = st.sidebar.slider("🎯 Minimum Odds", 1.0, 50.0, 2.0, 0.5)
max_odds = st.sidebar.slider("🎯 Maximum Odds", 1.0, 50.0, 20.0, 0.5)

# --- REFRESH BUTTON ---
if st.button("🔄 Refresh Data"):
    df = merge_best_prices()

    if df.empty:
        st.error("No race data found.")
    else:
        # Apply filters
        df = df[df["Country"].isin(country_filter)]
        df = df[(df["Value Score"] >= ev_threshold) &
                (df["Best Price"] >= min_odds) &
                (df["Best Price"] <= max_odds)]

        # Tooltip for Best Price (UK = RP + TF sources)
        if "Odds_RP" in df.columns or "Odds_TF" in df.columns:
            df["Best Price (Tooltip)"] = df.apply(
                lambda row: f"{row['Best Price']} (RP: {row.get('Odds_RP', 'N/A')}, TF: {row.get('Odds_TF', 'N/A')})"
                if row["Country"] == "UK" else f"{row['Best Price']} (Single Source)",
                axis=1
            )
        else:
            df["Best Price (Tooltip)"] = df["Best Price"]

        # Confidence Bar
        def confidence_bar(ev):
            if ev >= 0.1:
                return "🟩 High"
            elif ev >= 0.05:
                return "🟨 Medium"
            else:
                return "🟥 Low"
        df["Confidence"] = df["Value Score"].apply(confidence_bar)

        # Market Mover Tag
        if os.path.exists(CACHE_FILE):
            old_df = pd.read_csv(CACHE_FILE)
            merged = pd.merge(
                df, old_df,
                on=["Horse", "Race URL", "Country"],
                how="left",
                suffixes=("", "_old")
            )
            def movement(row):
                if pd.isna(row.get("Best Price_old")):
                    return "➖ New"
                if row["Best Price"] < row["Best Price_old"]:
                    return "🟢 Steamer"
                elif row["Best Price"] > row["Best Price_old"]:
                    return "🔴 Drifter"
                else:
                    return "➖ No change"
            df["Market Mover"] = merged.apply(movement, axis=1)
        else:
            df["Market Mover"] = "➖ New"

        # Save to cache
        df.to_csv(CACHE_FILE, index=False)

        # EV colour formatting
        def ev_colour(val):
            if val >= 0.1:
                return 'color: green'
            elif val >= 0.05:
                return 'color: orange'
            else:
                return 'color: red'

        # Display DataFrame
        display_cols = [
            "Country", "Race Time", "Countdown", "Horse",
            "Best Price (Tooltip)", "Market Prob", "Model Prob",
            "Value Score", "Confidence", "Market Mover", "Race URL"
        ]
        st.dataframe(
            df[display_cols].style.format({
                "Best Price (Tooltip)": "{:.2f}",
                "Market Prob": "{:.3f}",
                "Model Prob": "{:.3f}",
                "Value Score": "{:.3f}"
            }).applymap(ev_colour, subset=["Value Score"])
        )
