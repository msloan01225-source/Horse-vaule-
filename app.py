
import streamlit as st
import pandas as pd
from scraper import get_race_data

# Title and description
st.title("🐎 Horse Racing Value Finder")
st.write("Paste a Racing Post race URL below. This app finds horses with the best value based on model vs market odds.")

# Input for race URL
url = st.text_input("Enter Racing Post Race URL", "")

if url:
    try:
        # Get data from scraper
        df = get_race_data(url)

        if df.empty:
            st.warning("No runners found. Check the URL and try again.")
        else:
            # For now, mock model probabilities
            df["Model Prob"] = [0.25 - (i * 0.03) for i in range(len(df))]  # dummy values
            df["Market Prob"] = 1 / df["Exchange Odds"]
            df["Value Score"] = df["Model Prob"] - df["Market Prob"]
            df = df.sort_values("Value Score", ascending=False)

            st.dataframe(df.style.highlight_max(axis=0, subset=["Value Score"]))
    except Exception as e:
        st.error(f"Error fetching data: {e}")
