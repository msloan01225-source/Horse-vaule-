import streamlit as st
import pandas as pd
from racingpost_scraper import get_racingpost_data
from timeform_scraper import get_timeform_data
import numpy as np

st.set_page_config(page_title="BetEdge Value Finder", layout="wide")
st.title("🏇 BetEdge Horse Racing Value Finder")

race_day = st.selectbox("Select Day", ["Today", "Tomorrow"])
debug_mode = st.checkbox("Debug Mode", value=False)

@st.cache_data(ttl=300)
def load_data(day):
    rp_df = get_racingpost_data(day, debug=debug_mode)
    tf_df = get_timeform_data(day, debug=debug_mode)
    return rp_df, tf_df

rp_df, tf_df = load_data(race_day)

if rp_df.empty:
    st.error("No Racing Post data found.")
if tf_df.empty:
    st.error("No Timeform data found.")

if not rp_df.empty and not tf_df.empty:
    # Merge on Race name
    merged_df = pd.merge(rp_df, tf_df, on="Race", how="inner")

    if merged_df.empty:
        st.warning("No matching races found between Racing Post and Timeform.")
    else:
        # Add demo odds & win % columns for value calculation
        np.random.seed(42)
        merged_df["Best Odds"] = (np.random.uniform(1.5, 5.0, size=len(merged_df))).round(2)
        merged_df["Estimated Win %"] = (np.random.uniform(10, 70, size=len(merged_df))).round(1)

        # Calculate Value = (Odds * Win%) - 100
        merged_df["Value"] = (merged_df["Best Odds"] * merged_df["Estimated Win %"]) - 100

        # Sort by value descending
        merged_df = merged_df.sort_values("Value", ascending=False)

        # Highlight Value column with colors
        def highlight_value(val):
            if val > 20:
                return 'background-color: #85e085'  # green
            elif val > 0:
                return 'background-color: #ffff99'  # yellow
            else:
                return 'background-color: #f7b2b2'  # red

        st.write(f"### {len(merged_df)} Races Ranked by Value")
        st.dataframe(merged_df.style.applymap(highlight_value, subset=["Value"]))
else:
    st.info("Waiting for data...")
