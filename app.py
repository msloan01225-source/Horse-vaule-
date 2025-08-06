import streamlit as st
import pandas as pd
from datetime import datetime
import logging
from merge_odds import merge_best_prices

# ——— Streamlit page config ———
st.set_page_config(page_title="BetEdge Horse Value Tracker", layout="wide")

# ——— Logger setup ———
logger = logging.getLogger("betedge")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

# ——— Data loading with debug logging ———
@st.cache_data(ttl=600)
def load_data():
    try:
        df = merge_best_prices()  # expects columns: Course, Race Time, Horse, Win Value, Place Value
        logger.info(f"Loaded {len(df)} rows from merge_best_prices()")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        st.error("⚠️ Failed to fetch live data. Check logs for details.")
        return pd.DataFrame(columns=["Course", "Race Time", "Horse", "Win Value", "Place Value"])

# ——— Colour-coding function ———
def colour_value(v):
    if v >= 25:
        return "background-color: #a8e6a1"  # strong (green)
    elif v >= 10:
        return "background-color: #fff3b0"  # medium (yellow)
    else:
        return "background-color: #f4cccc"  # low (red)

# ——— Main app UI ———
st.title("🐎 BetEdge – Premium Horse Value Tracker")

# Last updated timestamp
last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last Updated: {last_updated}")

# Load and prepare data
df = load_data()
if df.empty:
    st.stop()

# Win Value table
st.subheader("🏆 Win Value Rankings")
df_win = (
    df[["Course", "Race Time", "Horse", "Win Value"]]
    .sort_values(by="Win Value", ascending=False)
    .reset_index(drop=True)
)
styled_win = (
    df_win.style
    .applymap(colour_value, subset=["Win Value"])
    .format({"Win Value": "{:.1f}%"})
    .set_properties(subset=["Course", "Race Time", "Horse"], **{"text-align": "left"})
)
st.dataframe(styled_win, use_container_width=True)

# Place Value table
st.subheader("🥈 Place Value Rankings")
df_place = (
    df[["Course", "Race Time", "Horse", "Place Value"]]
    .sort_values(by="Place Value", ascending=False)
    .reset_index(drop=True)
)
styled_place = (
    df_place.style
    .applymap(colour_value, subset=["Place Value"])
    .format({"Place Value": "{:.1f}%"})
    .set_properties(subset=["Course", "Race Time", "Horse"], **{"text-align": "left"})
)
st.dataframe(styled_place, use_container_width=True)
