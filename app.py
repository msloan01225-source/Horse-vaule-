import streamlit as st
import pandas as pd
from merge_odds import merge_best_prices

st.set_page_config(page_title="BetEdge – Value Finder", layout="wide")

# --- Sidebar Controls ---
st.sidebar.title("BetEdge Controls")
race_day = st.sidebar.selectbox("Select Day", ["Today", "Tomorrow"])
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# --- Main App ---
st.title("🏇 BetEdge – Horse Racing Value Tracker")
st.caption("Live merged odds from Racing Post & Timeform – ranked by win value")

with st.spinner(f"Fetching {race_day} race data..."):
    df = merge_best_prices(day=race_day, debug=debug_mode)

if df.empty:
    st.error("No data available – please try again later or enable Debug Mode.")
else:
    # Example "value score" – can replace with your own algorithm
    df["Value_Score"] = pd.to_numeric(df.get("WinPrice_RP", 0), errors="coerce") - pd.to_numeric(df.get("WinPrice_TF", 0), errors="coerce")
    df = df.sort_values(by="Value_Score", ascending=False)

    # Display table
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"betedge_{race_day.lower()}_value.csv",
        mime="text/csv"
    )

st.success("✅ Data load complete!")
