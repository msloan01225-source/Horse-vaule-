import streamlit as st
import pandas as pd
from scraper import get_today_race_urls, get_race_data

st.set_page_config(page_title="Horse Value Finder", page_icon="🐎", layout="wide")

st.title("🐎 Horse Value Finder")

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Racehorse.jpg/640px-Racehorse.jpg",
    use_column_width=True,
    caption="Horse racing data powered by Racing Post"
)

st.markdown("""
Welcome! This app scans today's UK horse races and finds **value horses** by comparing exchange odds against expected probabilities.

- 📊 Ranks all horses by value score  
- 🔍 Filter by odds or value score  
- 🏆 Displays the best value horses of the day  

Click below to scan and find today's best picks:
""")

# Sidebar filters
st.sidebar.header("🔍 Filters")
min_score = st.sidebar.slider("Minimum Value Score", 0.00, 0.20, 0.05, 0.01)
odds_range = st.sidebar.slider("Exchange Odds Range", 1.0, 50.0, (2.0, 15.0))

# Main button
if st.button("📊 Scan Today's Races"):
    with st.spinner("Scanning Racing Post..."):
        urls = get_today_race_urls()
        all_data = []

        for url in urls:
            try:
                df = get_race_data(url)
                if not df.empty:
                    num_runners = len(df)
                    df["Market Prob"] = 1 / df["Exchange Odds"]
                    df["Model Prob"] = 1 / num_runners
                    df["Value Score"] = df["Model Prob"] - df["Market Prob"]
                    all_data.append(df)
            except Exception as e:
                st.warning(f"⚠️ Failed to load a race: {e}")
                continue

        if all_data:
            full_df = pd.concat(all_data)

            # Apply filters
            filtered = full_df[
                (full_df["Exchange Odds"].between(odds_range[0], odds_range[1])) &
                (full_df["Value Score"] >= min_score)
            ].copy()

            filtered.sort_values("Value Score", ascending=False, inplace=True)

            # Format
            filtered["Exchange Odds"] = filtered["Exchange Odds"].round(2)
            filtered["Market Prob"] = filtered["Market Prob"].round(3)
            filtered["Model Prob"] = filtered["Model Prob"].round(3)
            filtered["Value Score"] = filtered["Value Score"].round(3)

            st.success(f"✅ Found {len(filtered)} value horses today")
            st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

            # Best of the day
            top_horses = filtered.head(5)
            if not top_horses.empty:
                st.subheader("🏆 Best Value Horses Today")
                st.table(top_horses[["Horse", "Race", "Exchange Odds", "Value Score"]])
        else:
            st.error("No valid races found.")
else:
    st.info("Click the button above to scan today's races.")

st.markdown("---")
st.markdown("Created with ❤️ by [YourName]. Data from Racing Post.")
