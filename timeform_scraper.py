import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re

def get_timeform_data(day="Today", debug=False):
    """
    Scrapes Timeform racecards for Race, Horse, Win odds, Place odds.
    """
    base_url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()

        if debug:
            with open(f"timeform_{day}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

        soup = BeautifulSoup(r.text, "html.parser")
        races = []

        # Each race card block
        race_blocks = soup.find_all("div", class_=re.compile("race-card"))

        for race in race_blocks:
            # Race time
            race_time_tag = race.find(string=re.compile(r"\d{1,2}:\d{2}"))
            race_time = race_time_tag.strip() if race_time_tag else "??:??"

            # Race name
            race_name_tag = race.find("h2")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Unknown Race"

            # Horse names
            horse_tags = race.find_all("a", class_=re.compile("horse-name"))
            for horse_tag in horse_tags:
                horse_name = horse_tag.get_text(strip=True)

                # Odds placeholders (can be filled in with bookmaker odds later)
                win_odds = None
                place_odds = None

                races.append({
                    "Race": f"{race_time} {race_name}",
                    "Horse": horse_name,
                    "Win_TF": win_odds,
                    "Place_TF": place_odds
                })

        return pd.DataFrame(races)

    except Exception as e:
        print(f"[Timeform] Error: {e}")
        return pd.DataFrame()
