import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re

def get_racingpost_data(day="Today", debug=False):
    """
    Scrapes Racing Post time-order page to get race info and odds.
    """
    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()

        if debug:
            with open(f"racingpost_{day}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

        soup = BeautifulSoup(r.text, "html.parser")
        races = []

        # Find all race blocks
        race_blocks = soup.find_all("li", class_=re.compile("RC-runnerRow"))
        for block in race_blocks:
            horse_name_tag = block.find("a", class_=re.compile("RC-runnerName"))
            odds_tag = block.find("span", class_=re.compile("RC-price"))

            if not horse_name_tag or not odds_tag:
                continue

            horse_name = horse_name_tag.get_text(strip=True)
            odds_str = odds_tag.get_text(strip=True)

            # Convert fractional odds to decimal
            try:
                if "/" in odds_str:
                    num, denom = odds_str.split("/")
                    odds = round((int(num) / int(denom)) + 1, 2)
                else:
                    odds = float(odds_str)
            except:
                continue

            race_title_tag = block.find_previous("a", class_=re.compile("RC-courseLink"))
            race_title = race_title_tag.get_text(strip=True) if race_title_tag else "Unknown Race"

            races.append({
                "Horse": horse_name,
                "Race": race_title,
                "Odds_RP": odds
            })

        if debug:
            print(f"[RacingPost] ✅ Found {len(races)} horses for {day}")
        return pd.DataFrame(races)

    except Exception as e:
        if debug:
            print(f"[RacingPost] ❌ Error: {e}")
        return pd.DataFrame()
