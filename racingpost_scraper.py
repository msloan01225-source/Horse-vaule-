import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta

def get_racingpost_data(day="Today", debug=False):
    """Scrape Racing Post racecard data without Selenium."""
    
    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        races = []
        for race_block in soup.find_all("a", href=re.compile(r"/racecards/")):
            race_text = race_block.get_text(strip=True)
            time_match = re.match(r"(\d{1,2}:\d{2})", race_text)
            if time_match:
                race_time = time_match.group(1)
                course = race_block.find_previous("h2").get_text(strip=True) if race_block.find_previous("h2") else "Unknown"
                race_name = race_text.replace(race_time, "").strip()

                races.append({
                    "Race": f"{course} {race_time} - {race_name}",
                    "Time": race_time,
                    "Course": course,
                    "Horse": race_name,
                    "Win Probability": 0.0  # placeholder, replaced after merge
                })

        return pd.DataFrame(races)

    except Exception as e:
        if debug:
            print(f"[RacingPost] Error: {e}")
        return pd.DataFrame()
