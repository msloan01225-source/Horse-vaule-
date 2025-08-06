import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta

def get_racingpost_data(day="Today", debug=False):
    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        races = []

        for tag in soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$")):
            race_time = tag.strip()
            course_tag = tag.find_next(string=True)
            course = course_tag.strip() if course_tag else "Unknown"
            race_name_tag = tag.find_next("a")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Race"
            races.append({
                "Race": f"{course} {race_time} – {race_name}",
                "Time": race_time,
                "Course": course,
                "RaceName": race_name
            })

        if debug:
            print(f"[Racing Post] Found {len(races)} races")
        return pd.DataFrame(races)

    except Exception as e:
        if debug:
            print(f"[Racing Post] Error: {e}")
        return pd.DataFrame()
