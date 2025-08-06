import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re

def get_all_todays_data(day="Today", debug=False):
    """
    Timeform scraper – Returns DataFrame with Race, Time, Course, Runners
    """
    base_url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()

        if debug:
            with open(f"timeform_racecards_{day}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

        soup = BeautifulSoup(r.text, "html.parser")
        races = []

        # Find all times
        time_tags = soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$"))

        for t in time_tags:
            race_time = t.strip()

            # Course
            course_tag = t.find_next(string=True)
            course = course_tag.strip() if course_tag else "Unknown"

            # Race name
            race_name_tag = t.find_next("a")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Race"

            # Runners
            runners_tag = t.find_next(string=re.compile(r"\d+\s+runners"))
            runners = runners_tag.strip() if runners_tag else "?"

            races.append({
                "Race": f"{course} {race_time} - {race_name}",
                "Time": race_time,
                "Course": course,
                "Runners": runners
            })

        print(f"[Timeform] ✅ Found {len(races)} races for {day}")
        return pd.DataFrame(races)

    except Exception as e:
        print(f"[Timeform] ❌ Error: {e}")
        return pd.DataFrame()
