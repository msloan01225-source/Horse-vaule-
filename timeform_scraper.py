import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta

def get_timeform_data(day="Today", debug=False):
    """Scrape Timeform race data without Selenium."""
    
    base_url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        races = []
        for race_block in soup.find_all("div", class_="race-block"):
            time_tag = race_block.find("span", class_="time")
            if not time_tag:
                continue
            race_time = time_tag.get_text(strip=True)

            course_tag = race_block.find("span", class_="course")
            course = course_tag.get_text(strip=True) if course_tag else "Unknown"

            horse_tags = race_block.find_all("span", class_="horse-name")
            for horse_tag in horse_tags:
                horse_name = horse_tag.get_text(strip=True)
                # Placeholder win probability — real logic would parse ratings
                races.append({
                    "Race": f"{course} {race_time}",
                    "Time": race_time,
                    "Course": course,
                    "Horse": horse_name,
                    "Win Probability": 0.05  # placeholder, replaced in merge
                })

        return pd.DataFrame(races)

    except Exception as e
