import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re

def get_timeform_data(day="Today", use_proxies=False, debug=False):
    """
    Scraper for Timeform racecards page.
    Returns a DataFrame with Time, Course, Race Name, and Runners.
    """
    base_url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    proxies = None
    if use_proxies:
        try:
            with open("proxies.txt", "r") as f:
                proxy_list = [p.strip() for p in f if p.strip()]
            if proxy_list:
                proxies = {"http": proxy_list[0], "https": proxy_list[0]}
        except FileNotFoundError:
            pass

    try:
        r = requests.get(base_url, proxies=proxies, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        if debug:
            with open(f"timeform_racecards_{day}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

        soup = BeautifulSoup(r.text, "html.parser")
        races = []

        # Timeform racecards have times in elements with class containing 'time'
        for race_card in soup.select(".racecard"):
            # Extract time
            time_tag = race_card.select_one(".time")
            race_time = time_tag.get_text(strip=True) if time_tag else "?"

            # Extract course name
            course_tag = race_card.select_one(".course")
            course = course_tag.get_text(strip=True) if course_tag else "Unknown"

            # Extract race title
            race_name_tag = race_card.select_one(".race-title")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Race"

            # Extract runners
            runners_tag = race_card.find(string=re.compile(r"runners?"))
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
        print(f"[Timeform] ❌ Error fetching Timeform: {e}")
        return pd.DataFrame()
