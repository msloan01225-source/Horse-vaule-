import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re

def get_racingpost_data(day="Today", use_proxies=False, debug=False):
    """
    Racing Post scraper for BetEdge – UK race listings.
    Scrapes Race Time, Course, Race Name, and Runners from the time-order page.
    """

    # Build URL for today or tomorrow
    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    # Proxy config
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
        r = requests.get(base_url, proxies=proxies, timeout=15)
        r.raise_for_status()

        if debug:
            with open(f"racingpost_timeorder_{day}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

        soup = BeautifulSoup(r.text, "html.parser")

        races = []
        # Match times like 2:15, 14:30, etc.
        time_tags = soup.find_all(string=re.compile(r"^\d{1,2}:\d{2}$"))

        for t in time_tags:
            race_time = t.strip()

            # Course name is usually next visible text after time
            course_tag = t.find_next(string=True)
            course = course_tag.strip() if course_tag else "Unknown"

            # Race title is usually after course
            race_name_tag = t.find_next("a")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Race"

            # Runners info
            runners_tag = t.find_next(string=re.compile(r"\d+\s+runners"))
            runners = runners_tag.strip() if runners_tag else "?"

            races.append({
                "Race": f"{course} {race_time} - {race_name}",
                "Time": race_time,
                "Course": course,
                "Runners": runners
            })

        print(f"[RacingPost] ✅ Found {len(races)} races for {day}")
        return pd.DataFrame(races)

    except Exception as e:
        if debug:
            print(f"[RacingPost] ❌ Error fetching Racing Post: {e}")
        return pd.DataFrame()
