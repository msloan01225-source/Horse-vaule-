import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time

def get_racingpost_data(day="Today", use_proxies=False, debug=False):
    """
    Scrapes Racing Post time-order page for Today or Tomorrow.
    Returns a DataFrame with Race, Time, Course, Runners.
    """

    # Build URL based on day
    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    # Load proxies if enabled
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
        # Fetch the page
        r = requests.get(base_url, proxies=proxies, timeout=15)
        r.raise_for_status()

        # Save debug HTML if enabled
        if debug:
            with open(f"racingpost_timeorder_{day}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

        soup = BeautifulSoup(r.text, "html.parser")

        # The races are structured in repeating course/time blocks
        races = []
        time_tags = soup.find_all("span", class_="time")

        for t in time_tags:
            race_time = t.get_text(strip=True)
            course_tag = t.find_next("span", class_="rc-course-name")
            race_name_tag = t.find_next("a", class_="RC-courseHeader__link")

            course = course_tag.get_text(strip=True) if course_tag else "Unknown"
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Race"

            # Number of runners
            runners_tag = t.find_next("span", class_="RC-cardHeader__runners")
            if runners_tag:
                runners = runners_tag.get_text(strip=True).replace(" runners", "")
            else:
                runners = "?"

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
