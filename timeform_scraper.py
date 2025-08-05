import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re

def get_timeform_data(day="Today", use_proxies=False, debug=False):
    """
    Timeform scraper for BetEdge – UK race listings.
    Scrapes Race Time, Course, Race Name, and Runners from the Timeform racecards page.
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
        print(f"[Timeform] 📡 Fetching racecards for {day} from {base_url}")
        r = requests.get(base_url, proxies=proxies, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        if debug:
            with open(f"timeform_racecards_{day}.html", "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"[Timeform] 📝 Saved debug HTML to timeform_racecards_{day}.html")

        soup = BeautifulSoup(r.text, "html.parser")
        races = []

        # Look for race blocks
        race_blocks = soup.find_all("div", class_=re.compile(r"racecard"))
        for race in race_blocks:
            time_tag = race.find(string=re.compile(r"^\d{1,2}:\d{2}$"))
            race_time = time_tag.strip() if time_tag else "?"

            course_tag = race.find("a", href=re.compile(r"/horse-racing/venue/"))
            course = course_tag.get_text(strip=True) if course_tag else "Unknown"

            race_name_tag = race.find("a", href=re.compile(r"/horse-racing/"))
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Race"

            runners_tag = race.find(string=re.compile(r"\d+\s+runners"))
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
