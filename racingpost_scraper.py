import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def get_racingpost_data(day="Today", use_proxies=False, debug=False):
    base_url = "https://www.racingpost.com/racecards/time-order/"
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
            with open(f"racingpost_debug_{day}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

        soup = BeautifulSoup(r.text, "html.parser")
        races = []

        race_blocks = soup.find_all("a", class_="RC-meetingItem")
        for race in race_blocks:
            race_name = race.get_text(strip=True)
            race_url = race["href"]

            # Example horse data — in reality you'd parse the horse table on race_url
            races.append({"Race": race_name, "Horse": "Example Horse", "Odds": 5.0, "Best Bookmaker": "Bet365"})

        return pd.DataFrame(races)

    except Exception as e:
        if debug:
            print(f"[RacingPost] Error: {e}")
        return pd.DataFrame()
