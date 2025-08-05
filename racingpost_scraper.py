import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re
import time

def get_racingpost_data(day="Today", use_proxies=False, debug=False):
    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        base_url += "?date=" + (datetime.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

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
        race_links = [a["href"] for a in soup.select("a.rc-meeting-item__link") if a.get("href")]

        all_rows = []
        for link in race_links:
            race_url = "https://www.racingpost.com" + link
            try:
                rr = requests.get(race_url, proxies=proxies, timeout=15)
                rr.raise_for_status()
                if debug:
                    with open(f"racingpost_race_{re.sub('[^0-9a-zA-Z]+', '_', link)}.html", "w", encoding="utf-8") as f:
                        f.write(rr.text)

                rsoup = BeautifulSoup(rr.text, "html.parser")

                # Race name & time
                meeting = rsoup.select_one(".RC-header__raceInstanceTitle").get_text(strip=True) if rsoup.select_one(".RC-header__raceInstanceTitle") else "Unknown Meeting"

                # Horses
                horse_rows = rsoup.select(".RC-runnerRow")
                for h in horse_rows:
                    horse_name = h.select_one(".RC-runnerName").get_text(strip=True) if h.select_one(".RC-runnerName") else None
                    odds_text = h.select_one(".RC-runnerPrice").get_text(strip=True) if h.select_one(".RC-runnerPrice") else None
                    bookmaker = "Unknown"
                    if odds_text:
                        # Convert fractional odds to decimal
                        try:
                            if "/" in odds_text:
                                num, denom = odds_text.split("/")
                                odds = round(1 + float(num) / float(denom), 2)
                            else:
                                odds = float(odds_text)
                        except:
                            odds = None
                    else:
                        odds = None

                    if horse_name and odds:
                        all_rows.append({
                            "Race": meeting,
                            "Horse": horse_name,
                            "Odds": odds,
                            "Best Bookmaker": bookmaker
                        })

                time.sleep(1)  # Be polite

            except Exception as e:
                if debug:
                    print(f"[RacingPost] Error scraping {race_url}: {e}")
                continue

        return pd.DataFrame(all_rows)

    except Exception as e:
        if debug:
            print(f"[RacingPost] Failed to fetch time-order page: {e}")
        return pd.DataFrame()
