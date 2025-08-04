
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

BASE_URL = "https://www.racingpost.com"

def get_today_race_urls():
    today = datetime.today().strftime('%Y-%m-%d')
    racecards_url = f"{BASE_URL}/racecards/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(racecards_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/racecards/" in href and today in href and "/results/" not in href:
            full_url = BASE_URL + href.split("?")[0]
            if full_url not in links:
                links.append(full_url)
    return links

def get_race_data(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    horses = []
    runners = soup.find_all("div", class_="RC-runnerRow")

    # Extract track and time for later tagging
    header = soup.find("span", class_="RC-header__raceInstanceTitle")
    time_venue = header.get_text(strip=True) if header else "Unknown Race"

    for runner in runners:
        name_tag = runner.find("a", class_="RC-runnerName")
        odds_tag = runner.find("span", class_="RC-runnerPrice")

        if name_tag and odds_tag:
            name = name_tag.get_text(strip=True)
            odds_str = odds_tag.get_text(strip=True).replace("/", ":")
            try:
                if ":" in odds_str:
                    num, den = map(float, odds_str.split(":"))
                    decimal_odds = (num / den) + 1
                else:
                    decimal_odds = float(odds_str)
                horses.append({
                    "Race": time_venue,
                    "Horse": name,
                    "Exchange Odds": round(decimal_odds, 2)
                })
            except:
                continue

    return pd.DataFrame(horses)
