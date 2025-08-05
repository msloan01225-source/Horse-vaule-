import requests
import random
import time
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://www.racingpost.com"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"
]

def fetch(url):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-GB,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    time.sleep(random.uniform(1, 2))
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r

def get_todays_races():
    url = f"{BASE_URL}/racecards/time-order/"
    r = fetch(url)
    soup = BeautifulSoup(r.text, "html.parser")
    races = []
    # New race link selector
    for link in soup.select("a.rc-cardHeader__link"):
        race_url = BASE_URL + link.get("href")
        race_time = link.get_text(strip=True).split()[0]
        races.append({"url": race_url, "Race Time": race_time})
    return races

def get_race_data(race_url, race_time):
    r = fetch(race_url)
    soup = BeautifulSoup(r.text, "html.parser")
    data = []
    # Updated selectors for new table layout
    for row in soup.select("tr.rc-cardMain__runner"):
        horse_tag = row.select_one(".rc-cardMain__horseName a")
        odds_tag = row.select_one(".rc-cardMain__price span")
        if not horse_tag or not odds_tag:
            continue
        horse = horse_tag.get_text(strip=True)
        odds_text = odds_tag.get_text(strip=True)
        try:
            if "/" in odds_text:
                num, den = odds_text.split("/")
                odds = round(1 + (float(num) / float(den)), 2)
            else:
                odds = float(odds_text)
        except:
            continue
        data.append({"Horse": horse, "Odds_RP": odds, "Race Time": race_time})

    num_runners = len(data)
    for d in data:
        market_prob = 1 / d["Odds_RP"]
        model_prob = 1 / num_runners if num_runners else 0
        d["Market Prob"] = round(market_prob, 4)
        d["Model Prob"] = round(model_prob, 4)
        d["Value Score"] = round(model_prob - market_prob, 4)
        d["Race URL"] = race_url
        d["Country"] = "UK"

    return pd.DataFrame(data)

def get_all_todays_data():
    races = get_todays_races()
    all_data = []
    for race in races:
        df = get_race_data(race["url"], race["Race Time"])
        if not df.empty:
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()
