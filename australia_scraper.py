import requests
import random
import time
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

BASE_URL = "https://www.racing.com"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"
]

def fetch(url):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-AU,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    time.sleep(random.uniform(1, 3))
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r

def get_todays_races():
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/racecards/{today}"
    r = fetch(url)
    soup = BeautifulSoup(r.text, "html.parser")
    races = []
    for link in soup.select("a.meeting-race__race"):
        race_url = BASE_URL + link.get("href")
        race_time = link.get_text(strip=True).split()[0]
        races.append({"url": race_url, "Race Time": race_time})
    return races

def get_race_data(race_url, race_time):
    r = fetch(race_url)
    soup = BeautifulSoup(r.text, "html.parser")
    runners = []
    for row in soup.select("div.runner"):
        name_tag = row.select_one(".runner-name")
        odds_tag = row.select_one(".fixed-odds__price")
        if not name_tag or not odds_tag:
            continue
        horse = name_tag.get_text(strip=True)
        try:
            odds = float(odds_tag.get_text(strip=True))
        except:
            continue
        runners.append({"Horse": horse, "Odds_AUS": odds, "Race Time": race_time})
    num_runners = len(runners)
    for r in runners:
        market_prob = 1 / r["Odds_AUS"]
        model_prob = 1 / num_runners if num_runners else 0
        r["Market Prob"] = round(market_prob, 4)
        r["Model Prob"] = round(model_prob, 4)
        r["Value Score"] = round(model_prob - market_prob, 4)
        r["Race URL"] = race_url
        r["Country"] = "AUS"
    return pd.DataFrame(runners)

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
