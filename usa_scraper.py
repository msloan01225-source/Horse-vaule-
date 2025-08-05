import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

BASE_URL = "https://www.equibase.com"

def get_todays_races():
    url = f"{BASE_URL}/static/entry/{datetime.now().strftime('%m%d%Y')}USA-EQB.html"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    races = []
    for link in soup.select("a[href*='race=']"):
        race_url = BASE_URL + link.get("href")
        race_time_tag = link.get_text(strip=True).split()[0]
        races.append({"url": race_url, "Race Time": race_time_tag})
    return races

def get_race_data(race_url, race_time):
    r = requests.get(race_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    runners = []
    for row in soup.select("tr.blink"):
        name_tag = row.select_one("td a")
        odds_tag = row.select_one("td[align='right']")
        if not name_tag or not odds_tag:
            continue
        horse = name_tag.get_text(strip=True)
        odds_text = odds_tag.get_text(strip=True)
        try:
            if "/" in odds_text:
                num, den = odds_text.split("/")
                odds = round(1 + (float(num) / float(den)), 2)
            else:
                odds = float(odds_text)
        except:
            continue
        runners.append({"Horse": horse, "Odds_USA": odds, "Race Time": race_time})
    num_runners = len(runners)
    for r in runners:
        market_prob = 1 / r["Odds_USA"]
        model_prob = 1 / num_runners if num_runners else 0
        r["Market Prob"] = round(market_prob, 4)
        r["Model Prob"] = round(model_prob, 4)
        r["Value Score"] = round(model_prob - market_prob, 4)
        r["Race URL"] = race_url
        r["Country"] = "USA"
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
