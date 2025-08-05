import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://www.timeform.com"

def get_todays_races():
    url = f"{BASE_URL}/horse-racing/racecards"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    races = []
    for link in soup.select("a.race-link"):
        race_url = BASE_URL + link.get("href")
        time_course = link.get_text(strip=True)
        races.append({"url": race_url, "time_course": time_course})
    return races

def get_race_data(race_url):
    r = requests.get(race_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    runner_rows = soup.select("tr.runner-row")
    data = []
    for row in runner_rows:
        horse_tag = row.select_one(".runner-name")
        odds_tag = row.select_one(".odds")
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
        data.append({"Horse": horse, "Odds_TF": odds})
    num_runners = len(data)
    for d in data:
        market_prob = 1 / d["Odds_TF"]
        model_prob = 1 / num_runners if num_runners else 0
        d["Market Prob"] = round(market_prob, 4)
        d["Model Prob"] = round(model_prob, 4)
        d["Value Score"] = round(model_prob - market_prob, 4)
        d["Race URL"] = race_url
    return pd.DataFrame(data)

def get_all_todays_data():
    races = get_todays_races()
    all_data = []
    for race in races:
        df = get_race_data(race["url"])
        if not df.empty:
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()
