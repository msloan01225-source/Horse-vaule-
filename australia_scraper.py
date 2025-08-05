import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

BASE_URL = "https://www.racing.com"

def get_todays_races():
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/racecards/{today}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")

    races = []
    for link in soup.select("a.meeting-race__race"):
        race_url = BASE_URL + link.get("href")
        races.append({"url": race_url})
    return races

def get_race_data(race_url):
    r = requests.get(race_url, headers={"User-Agent": "Mozilla/5.0"})
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

        runners.append({"Horse": horse, "Odds_AUS": odds, "Race URL": race_url})

    # Calculate EV
    num_runners = len(runners)
    for r in runners:
        market_prob = 1 / r["Odds_AUS"]
        model_prob = 1 / num_runners if num_runners else 0
        r["Market Prob"] = round(market_prob, 4)
        r["Model Prob"] = round(model_prob, 4)
        r["Value Score"] = round(model_prob - market_prob, 4)
        r["Country"] = "AUS"

    return pd.DataFrame(runners)

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

if __name__ == "__main__":
    df = get_all_todays_data()
    print(df.head())
