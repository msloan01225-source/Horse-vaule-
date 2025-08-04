
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_race_data(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    horses = []
    runners = soup.find_all("div", class_="RC-runnerRow")

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
                horses.append({"Horse": name, "Exchange Odds": round(decimal_odds, 2)})
            except:
                continue

    return pd.DataFrame(horses)
