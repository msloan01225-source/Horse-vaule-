
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_today_race_urls():
    url = "https://www.racingpost.com/racecards"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.select("a.rc-meeting-item__link")
    base_url = "https://www.racingpost.com"
    urls = [base_url + link.get("href") for link in links if "/racecards/" in link.get("href")]
    return urls

def get_race_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    race_title = soup.find("h2", class_="RC-header__raceInstanceTitle")
    race_title = race_title.get_text(strip=True) if race_title else "Unknown Race"

    horse_rows = soup.select("tr.RC-runnerRow")
    data = []

    for row in horse_rows:
        name_tag = row.select_one("a.RC-runnerName")
        odds_tag = row.select_one("span.RC-runnerPrice")

        if name_tag and odds_tag:
            horse_name = name_tag.get_text(strip=True)
            odds_text = odds_tag.get_text(strip=True)

            # Convert fractional odds to decimal
            try:
                if "/" in odds_text:
                    num, denom = odds_text.split("/")
                    decimal_odds = round(1 + int(num) / int(denom), 2)
                else:
                    decimal_odds = float(odds_text)
            except:
                continue

            data.append({
                "Horse": horse_name,
                "Exchange Odds": decimal_odds,
                "Race": race_title,
                "URL": url
            })

    return pd.DataFrame(data)

def get_tomorrow_race_urls():
    from datetime import datetime, timedelta
    from bs4 import BeautifulSoup
    import requests

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    base_url = "https://www.racingpost.com/racecards/"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")

    urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if tomorrow in href and "/racecards/" in href and "/racecard/" in href:
            full_url = "https://www.racingpost.com" + href
            if full_url not in urls:
                urls.append(full_url)

    return urls
