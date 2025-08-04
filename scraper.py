import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

def get_today_race_urls():
    base_url = "https://www.racingpost.com/racecards/"
    today = datetime.now().strftime("%Y-%m-%d")
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")

    urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if today in href and "/racecards/" in href and "/racecard/" in href:
            full_url = "https://www.racingpost.com" + href
            if full_url not in urls:
                urls.append(full_url)

    return urls

def ():
    base_url = "https://www.racingpost.com/racecards/"
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
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

def get_race_data(race_url):
    response = requests.get(race_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Table rows for runners
    runners = soup.find_all("tr", class_="rp-racecard-runner")

    data = []
    for runner in runners:
        try:
            horse = runner.find("a", class_="rp-horse-name").text.strip()
            jockey = runner.find("span", class_="rp-horse-jockey").text.strip()
            trainer = runner.find("span", class_="rp-horse-trainer").text.strip()
            form = runner.find("span", class_="rp-horse-form").text.strip()
            official_rating_tag = runner.find("td", class_="rp-official-rating")
            official_rating = official_rating_tag.text.strip() if official_rating_tag else None
            odds_tag = runner.find("span", class_="rp-odds")
            exchange_odds = float(odds_tag.text.strip()) if odds_tag and odds_tag.text.strip() != "-" else None
            track_condition_tag = soup.find("span", class_="rp-track-condition")
            track_condition = track_condition_tag.text.strip() if track_condition_tag else None

            data.append({
                "Horse": horse,
                "Jockey": jockey,
                "Trainer": trainer,
                "Form": form,
                "Official Rating": official_rating,
                "Exchange Odds": exchange_odds,
                "Track Condition": track_condition,
            })
        except Exception as e:
            # Skip any runners where data is incomplete
            continue

    df = pd.DataFrame(data)

    # Drop rows with missing odds (can't calculate value score)
    df = df.dropna(subset=["Exchange Odds"])

    return df
