import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re

def get_timeform_data(day="Today", debug=False):
    """
    Scrapes Timeform racecards for win and place probabilities.
    """
    base_url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()

        if debug:
            with open(f"timeform_{day}.html", "w", encoding="utf-8") as f:
                f.write(r.text)

        soup = BeautifulSoup(r.text, "html.parser")
        horses = []

        # Each race section
        race_sections = soup.find_all("div", class_=re.compile("race-card"))
        for race in race_sections:
            race_name_tag = race.find("h2")
            race_name = race_name_tag.get_text(strip=True) if race_name_tag else "Unknown Race"

            # Find each runner
            runner_rows = race.find_all("tr", class_=re.compile("runner-row"))
            for runner in runner_rows:
                name_tag = runner.find("td", class_=re.compile("runner-name"))
                win_prob_tag = runner.find("td", class_=re.compile("win-percentage"))
                place_prob_tag = runner.find("td", class_=re.compile("place-percentage"))

                if not name_tag or not win_prob_tag:
                    continue

                horse_name = name_tag.get_text(strip=True)

                try:
                    win_prob = float(win_prob_tag.get_text(strip=True).replace("%", ""))
                except:
                    win_prob = 0.0

                try:
                    place_prob = float(place_prob_tag.get_text(strip=True).replace("%", ""))
                except:
                    place_prob = None

                horses.append({
                    "Horse": horse_name,
                    "Race": race_name,
                    "WinProb_TF": win_prob,
                    "PlaceProb_TF": place_prob
                })

        if debug:
            print(f"[Timeform] ✅ Found {len(horses)} horses for {day}")
        return pd.DataFrame(horses)

    except Exception as e:
        if debug:
            print(f"[Timeform] ❌ Error: {e}")
        return pd.DataFrame()
