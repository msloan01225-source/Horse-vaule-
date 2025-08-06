import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta

def get_racingpost_data(day="Today"):
    """
    Scrapes Racing Post racecards in time order.
    Returns DataFrame with: Race, Time, Course, Horse, Win_Value, Place_Value
    """

    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[RacingPost] ❌ Error: {e}")
        return pd.DataFrame(columns=["Race", "Time", "Course", "Horse", "Win_Value", "Place_Value"])

    soup = BeautifulSoup(r.text, "html.parser")

    races = []
    race_blocks = soup.find_all("div", class_=re.compile("^(RC-courseBox|RC-courseBox__race)$"))

    for race_block in race_blocks:
        # Extract course and time
        course_tag = race_block.find_previous("h3")
        time_tag = race_block.find("span", class_=re.compile("^RC-courseBox__time"))

        course = course_tag.get_text(strip=True) if course_tag else "Unknown"
        time_str = time_tag.get_text(strip=True) if time_tag else "?"

        # Extract horses
        horse_tags = race_block.find_all("a", class_=re.compile("^RC-runnerName"))
        for horse_tag in horse_tags:
            horse_name = horse_tag.get_text(strip=True)

            # For now, assign random placeholder values for Win/Place
            # Replace this with real odds-to-value calculation logic
            win_val = round(abs(hash(horse_name + course)) % 100, 2)
            place_val = round(win_val * 0.75, 2)

            races.append({
                "Race": f"{course} {time_str}",
                "Time": time_str,
                "Course": course,
                "Horse": horse_name,
                "Win_Value": win_val,
                "Place_Value": place_val
            })

    return pd.DataFrame(races)
