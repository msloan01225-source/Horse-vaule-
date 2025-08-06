import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta

def get_timeform_data(day="Today"):
    """
    Scrapes Timeform racecards for UK racing.
    Returns DataFrame with: Race, Time, Course, Horse, Win_Value, Place_Value
    """

    base_url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    try:
        r = requests.get(base_url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[Timeform] ❌ Error: {e}")
        return pd.DataFrame(columns=["Race", "Time", "Course", "Horse", "Win_Value", "Place_Value"])

    soup = BeautifulSoup(r.text, "html.parser")

    races = []
    race_blocks = soup.find_all("div", class_=re.compile("^race-card__race"))

    for race_block in race_blocks:
        # Extract course and time
        course_tag = race_block.find_previous("h2")
        time_tag = race_block.find("span", class_=re.compile("^race-card__time"))

        course = course_tag.get_text(strip=True) if course_tag else "Unknown"
        time_str = time_tag.get_text(strip=True) if time_tag else "?"

        # Extract horses
        horse_tags = race_block.find_all("a", class_=re.compile("^horse-name"))
        for horse_tag in horse_tags:
            horse_name = horse_tag.get_text(strip=True)

            # Placeholder values for now
            win_val = round(abs(hash(horse_name + course + "TF")) % 100, 2)
            place_val = round(win_val * 0.8, 2)

            races.append({
                "Race": f"{course} {time_str}",
                "Time": time_str,
                "Course": course,
                "Horse": horse_name,
                "Win_Value": win_val,
                "Place_Value": place_val
            })

    return pd.DataFrame(races)
