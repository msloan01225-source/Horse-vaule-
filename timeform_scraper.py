import time
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def get_timeform_data(day="Today", debug=False):
    """
    Scrapes Timeform racecards for UK races (today/tomorrow).
    """

    base_url = "https://www.timeform.com/horse-racing/racecards"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    races = []
    try:
        driver.get(base_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.race-card"))
        )

        if debug:
            with open(f"timeform_{day}.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)

        race_cards = driver.find_elements(By.CSS_SELECTOR, "div.race-card")
        for card in race_cards:
            try:
                time_text = card.find_element(By.CSS_SELECTOR, "div.time").text
                course = card.find_element(By.CSS_SELECTOR, "div.course").text
                race_name = card.find_element(By.CSS_SELECTOR, "div.race-name").text

                races.append({
                    "Race": f"{course} {time_text} - {race_name}",
                    "Time": time_text,
                    "Course": course,
                    "Race Name": race_name
                })
            except:
                continue

    finally:
        driver.quit()

    return pd.DataFrame(races)
