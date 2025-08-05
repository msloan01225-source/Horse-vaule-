import time
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def get_racingpost_data(day="Today", debug=False):
    """
    Scrapes Racing Post time-order page for UK races (today/tomorrow).
    """

    base_url = "https://www.racingpost.com/racecards/time-order/"
    if day.lower() == "tomorrow":
        target_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        base_url += f"?date={target_date}"

    # Selenium headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    races = []
    try:
        driver.get(base_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.rc-cardHeader__raceInstanceTitle"))
        )

        if debug:
            with open(f"racingpost_{day}.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)

        race_cards = driver.find_elements(By.CSS_SELECTOR, "div.rc-card")
        for card in race_cards:
            try:
                time_text = card.find_element(By.CSS_SELECTOR, "span.rc-cardHeader__time").text
                course = card.find_element(By.CSS_SELECTOR, "span.rc-cardHeader__courseName").text
                race_name = card.find_element(By.CSS_SELECTOR, "a.rc-cardHeader__raceInstanceTitle").text
                runners = card.find_element(By.CSS_SELECTOR, "span.rc-cardHeader__runners").text

                races.append({
                    "Race": f"{course} {time_text} - {race_name}",
                    "Time": time_text,
                    "Course": course,
                    "Race Name": race_name,
                    "Runners": runners
                })
            except:
                continue

    finally:
        driver.quit()

    return pd.DataFrame(races)
