# test_api.py
import requests
from requests.auth import HTTPBasicAuth
import json

USERNAME = "VxPO6jl8GNfsg7fchzUEt6MF"
PASSWORD = "CPdOLINCiTdLDgVZoCzo8c9Y"

def test_racecards_api():
    url = "https://api.theracingapi.com/v1/racecards"
    # Try calling with no params first
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=10)
    print("REQUESTED URL:", r.request.url)
    print("STATUS CODE :", r.status_code)
    try:
        data = r.json()
        print("JSON KEYS   :", list(data.keys()))
        print("SAMPLE DATA :")
        print(json.dumps(data, indent=2)[:1000])  # first 1000 chars
    except ValueError:
        print("Invalid JSON response, raw text below:")
        print(r.text)

if __name__ == "__main__":
    test_racecards_api()
