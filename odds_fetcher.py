# odds_fetcher.py

import numpy as np

BOOKIES = ["SkyBet", "Bet365", "Betfair", "Paddy Power", "William Hill", "Ladbrokes"]

def fetch_live_odds(horse: str, bookie: str) -> float:
    """
    Stub: Returns a random decimal price for now.
    Replace with real HTTP/API calls per bookie.
    """
    # TODO: swap out this stub with real API fetch logic
    return round(np.random.uniform(1.5, 10.0), 2)
