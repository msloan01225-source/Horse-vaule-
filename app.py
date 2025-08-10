# app.py — EdgeBrain API (FastAPI) with cache/throttle/breaker
# ------------------------------------------------------------
# Endpoints:
#   GET  /health
#   GET  /config                 -> remote config (weights/alerts/ui/limits)
#   POST /config                 -> update config (persist to config.json)
#   GET  /presets                -> EdgeBrain weight presets
#   GET  /markets?day=today      -> all runners with metrics (cached)
#   GET  /races?day=today        -> list of "Course | HH:MM"
#   GET  /race?course&time_hhmm  -> runners for that race
#   GET  /race/top3?course&time  -> top 3 picks
#   GET  /race/history?course... -> EV/odds history (for Race Tape)
#   GET  /alerts?day=today       -> hot picks since last poll (client dedupe)
#   POST /push/register          -> register Expo push token
#   POST /alerts/push-now        -> send current hot picks as push to all tokens

import os
import json
import time
import hashlib
import random
import logging
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Env ----------
load_dotenv()

BF_APP_KEY = os.getenv("BF_APP_KEY", "")
BF_SESSION = os.getenv("BF_SESSION", "")
BF_USER    = os.getenv("BF_USER", "")
BF_PASS    = os.getenv("BF_PASS", "")
RA_USER    = os.getenv("RACING_API_USERNAME", "")
RA_PASS    = os.getenv("RACING_API_PASSWORD", "")
ALLOWED    = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

BF_IDENTITY_URL = "https://identitysso.betfair.com/api/login"
BF_API_URL      = "https://api.betfair.com/exchange/betting/json-rpc/v1"

# ---------- Files ----------
ROOT = Path(__file__).parent
CFG_PATH    = ROOT / "config.json"
TOKENS_PATH = ROOT / "push_tokens.json"

# ---------- App ----------
app = FastAPI(title="EdgeBrain API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in ALLOWED else ALLOWED,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- Helpers ----------
def now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def iso_range_for_day(day: str) -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    s = base if day == "today" else base + timedelta(days=1)
    e = s + timedelta(days=1)
    return s.isoformat().replace("+00:00","Z"), e.isoformat().replace("+00:00","Z")

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def stable_rand01(key: str) -> float:
    h = hashlib.sha256(key.encode()).hexdigest()
    return random.Random(int(h[:8], 16)).random()

def load_cfg() -> dict:
    try:
        return json.loads(CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "weights": {"w_base":0.50,"w_mom":0.18,"w_press":0.10,"w_sp":0.06,"w_value":0.08,"w_liq":0.05,"w_spread":-0.04,"w_vol":-0.03},
            "alerts":  {"ev_min":0.10,"eb_min":32,"cooldown_sec":180},
            "ui":      {"top_bars":12,"top_runners":6},
            "limits":  {"markets_ttl_sec":30,"betfair_min_interval_sec":5,"breaker_threshold":3,"breaker_cooldown_sec":180,"max_stale_sec":300}
        }

def save_cfg(cfg: dict):
    CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

# ---- Limits helpers ----
def cfg_limits():
    c = load_cfg().get("limits", {})
    return {
        "markets_ttl_sec": int(c.get("markets_ttl_sec", 30)),
        "betfair_min_interval_sec": int(c.get("betfair_min_interval_sec", 5)),
        "breaker_threshold": int(c.get("breaker_threshold", 3)),
        "breaker_cooldown_sec": int(c.get("breaker_cooldown_sec", 180)),
        "max_stale_sec": int(c.get("max_stale_sec", 300)),
    }

# ---------- Tokens ----------
def load_tokens() -> set[str]:
    try:
        return set(json.loads(TOKENS_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()

def save_tokens(tokens: set[str]):
    TOKENS_PATH.write_text(json.dumps(sorted(tokens), ensure_ascii=False), encoding="utf-8")

# ---------- Betfair auth/calls ----------
_session = {"token": None, "ts": 0.0}

def bf_get_session_token() -> Optional[str]:
    # cache token for 10 minutes
    if _session["token"] and (time.time() - _session["ts"] < 600):
        return _session["token"]
    if BF_SESSION:
        _session.update(token=BF_SESSION, ts=time.time())
        return BF_SESSION
    if not (BF_APP_KEY and BF_USER and BF_PASS):
        return None
    try:
        r = requests.post(
            BF_IDENTITY_URL,
            headers={"X-Application": BF_APP_KEY, "Content-Type": "application/x-www-form-urlencoded"},
            data={"username": BF_USER, "password": BF_PASS},
            timeout=12
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "SUCCESS":
            tok = data.get("token")
            _session.update(token=tok, ts=time.time())
            return tok
    except Exception:
        return None
    return None

def bf_headers(tok: str) -> Dict[str, str]:
    return {"X-Application": BF_APP_KEY, "X-Authentication": tok, "Content-Type": "application/json"}

def bf_post(method: str, params: dict, tok: str) -> dict:
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    r = requests.post(BF_API_URL, headers=bf_headers(tok), data=json.dumps(payload), timeout=12)
    r.raise_for_status()
    return r.json()

# ---- throttle & breaker state ----
_last_bf_call_ts = 0.0
_bf_fail_count = 0
_breaker_open_until = 0.0

def breaker_is_open() -> bool:
    return time.time() < _breaker_open_until

def breaker_note_success():
    global _bf_fail_count
    _bf_fail_count = 0

def breaker_note_failure(http_status: Optional[int] = None):
    """Increase failure count and open breaker for configured cooldown if threshold reached."""
    global _bf_fail_count, _breaker_open_until
    lim = cfg_limits()
    if http_status in (403, 429, 500, 502, 503, 504, None):
        _bf_fail_count += 1
        if _bf_fail_count >= lim["breaker_threshold"]:
            _breaker_open_until = time.time() + lim["breaker_cooldown_sec"]
            logging.warning(f"[Breaker] Open for {lim['breaker_cooldown_sec']}s after {_bf_fail_count} failures.")

def bf_call(method: str, params: dict) -> dict:
    """Betfair call with throttling and circuit breaker."""
    global _last_bf_call_ts
    lim = cfg_limits()

    # Circuit breaker guard
    if breaker_is_open():
        raise RuntimeError("Betfair breaker open (cooldown)")

    # Throttle
    dt = time.time() - _last_bf_call_ts
    if dt < lim["betfair_min_interval_sec"]:
        wait = lim["betfair_min_interval_sec"] - dt
        if wait > 0:
            time.sleep(wait)

    tok = bf_get_session_token()
    if not tok:
        breaker_note_failure(None)
        raise RuntimeError("No Betfair token")

    try:
        out = bf_post(method, params, tok)
        _last_bf_call_ts = time.time()

        if "result" in out:
            breaker_note_success()
            return out["result"]

        # retry once on auth errors
        code = (out.get("error", {}).get("data", {}) or {}).get("APINGException", {}).get("errorCode")
        if code in {"INVALID_SESSION_INFORMATION", "NO_APP_KEY"}:
            logging.info("[BF] Auth issue; refreshing token.")
            _session.update(token=None, ts=0.0)
            tok = bf_get_session_token()
            if not tok:
                breaker_note_failure(None)
                raise RuntimeError("Betfair auth failed")
            out2 = bf_post(method, params, tok)
            _last_bf_call_ts = time.time()
            if "result" in out2:
                breaker_note_success()
                return out2["result"]
            breaker_note_failure(None)
            raise RuntimeError(str(out2))

        breaker_note_failure(None)
        raise RuntimeError(str(out))

    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        logging.warning(f"[BF] HTTPError {status}: {e}")
        if status == 403:
            breaker_note_failure(403)
        else:
            breaker_note_failure(status)
        raise
    except Exception as e:
        logging.warning(f"[BF] Call failed: {e}")
        breaker_note_failure(None)
        raise

def best_prices_sizes(rb: dict) -> Tuple[Optional[float], Optional[float], float, float, Optional[float]]:
    ex = rb.get("ex") or {}
    atb = ex.get("availableToBack") or []
    atl = ex.get("availableToLay") or []
    bb = atb[0]["price"] if atb else None
    bl = atl[0]["price"] if atl else None
    bsz = float(sum([x.get("size", 0.0) for x in atb[:3]]))
    lsz = float(sum([x.get("size", 0.0) for x in atl[:3]]))
    sp = (rb.get("sp") or {}).get("farPrice") or (rb.get("sp") or {}).get("nearPrice")
    return bb, bl, bsz, lsz, sp

# ---------- De‑vig ----------
def devig_power(odds: pd.Series) -> pd.Series:
    x = np.clip(1.0 / np.clip(odds.astype(float), 1.01, None), 1e-9, None)
    lo, hi = 0.5, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s = float(np.sum(np.power(x, mid)))
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    p = np.power(x, alpha)
    return pd.Series(p / p.sum())

def add_fair_probs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["RaceKey"] = df["Course"].astype(str) + " | " + df["Time"].astype(str)
    parts = []
    for _, g in df.groupby("RaceKey"):
        p = devig_power(g["Odds"])
        parts.append(pd.DataFrame({"idx": g.index, "Fair p": p.values}))
    fair = pd.concat(parts) if parts else pd.DataFrame(columns=["idx", "Fair p"])
    df = df.merge(fair.set_index("idx"), left_index=True, right_index=True, how="left")
    df["Fair Odds"] = (1.0 / df["Fair p"].clip(1e-6, None)).round(2)
    return df

# ---------- EdgeBrain ----------
def edgebrain(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Fair p"] = out["Fair p"].clip(0.001, 0.999)
    out["Base p %"] = (out["Fair p"] * 100.0).round(2)
    out["Momentum"] = 0.0
    out["Pressure"] = ((out["Back Size"] - out["Lay Size"]) / (out["Back Size"] + out["Lay Size"] + 1e-6) * 10.0).clip(-6, 6)
    out["SP Deviation %"] = np.where(out["SP (proj)"].notna(),
                                     (out["SP (proj)"] - out["Odds"]) / out["Odds"] * 100.0, 0.0).round(1)
    out["Spread %"] = np.where(out["Best Lay"].notna(),
                               ((out["Best Lay"] - out["Odds"]) / ((out["Best Lay"] + out["Odds"]) / 2 + 1e-9) * 100.0).clip(0, 12),
                               0.0)
    out["LiqScore"] = np.log10(np.maximum(out["Back Size"] + out["Lay Size"], 1.0)).clip(0, 4.0)
    value_boost = ((1.0 / np.clip(out["Odds"], 1.01, None)) - out["Fair p"]) * 100.0

    w = {"w_base":0.50,"w_mom":0.18,"w_press":0.10,"w_sp":0.06,"w_value":0.08,"w_liq":0.05,"w_spread":-0.04,"w_vol":-0.03}
    w.update(weights or {})

    out["EdgeBrain Score"] = (
        w["w_base"]   * out["Base p %"] +
        w["w_mom"]    * out["Momentum"] +
        w["w_press"]  * out["Pressure"] +
        w["w_sp"]     * np.clip(-out["SP Deviation %"] / 2.0, -6, 6) +
        w["w_value"]  * value_boost +
        w["w_liq"]    * out["LiqScore"] * 10.0 +
        w["w_spread"] * out["Spread %"] +
        w["w_vol"]    * 0.0
    ).clip(0, 100).round(1)

    p = out["Fair p"]; q = 1 - p
    out["EV"] = (p * out["Odds"] - q).round(2)
    return out

# ---------- Cache & History ----------
# Per-day cache maps
_cache_map = {
    "today": {"ts": 0.0, "df": pd.DataFrame()},
    "tomorrow": {"ts": 0.0, "df": pd.DataFrame()},
}

# live price history: key "MarketId:SelectionId" -> deque[(ts, odds)]
HIST: Dict[str, deque] = defaultdict(lambda: deque(maxlen=180))  # ~15 min @ 5s

def refresh(day: str, countries: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch live markets with throttling + breaker, update history, compute metrics, cache per day."""
    lim = cfg_limits()
    now_ts = time.time()

    cache = _cache_map.get(day, {"ts": 0.0, "df": pd.DataFrame()})
    cache_age = now_ts - cache["ts"]

    # Serve fresh cache
    if cache_age < lim["markets_ttl_sec"] and not cache["df"].empty:
        return cache["df"]

    # If breaker is open, serve stale (if allowed)
    if breaker_is_open():
        if cache["df"].empty or cache_age > lim["max_stale_sec"]:
            logging.warning("[BF] Breaker open & no acceptable stale cache; returning empty.")
            return pd.DataFrame()
        logging.info("[BF] Breaker open; serving stale cache.")
        return cache["df"]

    countries = countries or ["GB"]
    fr, to = iso_range_for_day(day)

    # 1) Catalogue
    try:
        cat = bf_call("listMarketCatalogue", {
            "filter": {"eventTypeIds": ["7"], "marketCountries": countries, "marketTypeCodes": ["WIN"], "marketStartTime": {"from": fr, "to": to}},
            "maxResults": 200, "marketProjection": ["RUNNER_DESCRIPTION", "MARKET_START_TIME", "EVENT"]
        })
    except Exception as e:
        # On failure, serve stale if within max_stale_sec
        if not cache["df"].empty and cache_age <= lim["max_stale_sec"]:
            logging.warning(f"[BF] Catalogue failed ({e}); serving stale cache.")
            return cache["df"]
        raise HTTPException(status_code=502, detail=f"Betfair error: {e}")

    ids = [m.get("marketId") for m in cat if m.get("marketId")]
    if not ids:
        _cache_map[day] = {"ts": time.time(), "df": pd.DataFrame()}
        return _cache_map[day]["df"]

    # 2) Books
    books = []
    for c in chunks(ids, 25):
        try:
            res = bf_call("listMarketBook", {
                "marketIds": c,
                "priceProjection": {"priceData": ["EX_BEST_OFFERS", "SP_PROJECTED"], "virtualise": True}
            })
            books.extend(res or [])
        except Exception as e:
            logging.warning(f"[BF] listMarketBook chunk failed: {e}")
            continue

    # 3) Map prices & build rows
    pm: Dict[str, Dict[int, Tuple]] = {}
    for mb in books:
        mid = mb.get("marketId")
        pm[mid] = {}
        for rb in mb.get("runners") or []:
            pm[mid][rb.get("selectionId")] = best_prices_sizes(rb)

    rows = []
    for m in cat:
        mid = m.get("marketId")
        venue = (m.get("event") or {}).get("venue", "Unknown")
        tstr = (m.get("marketStartTime", "") or "")[11:16]
        for r in (m.get("runners") or []):
            name = r.get("runnerName"); sel = r.get("selectionId")
            bb, bl, bsz, lsz, sp = (pm.get(mid, {}) or {}).get(sel, (None, None, 0.0, 0.0, None))
            if not name or bb is None:
                continue
            key = f"{mid}:{sel}"
            ts = int(time.time())
            if not HIST[key] or HIST[key][-1][0] != ts:
                HIST[key].append((ts, float(bb)))
            rows.append({
                "MarketId": mid, "SelectionId": sel, "Course": venue, "Time": tstr, "Horse": name,
                "Odds": round(float(bb), 2),
                "Best Lay": round(float(bl), 2) if bl else None,
                "Back Size": float(bsz or 0.0), "Lay Size": float(lsz or 0.0),
                "SP (proj)": round(float(sp), 2) if sp else None
            })

    df = pd.DataFrame(rows)
    if df.empty:
        _cache_map[day] = {"ts": time.time(), "df": df}
        return df

    # Faux value fields for parity with the original UI
    key = (df["Course"].astype(str) + "|" + df["Time"].astype(str) + "|" + df["Horse"].astype(str))
    df["Win_Value"] = (5 + key.apply(stable_rand01) * 25).round(1)
    df["Predicted Win %"] = (100.0 / df["Odds"].clip(lower=1.01)).round(1)
    df["BetEdge Win %"] = ((df["Predicted Win %"] * 0.6) + (df["Win_Value"] * 0.4)).round(1)

    # fair odds + EdgeBrain
    df = add_fair_probs(df)
    cfg = load_cfg()
    df = edgebrain(df, cfg.get("weights", {}))

    _cache_map[day] = {"ts": time.time(), "df": df}
    return df

# ---------- Presets ----------
PRESETS = {
    "Conservative": {"w_base":0.60,"w_mom":0.10,"w_press":0.08,"w_sp":0.05,"w_value":0.05,"w_liq":0.07,"w_spread":-0.03,"w_vol":-0.02},
    "Balanced":     {"w_base":0.50,"w_mom":0.18,"w_press":0.10,"w_sp":0.06,"w_value":0.08,"w_liq":0.05,"w_spread":-0.04,"w_vol":-0.03},
    "Aggressive":   {"w_base":0.40,"w_mom":0.25,"w_press":0.12,"w_sp":0.04,"w_value":0.12,"w_liq":0.03,"w_spread":-0.05,"w_vol":-0.04}
}

# ---------- Push (Expo) ----------
def expo_send_push(tokens: List[str], title: str, body: str) -> Dict[str, int]:
    if not tokens:
        return {"ok": 0, "err": 0}
    try:
        r = requests.post(
            "https://exp.host/--/api/v2/push/send",
            json=[{"to": t, "title": title, "body": body, "sound": "default"} for t in tokens],
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        ok = sum(1 for i in data.get("data", []) if i.get("status") == "ok")
        err = len(tokens) - ok
        return {"ok": ok, "err": err}
    except Exception:
        return {"ok": 0, "err": len(tokens)}

# ---------- Endpoints ----------
@app.get("/health")
def health():
    lim = cfg_limits()
    return {
        "ok": True,
        "time": now_iso(),
        "breaker_open": breaker_is_open(),
        "cache_ages_sec": {
            k: (time.time() - v["ts"]) if v["ts"] else None for k, v in _cache_map.items()
        },
        "limits": lim
    }

@app.get("/config")
def get_config():
    return {"updated": now_iso(), **load_cfg()}

@app.post("/config")
def set_config(payload: Dict[str, Any] = Body(...)):
    cfg = load_cfg()
    if "weights" in payload: cfg["weights"] = payload["weights"]
    if "alerts"  in payload: cfg["alerts"]  = payload["alerts"]
    if "ui"      in payload: cfg["ui"]      = payload["ui"]
    if "limits"  in payload: cfg["limits"]  = payload["limits"]
    save_cfg(cfg)
    # Invalidate all day caches
    for k in _cache_map.keys():
        _cache_map[k] = {"ts": 0.0, "df": pd.DataFrame()}
    return {"ok": True, "updated": now_iso(), "config": cfg}

@app.get("/presets")
def get_presets():
    return {"presets": PRESETS}

@app.get("/markets")
def markets(day: str = Query("today", enum=["today", "tomorrow"])):
    logging.info(f"/markets day={day}")
    df = refresh(day)
    return json.loads(df.to_json(orient="records"))

@app.get("/races")
def races(day: str = Query("today", enum=["today", "tomorrow"])):
    logging.info(f"/races day={day}")
    df = refresh(day)
    if df.empty: return []
    return sorted((df["Course"] + " | " + df["Time"]).unique().tolist())

@app.get("/race")
def race(course: str, time_hhmm: str, day: str = Query("today", enum=["today", "tomorrow"])):
    logging.info(f"/race {course} {time_hhmm} day={day}")
    df = refresh(day)
    if df.empty: return []
    rd = df[(df["Course"] == course) & (df["Time"] == time_hhmm)].copy()
    return json.loads(rd.to_json(orient="records"))

@app.get("/race/top3")
def race_top3(course: str, time_hhmm: str, day: str = Query("today", enum=["today", "tomorrow"])):
    logging.info(f"/race/top3 {course} {time_hhmm} day={day}")
    df = refresh(day)
    if df.empty: return {"race": f"{course} | {time_hhmm}", "top3": []}
    rd = df[(df["Course"] == course) & (df["Time"] == time_hhmm)].copy()
    if rd.empty: return {"race": f"{course} | {time_hhmm}", "top3": []}
    score = (1.10 * rd["Fair p"] + 0.010 * rd["Pressure"] +
             0.006 * (-rd["SP Deviation %"]) + 0.002 * (rd["LiqScore"] * 10.0) +
             -0.002 * rd["Spread %"])
    logits = score / 0.85
    logits = logits - logits.max()
    winp = np.exp(np.clip(logits, -50, 50)); winp = winp / (winp.sum() + 1e-9)
    rd = rd.copy()
    rd["Win p %"] = (winp * 100.0).round(1)
    out = rd.sort_values("Win p %", ascending=False).head(3)[
        ["Horse","Odds","Fair Odds","Win p %","EdgeBrain Score","EV","Course","Time"]
    ]
    return {"race": f"{course} | {time_hhmm}", "top3": json.loads(out.to_json(orient="records"))}

@app.get("/race/history")
def race_history(course: str, time_hhmm: str, day: str = Query("today", enum=["today", "tomorrow"])):
    logging.info(f"/race/history {course} {time_hhmm} day={day}")
    df = refresh(day)
    if df.empty: return {"race": f"{course} | {time_hhmm}", "series": {}}
    rd = df[(df["Course"] == course) & (df["Time"] == time_hhmm)].copy()
    series: Dict[str, List[Dict[str, float]]] = {}
    for _, r in rd.iterrows():
        key = f"{r['MarketId']}:{r['SelectionId']}"
        fair_p = float(r["Fair p"])
        dq = HIST.get(key, deque())
        if not dq:
            continue
        t0 = dq[0][0]
        seq = [{"t": int(ts), "rel": int(ts - t0), "odds": float(od), "ev": float(fair_p * float(od) - (1.0 - fair_p))} for ts, od in list(dq)]
        if seq:
            series[r["Horse"]] = seq
    return {"race": f"{course} | {time_hhmm}", "series": series}

@app.get("/alerts")
def get_alerts(day: str = Query("today", enum=["today", "tomorrow"])):
    logging.info(f"/alerts day={day}")
    cfg = load_cfg()
    ev_min = float(cfg.get("alerts", {}).get("ev_min", 0.10))
    eb_min = int(cfg.get("alerts", {}).get("eb_min", 32))
    cooldown = int(cfg.get("alerts", {}).get("cooldown_sec", 180))

    df = refresh(day)
    if df.empty: return {"now": now_iso(), "items": []}

    hot = df[(df["EV"] >= ev_min) & (df["EdgeBrain Score"] >= eb_min)].copy()
    if hot.empty: return {"now": now_iso(), "items": []}

    # crude cooldown per batch; clients should dedupe by id
    hot["id"] = hot["MarketId"].astype(str) + ":" + hot["SelectionId"].astype(str)
    now_ts = int(time.time())

    # Simple in-process cooldown state
    if not hasattr(get_alerts, "_last_ts"):
        get_alerts._last_ts = 0  # type: ignore
        get_alerts._last_ids = set()  # type: ignore

    items = []
    for _, r in hot.iterrows():
        iid = r["id"]
        if iid in get_alerts._last_ids and now_ts - get_alerts._last_ts < cooldown:  # type: ignore
            continue
        items.append({
            "id": iid, "horse": r["Horse"], "course": r["Course"], "time": r["Time"],
            "odds": float(r["Odds"]), "ev": float(r["EV"]), "eb": float(r["EdgeBrain Score"])
        })

    if items:
        get_alerts._last_ids = {x["id"] for x in items}  # type: ignore
        get_alerts._last_ts = now_ts  # type: ignore

    return {"now": now_iso(), "items": items}

@app.post("/push/register")
def push_register(payload: Dict[str, Any] = Body(...)):
    token = (payload or {}).get("token", "")
    if not isinstance(token, str) or not token.startswith("ExponentPushToken"):
        raise HTTPException(400, "Invalid token")
    t = load_tokens()
    t.add(token)
    save_tokens(t)
    return {"ok": True, "count": len(t)}

@app.post("/alerts/push-now")
def alerts_push_now(day: str = Query("today", enum=["today", "tomorrow"])):
    cfg = load_cfg()
    ev_min = float(cfg.get("alerts", {}).get("ev_min", 0.10))
    eb_min = int(cfg.get("alerts", {}).get("eb_min", 32))
    df = refresh(day)
    if df.empty: return {"sent": 0, "message": "no data"}

    hot = df[(df["EV"] >= ev_min) & (df["EdgeBrain Score"] >= eb_min)].sort_values(
        ["EV", "EdgeBrain Score"], ascending=[False, False]).head(3)
    if hot.empty: return {"sent": 0, "message": "no hot picks"}

    msg = "; ".join(f"{r.Horse} @{r.Odds:.2f} EV {r.EV:+.2f} ({r.Course} {r.Time})" for _, r in hot.iterrows())
    res = expo_send_push(list(load_tokens()), "EdgeBet Hot Picks", msg)
    return {"sent": res["ok"], "failed": res["err"], "message": msg}

# Run: uvicorn app:app --host 0.0.0.0 --port 8787
