"""Daily exchange rates from the European Central Bank (ECB).

The ECB publishes reference rates on working days at around 16:00 CET (units per 1 EUR).
The app fetches them on start and then at most every few hours.
If the ECB can't be reached, the last saved rates are used.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import requests

import store

ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
CURRENCIES = ["EUR", "CHF", "USD", "GBP", "PLN", "CZK", "HUF", "SEK", "DKK", "NOK", "JPY"]
# Fallback values (ECB, 20 Oct 2026) in case rates were never loaded
FALLBACK = {"date": "2026-10-20", "source": "fallback", "perEur": {
    "CHF": 0.9237, "USD": 1.1215, "GBP": 0.84028, "PLN": 4.3768, "CZK": 24.329, "HUF": 364.68,
    "SEK": 11.123, "DKK": 7.4746, "NOK": 10.5245, "JPY": 177.34}}


def fetch_ecb(timeout: float = 8) -> dict:
    resp = requests.get(ECB_URL, timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    day, rates = None, {}
    for el in root.iter():
        if el.tag.endswith("Cube"):
            if "time" in el.attrib:
                day = el.attrib["time"]
            if "currency" in el.attrib:
                rates[el.attrib["currency"]] = float(el.attrib["rate"])
    if not day or not rates:
        raise ValueError("ECB response contained no rates")
    return {"date": day, "source": "ECB", "perEur": {c: rates[c] for c in CURRENCIES if c in rates}}


def load_rates() -> tuple[dict, str | None]:
    """Returns (rates, error message). Saves successful fetches to the database."""
    try:
        fx = fetch_ecb()
        store.put("config", {"id": "fx", **fx})
        return fx, None
    except Exception as exc:  # network error, timeout, broken XML
        saved = store.get("config", "fx")
        return (saved or FALLBACK), f"Couldn't reach the ECB right now ({type(exc).__name__})."


def eur_per_unit(fx: dict, cur: str) -> float:
    if cur == "EUR":
        return 1.0
    per = fx["perEur"].get(cur) or FALLBACK["perEur"].get(cur)
    return 1 / per if per else 1.0
