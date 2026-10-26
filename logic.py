"""Pure logic for the WG Organizer app: no Streamlit, no database.

Data are plain dicts:
  member   = {id, name, ci, present, favs[], prefs{diet, allergies[], likes, dislikes, cuisines[], done}, away[{id, from, to}]}
  expense  = {id, title, cat, amount, currency, eur, payer, parts[], date}
  task     = {id, kind: "chore"|"todo", title, assignee, due, every?, done?}
  shopping = {id, name, qty, unit, done, from}
Dates are ISO strings "YYYY-MM-DD".
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from recipes import DIET_NEED, DIETS, RECIPE_LEVEL, UNITS


# ---------- Dates ----------
def today() -> str:
    return date.today().isoformat()


def add_days(d: str, n: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=n)).isoformat()


def days_until(d: str, ref: str | None = None) -> int:
    return (date.fromisoformat(d) - date.fromisoformat(ref or today())).days


# ---------- Who is home ----------
def away_on(m: dict, day: str) -> dict | None:
    for a in m.get("away") or []:
        if a["from"] <= day <= a["to"]:
            return a
    return None


def is_home(m: dict, day: str | None = None, ref: str | None = None) -> bool:
    """Home on `day`? Planned absences always count; the 'present' switch only applies to today."""
    ref = ref or today()
    day = day or ref
    if away_on(m, day):
        return False
    return m.get("present", True) is not False if day == ref else True


def present(members: list[dict], ref: str | None = None) -> list[dict]:
    return [m for m in members if is_home(m, ref=ref)]


def next_available(members: list[dict], current: str, day: str, ref: str | None = None) -> str:
    """Next person after `current` (in flat order) who is home on `day`."""
    if not members:
        return current
    ids = [m["id"] for m in members]
    i = ids.index(current) if current in ids else 0
    for k in range(1, len(members) + 1):
        m = members[(i + k) % len(members)]
        if is_home(m, day, ref):
            return m["id"]
    return current


# ---------- Fair Split ----------
def to_base(eur: float, base: str, rate) -> float:
    """`rate(cur)` returns EUR per unit of `cur`."""
    return eur / rate(base)


def balances(members, expenses, base, rate) -> dict[str, float]:
    """Positive = gets money back, negative = owes money."""
    b = {m["id"]: 0.0 for m in members}
    for e in expenses:
        parts = [p for p in e.get("parts") or [] if p]
        if not parts:
            continue
        v = to_base(e["eur"], base, rate)
        b[e["payer"]] = b.get(e["payer"], 0) + v
        for p in parts:
            b[p] = b.get(p, 0) - v / len(parts)
    return b


def settle(b: dict[str, float]) -> list[dict]:
    """Few transfers that bring every balance to zero (largest debtor pays largest creditor)."""
    cr = sorted([[k, v] for k, v in b.items() if v > 0.005], key=lambda x: -x[1])
    db = sorted([[k, -v] for k, v in b.items() if v < -0.005], key=lambda x: -x[1])
    out, i, j = [], 0, 0
    while i < len(db) and j < len(cr):
        x = min(db[i][1], cr[j][1])
        out.append({"from": db[i][0], "to": cr[j][0], "v": round(x, 2)})
        db[i][1] -= x
        cr[j][1] -= x
        if db[i][1] < 0.005:
            i += 1
        if cr[j][1] < 0.005:
            j += 1
    return [t for t in out if t["v"] >= 0.01]


# ---------- Chores & calendar ----------
def chore_occurrences(task, members, start: str, end: str, ref: str | None = None) -> list[dict]:
    """Dates of a recurring chore between start and end, with rotation."""
    ref = ref or today()
    out, d, who, n = [], task["due"], task["assignee"], 0
    every = int(task.get("every") or 7)
    while d <= end and n < 500:
        n += 1
        if d >= start:
            out.append({"date": d, "who": who, "title": task["title"], "late": d < ref, "kind": "chore"})
        nd = add_days(d, every)
        who = next_available(members, who, nd, ref)
        d = nd
    return out


def events_between(tasks, members, start: str, end: str, ref: str | None = None) -> dict[str, list[dict]]:
    ref = ref or today()
    ev: dict[str, list[dict]] = {}
    for t in tasks:
        if t.get("kind") == "chore":
            for o in chore_occurrences(t, members, start, end, ref):
                ev.setdefault(o["date"], []).append(o)
        elif t.get("kind") == "todo" and t.get("due") and not t.get("done") and start <= t["due"] <= end:
            ev.setdefault(t["due"], []).append({"date": t["due"], "who": t.get("assignee"), "title": t["title"],
                                                "late": t["due"] < ref, "kind": "todo"})
    return ev


def chore_done(task, members, ref: str | None = None) -> dict:
    """Tick off a chore: next date, next person who is home that day."""
    ref = ref or today()
    base = ref if task["due"] < ref else task["due"]
    nd = add_days(base, int(task.get("every") or 7))
    return {**task, "due": nd, "assignee": next_available(members, task["assignee"], nd, ref), "last": task["assignee"]}


def month_grid(year: int, month: int) -> list[str]:
    """All days shown in a month view (Monday to Sunday, full weeks)."""
    first = date(year, month, 1)
    start = first - timedelta(days=first.weekday())
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    end = last + timedelta(days=6 - last.weekday())
    return [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]


# ---------- Recipes ----------
def words(s: str | None) -> list[str]:
    parts = re.split(r"[,;/\n]|\band\b", (s or "").lower())
    return [w.strip() for w in parts if len(w.strip()) >= 3]


def recipe_text(recipe: dict) -> str:
    """Lower-case text used for search, likes and dislikes (precomputed for database recipes)."""
    if recipe.get("search"):
        return recipe["search"]
    return " ".join([recipe["name"], *[i[2] for i in recipe.get("ingredients") or []], *(recipe.get("tags") or [])]).lower()


def shopping_name(name: str) -> str:
    """'boneless chicken breasts, cooked and diced' -> 'boneless chicken breasts'"""
    return name.split(",")[0].strip() or name


def evaluate(recipe: dict, people: list[dict]) -> dict:
    """Does a recipe suit everyone at the table? Returns blockers, reasons and a score."""
    blockers, reasons, score = [], [], 0
    text = recipe_text(recipe)
    level = RECIPE_LEVEL.get(recipe.get("diet"), 0)
    for m in people:
        p = m.get("prefs") or {}
        diet = p.get("diet") or "any"
        if DIET_NEED.get(diet, 0) > level:
            blockers.append(f"{m['name']} is {DIETS[diet].lower()}")
        for a in p.get("allergies") or []:
            if a in (recipe.get("allergens") or []):
                blockers.append(f"{a} ({m['name']})")
        for w in words(p.get("dislikes")):
            if w in text:
                blockers.append(f"{m['name']} doesn't like {w}")
        if recipe["id"] in (m.get("favs") or []):
            score += 3
            reasons.append((m, "♥ favourite"))
        for w in words(p.get("likes")):
            if w in text:
                score += 2
                reasons.append((m, w))
        if recipe.get("cuisine") and recipe["cuisine"] in (p.get("cuisines") or []):
            score += 1
            reasons.append((m, recipe["cuisine"]))
    return {"r": recipe, "ok": not blockers, "blockers": list(dict.fromkeys(blockers)), "reasons": reasons, "score": score}


def parse_ingredient(line: str):
    """'200 g lentils' -> [200.0, 'g', 'lentils'];  '1 onion' -> [1.0, '', 'onion']"""
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*(\S+)?\s*(.*)$", line)
    if not m:
        return [0, "", line]
    q = float(m.group(1).replace(",", "."))
    u, n = m.group(2) or "", m.group(3)
    if u not in UNITS:
        n, u = f"{u} {n}".strip(), ""
    return [q, u, n or line]


def fmt_qty(q) -> str:
    if not q:
        return ""
    r = round(q) if q >= 10 else round(q * 4) / 4
    return str(int(r)) if float(r).is_integer() else str(r)


SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥"}


def money(v: float, cur: str) -> str:
    sign = "-" if v < 0 else ""
    sym = SYMBOLS.get(cur)
    return f"{sign}{sym}{abs(v):,.2f}" if sym else f"{sign}{cur} {abs(v):,.2f}"
