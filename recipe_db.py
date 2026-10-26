"""Read-only access to the big recipe database.

The database is built by tools/build_recipe_db.py from the Allrecipes dataset and stored as
data/recipes.db.xz.part01, part02, … (one xz-compressed SQLite file, split into small parts so
each file stays well under GitHub's upload limits). On first use the parts are joined and
unpacked to data/recipes.db. For speed, the app keeps a light index (name, course, cuisine, diet,
allergens, time, search text) in memory and loads the full recipe only when it is opened.
"""
from __future__ import annotations

import glob
import json
import lzma
import os
import sqlite3
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DB_PATH = os.path.join(DATA, "recipes.db")
ASSUMED_SERVINGS = 4  # the dataset has no serving sizes
_lock = threading.Lock()

COURSES = {"main": "Main dishes", "soup": "Soups & stews", "salad": "Salads", "side": "Side dishes",
           "breakfast": "Breakfast", "snack": "Snacks & starters", "dessert": "Desserts", "baking": "Bread & baking",
           "drinks": "Drinks", "other": "Other (sauces, preserves …)"}


PATTERN = "recipes.db.xz.part*"


def _parts() -> list[str]:
    """The database parts: normally in data/, but also found in the app folder or any subfolder."""
    for folder in (DATA, HERE):
        found = sorted(glob.glob(os.path.join(folder, PATTERN)))
        if found:
            return found
    found = glob.glob(os.path.join(HERE, "**", PATTERN), recursive=True)
    return sorted(found, key=os.path.basename)


def signature() -> tuple:
    """Changes whenever the part files change, so the app reloads its cached index."""
    return tuple((os.path.basename(p), os.path.getsize(p)) for p in _parts())


def status() -> str:
    parts = _parts()
    if not parts:
        return ("No recipe database found. Upload the files recipes.db.xz.part01, part02 and part03 "
                "into a folder called data next to app.py.")
    return f"Found {len(parts)} database file(s): " + ", ".join(os.path.basename(p) for p in parts)


def available() -> bool:
    return os.path.exists(DB_PATH) or bool(_parts())


def _ensure_unpacked() -> None:
    with _lock:
        parts = _parts()
        newest = max((os.path.getmtime(p) for p in parts), default=0)
        if os.path.exists(DB_PATH) and os.path.getmtime(DB_PATH) >= newest:
            return
        if not parts:
            raise RuntimeError(status())
        os.makedirs(DATA, exist_ok=True)
        dec = lzma.LZMADecompressor()
        tmp = DB_PATH + ".tmp"
        try:
            with open(tmp, "wb") as fo:
                for p in parts:  # part01, part02, … in order
                    with open(p, "rb") as fi:
                        fo.write(dec.decompress(fi.read()))
            complete = dec.eof
        except lzma.LZMAError:
            complete = False
        if not complete:
            os.remove(tmp)
            raise RuntimeError("The recipe database is incomplete or damaged. " + status() +
                               ". All three files (part01, part02, part03) are needed, uploaded unchanged.")
        os.replace(tmp, DB_PATH)


def _conn() -> sqlite3.Connection:
    _ensure_unpacked()
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)


def _search_text(name: str, ingredients: list, categories: list) -> str:
    return " ".join([name, *[i[2] for i in ingredients], *categories]).lower()


def load_index() -> list[dict]:
    """Light metadata for every recipe (no steps). Raises RuntimeError with a readable message if missing."""
    with _conn() as c:
        rows = c.execute("SELECT id, name, course, cuisine, diet, allergens, time, categories, ingredients "
                         "FROM recipes").fetchall()
    return [{"id": i, "name": n, "course": co, "cuisine": cu, "diet": d,
             "allergens": [a for a in (al or "").split(",") if a], "time": t,
             "search": _search_text(n, json.loads(ing), json.loads(cats)), "source": "db"}
            for i, n, co, cu, d, al, t, cats, ing in rows]


def get(rid: str) -> dict | None:
    """The full recipe in the same shape as the built-in recipes."""
    if not available():
        return None
    with _conn() as c:
        row = c.execute("SELECT id, name, description, course, cuisine, diet, allergens, time, categories, "
                        "ingredients, steps FROM recipes WHERE id = ?", (rid,)).fetchone()
    if not row:
        return None
    i, n, desc, co, cu, d, al, t, cats, ing, steps = row
    cats, ing = json.loads(cats), json.loads(ing)
    return {"id": i, "name": n, "description": desc, "course": co, "cuisine": cu, "diet": d,
            "allergens": [a for a in (al or "").split(",") if a], "time": t, "tags": cats,
            "ingredients": ing, "steps": json.loads(steps), "search": _search_text(n, ing, cats),
            "servings": ASSUMED_SERVINGS, "source": "db"}
