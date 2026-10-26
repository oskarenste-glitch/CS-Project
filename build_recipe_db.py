"""Turn the Allrecipes dataset (CSV) into the app's recipe database.

Usage:
    python tools/build_recipe_db.py path/to/1_Recipe_csv.csv

Writes data/recipes.db.xz.part01, part02, … (an xz-compressed SQLite file split into parts of at most 2 MB).
The app joins and unpacks them on first start.

What this script does:
  1. Reads the CSV and removes duplicates (the same recipe is often listed in several categories;
     those categories are merged into one list).
  2. Drops recipes that aren't food for people (pet food).
  3. Derives fields the dataset doesn't have, from the categories, title and ingredient lines:
     course, cuisine, diet (meat / pesce / veg / vegan), allergens and an estimated cooking time.
  4. Parses each ingredient line into [quantity, unit, name] and converts US units to metric
     (cups -> ml, pounds/ounces -> g) so amounts can be scaled and merged on the shopping list.
All derived fields are keyword-based estimates, not verified data.
"""
from __future__ import annotations

import csv
import glob
import json
import lzma
import os
import re
import sqlite3
import sys
from fractions import Fraction

csv.field_size_limit(10 ** 8)

# ------------------------------------------------------------------ course & cuisine
COURSE_RULES = [  # first match wins, checked against all category + subcategory labels
    ("exclude", r"pet food"),
    ("drinks", r"cocktail|drink|margarita|lemonade|smoothie|eggnog|mojito|martini|bloody mary|punch|sangria|"
               r"\bgin\b|vodka|whiskey|\brum\b|tequila|shots"),
    ("dessert", r"cake|cookie|pie(?!s? crust)|\bpies\b|dessert|brownie|blondie|fudge|candy|cupcake|frosting|icing|pastr|"
                r"cobbler|crisps and crumbles|cheesecake|ice cream|mousse|flan|creme brulee|pavlova|macaroon|biscotti|"
                r"doughnut|danish|cinnamon roll|sticky bun|monkey bread|whoopie|divinity|fondant|gingersnap|lemon bars|"
                r"custard|pudding|jell-o|tart|energy balls|sweet"),
    ("baking", r"bread|muffin|biscuit|bagel|roll and bun|popover|english muffin|cornbread|scone|pizza dough|crust|"
               r"kolache|crackers"),
    ("breakfast", r"breakfast|brunch|pancake|waffle|french toast|omelet|frittata|oatmeal|overnight oats|granola|"
                  r"quiche|cereal|grits|strata"),
    ("soup", r"soup|stew|chowder|gazpacho|borscht|gumbo|bisque|chili"),
    ("salad", r"salad|coleslaw|slaw"),
    ("main", r"main dish|dinner|casserole|beef|pork|lamb|ground (turkey|chicken|pork|lamb)|chicken|turkey|burger|"
             r"pizza|lasagna|meatloaf|meatball|enchilada|fajita|burrito|taco|curry|stir-fr|pasta|fettuccin|linguine|"
             r"spaghetti|ziti|gnocchi|carbonara|manicotti|fried rice|paella|jambalaya|etouffee|seafood|steak|ribs|"
             r"roast|brisket|tenderloin|stroganoff|shepherd|cacciatore|piccata|marsala|cordon bleu|parmesan|teriyaki|"
             r"adobo|bulgogi|kalbi|pad thai|pancit|gyro|calzone|empanada|pierog|cabbage roll|lettuce wrap|"
             r"stuffed (main|bell)|sandwich|wraps|panini|sloppy joe|lunch|bowl|cooking for (one|two)|comfort food|"
             r"macaroni and cheese|goulash|dumpling|fried chicken|mushroom|winter squash|meal prep|freezer meal"),
    ("side", r"side dish|mashed potato|fries|baked beans|polenta|gravies|gravy|yams|potatoes|rice"),
    ("snack", r"appetizer|snack|dip|hummus|guacamole|salsa|nacho|bruschetta|deviled egg|popcorn|cheese ball|"
              r"jalapeno popper|antipasti|canape|party mix|tapas|wings|jerky|pate"),
    ("other", r".*"),
]
COURSE_RULES = [(c, re.compile(p)) for c, p in COURSE_RULES]

CUISINE_RULES = [  # checked against labels + title
    ("mexican", r"mexican|enchilada|fajita|burrito|taco|quesadilla|chilaquiles|nacho|guacamole|salsa|tamale|"
                r"chiles rellenos|carnitas|pozole|mole\b|tex-mex|margarita"),
    ("italian", r"italian|lasagna|spaghetti|fettuccin|linguine|risotto|ziti|gnocchi|carbonara|manicotti|"
                r"cacciatore|piccata|marsala|calzone|bruschetta|antipast|pesto|minestrone|parmesan|pizza|focaccia|"
                r"tiramisu|biscotti|panini|polenta|ravioli|tortellini|penne|alfredo|bolognese|osso buco"),
    ("indian", r"indian|diwali|curry|tikka|masala|tandoori|dal\b|dahl|naan|paneer|biryani|samosa|chutney|korma|"
               r"vindaloo|chana"),
    ("asian", r"chinese|korean|thai|japanese|filipino|vietnamese|asian|lunar new year|teriyaki|stir-fr|fried rice|"
              r"bulgogi|kalbi|pad thai|pancit|adobo|egg roll|lo mein|chow mein|sushi|ramen|udon|miso|kimchi|"
              r"wonton|dumpling|pho\b|satay|hoisin|sesame chicken|general tso|kung pao|sweet and sour"),
    ("middle eastern", r"middle eastern|lebanese|israeli|persian|turkish|hummus|falafel|shawarma|tahini|kebab|"
                       r"tabbouleh|baba ganoush|pita|shakshuka|jewish|kosher|passover|hanukkah|rosh hashanah|purim"),
    ("mediterranean", r"mediterranean|greek|gyro|tzatziki|spanish|paella|tapas|gazpacho|moussaka|spanakopita|"
                      r"souvlaki"),
    ("french", r"french|crepe|quiche|ratatouille|bourguignon|coq au vin|creme brulee|souffle|cassoulet|"
               r"croque|gratin|nicoise|provencal"),
    ("german", r"german|bavarian|schnitzel|bratwurst|sauerkraut|spaetzle|strudel|pretzel|kielbasa|pierog|"
               r"goulash|borscht|cabbage roll|polish"),
    ("african", r"african|moroccan|ethiopian|nigerian|tagine|jollof|kwanzaa|harissa"),
    ("american", r"southern|cajun|creole|louisiana|bbq|barbecue|burger|meatloaf|mac(aroni)? and cheese|"
                 r"buffalo|sloppy joe|cornbread|grits|gumbo|jambalaya|hawaiian|july 4|labor day|thanksgiving|"
                 r"fried chicken|chicken and dumplings|pot roast|chowder|biscuits and gravy|copycat"),
]
CUISINE_RULES = [(c, re.compile(p)) for c, p in CUISINE_RULES]

# ------------------------------------------------------------------ diet & allergens
def kw(words, exclude=None):
    """Regex for any of `words` (word-boundary at start), ignoring lines that match `exclude`."""
    return re.compile(r"\b(?:" + "|".join(words) + r")"), (re.compile(exclude) if exclude else None)


MEAT = kw(["beef", "steak", "chicken", "pork", "bacon", "ham\\b", "hams\\b", "sausage", "turkey", "lamb", "veal",
           "venison", "duck", "prosciutto", "pancetta", "pepperoni", "salami", "chorizo", "meatball", "hot dog",
           "frankfurter", "bratwurst", "kielbasa", "liver", "gelatin", "lard", "suet", "brisket", "spareribs",
           "short ribs", "ground meat", "goat", "rabbit", "bison", "corned", "pastrami", "mincemeat", "jell-o",
           "chuck", "sirloin", "ground round", "hamburger", "meatloaf mix", "tenderloin", "loin\\b", "oxtail",
           "tripe", "giblet", "drumstick", "ribeye", "rib-eye", "t-bone", "porterhouse", "filet mignon", "cutlet",
           "escalope", "flank", "rump", "chops?\\b", "wings?\\b", "thighs?\\b", "carnitas", "andouille", "capicola",
           "mortadella", "guanciale", "spam\\b", "meat\\b"],
          exclude=r"vegetarian|vegan|meatless|plant-based|meat-free|vegetable bouillon|veggie|mushroom|hamburger bun|"
                  r"no-chicken|beyond|impossible|goat cheese|hamburger bun|hamburger roll|meat-free|meat substitute|lamb's lettuce|steak sauce")
FISH = kw(["salmon", "tuna", "cod\\b", "tilapia", "halibut", "trout", "anchov", "sardine", "mackerel", "haddock",
           "catfish", "snapper", "swordfish", "mahi", "fish", "worcestershire", "caviar", "bonito", "pollock",
           "flounder", "sole\\b", "bass\\b"],
          exclude=r"vegan worcestershire|fish-free")
SHELLFISH = kw(["shrimp", "prawn", "crab", "lobster", "scallop", "clam", "mussel", "oyster", "crawfish", "crayfish",
                "squid", "calamari", "octopus"],
               exclude=r"oyster mushroom|imitation crab|crabapple|crab apple")
DAIRY = kw(["milk", "butter", "cheese", "cream", "yogurt", "yoghurt", "buttermilk", "ghee", "whey", "half-and-half",
            "mascarpone", "ricotta", "parmesan", "parmigiano", "mozzarella", "cheddar", "feta", "brie", "gouda",
            "gruyere", "custard", "ice cream", "queso", "paneer", "romano", "asiago", "provolone", "monterey jack",
            "colby", "swiss\\b", "creme fraiche", "crème fraîche", "kefir", "casein", "alfredo"],
           exclude=r"coconut milk|almond milk|oat milk|soy milk|rice milk|cashew milk|plant-based|dairy-free|vegan|"
                   r"non-dairy|nondairy|peanut butter|almond butter|cashew butter|nut butter|sunflower seed butter|"
                   r"apple butter|cocoa butter|butternut|cream of tartar|swiss chard|coconut cream|"
                   r"cream of coconut|butter beans|butter lettuce|cream soda|creamed corn|cream-style corn|"
                   r"milk of magnesia|ice cream cone|nutritional yeast|butter-flavored")
EGG = kw(["eggs?\\b", "egg ", "yolks?\\b", "egg whites?", "mayonnaise", "mayo\\b", "meringue", "aioli", "egg noodles"],
         exclude=r"eggplant|vegan mayo|egg-free|eggless|egg substitute|egg replacer|flax egg")
HONEY = kw(["honey"], exclude=r"honeydew|honeycrisp|honey crisp")
GLUTEN = kw(["flour", "wheat", "bread", "breadcrumb", "bun\\b", "buns\\b", "rolls?\\b", "tortilla", "pasta",
             "spaghetti", "noodle", "macaroni", "couscous", "bulgur", "barley", "rye", "semolina", "cracker",
             "pie crust", "pastry", "biscuit", "cake mix", "beer", "soy sauce", "panko", "pretzel", "crouton",
             "seitan", "farro", "spelt", "graham", "cookie", "wafer", "pizza dough", "phyllo", "filo", "dumpling",
             "lasagna", "fettuccin", "linguine", "penne", "ziti", "rigatoni", "gnocchi", "orzo", "ramen", "udon",
             "pita", "naan", "bagel", "croissant", "brownie mix", "baking mix", "bisquick", "stuffing", "malt",
             "teriyaki", "hoisin", "wonton", "egg roll", "baguette", "ciabatta", "focaccia", "english muffin",
             "breadstick", "crust", "cereal", "matzo", "ravioli", "tortellini", "manicotti", "shells", "cake\\b"],
            exclude=r"gluten-free|gluten free|almond flour|coconut flour|rice flour|tapioca|cornstarch|corn flour|"
                    r"chickpea flour|corn tortilla|rice noodle|rice paper|tamari|buckwheat|cauliflower crust|spaghetti squash|"
                    r"rice cracker|potato flour|oat flour|arrowroot|cassava|corn bread mix")
NUTS = kw(["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia", "peanut", "pine nut",
           "brazil nut", "nutella", "praline", "marzipan", "nuts\\b", "nut butter", "amaretto", "frangelico",
           "pignoli"],
          exclude=r"nutmeg|butternut|coconut|water chestnut|doughnut|nut-free")
SOY = kw(["soy", "tofu", "edamame", "miso", "tempeh", "soya", "tamari", "teriyaki", "hoisin"], exclude=None)
SESAME = kw(["sesame", "tahini", "hummus", "halvah", "halva", "furikake"], exclude=None)

ALLERGEN_RULES = [("Gluten", GLUTEN), ("Lactose", DAIRY), ("Egg", EGG), ("Nuts", NUTS), ("Soy", SOY),
                  ("Fish", FISH), ("Shellfish", SHELLFISH), ("Sesame", SESAME)]


def hits(rule, lines) -> bool:
    pat, excl = rule
    for ln in lines:
        if pat.search(ln):
            if excl is None or not excl.search(ln):
                return True
            # line matched an exclusion; check whether the keyword still appears outside the excluded phrase
            if pat.search(excl.sub(" ", ln)):
                return True
    return False


def diet_of(lines) -> str:
    if hits(MEAT, lines):
        return "meat"
    if hits(FISH, lines) or hits(SHELLFISH, lines):
        return "pesce"
    if hits(DAIRY, lines) or hits(EGG, lines) or hits(HONEY, lines):
        return "veg"
    return "vegan"


# ------------------------------------------------------------------ ingredient parsing & units
UNICODE_FRAC = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5", "⅖": "2/5", "⅗": "3/5",
                "⅘": "4/5", "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}
NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
QTY_RE = re.compile(rf"^\s*({NUM})(?:\s*(?:to|-|–)\s*({NUM}))?\s*")
PAREN_RE = re.compile(r"^\(([^)]*)\)\s*")

# unit word -> (metric unit, factor) ; factor None = keep as counted unit
UNIT_MAP = {
    "cup": ("ml", 240), "cups": ("ml", 240), "c.": ("ml", 240),
    "tablespoon": ("tbsp", 1), "tablespoons": ("tbsp", 1), "tbsp": ("tbsp", 1), "tbsp.": ("tbsp", 1),
    "teaspoon": ("tsp", 1), "teaspoons": ("tsp", 1), "tsp": ("tsp", 1), "tsp.": ("tsp", 1),
    "pound": ("g", 454), "pounds": ("g", 454), "lb": ("g", 454), "lbs": ("g", 454), "lb.": ("g", 454),
    "ounce": ("g", 28.35), "ounces": ("g", 28.35), "oz": ("g", 28.35), "oz.": ("g", 28.35),
    "quart": ("ml", 946), "quarts": ("ml", 946), "pint": ("ml", 473), "pints": ("ml", 473),
    "gallon": ("l", 3.785), "gallons": ("l", 3.785),
    "liter": ("l", 1), "liters": ("l", 1), "litre": ("l", 1), "ml": ("ml", 1), "milliliters": ("ml", 1),
    "gram": ("g", 1), "grams": ("g", 1), "g": ("g", 1), "kilogram": ("kg", 1), "kg": ("kg", 1),
    "pinch": ("pinch", 1), "pinches": ("pinch", 1), "dash": ("pinch", 1), "dashes": ("pinch", 1),
    "clove": ("cloves", 1), "cloves": ("cloves", 1), "can": ("can", 1), "cans": ("cans", 1),
    "package": ("pack", 1), "packages": ("packs", 1), "packet": ("pack", 1), "packets": ("packs", 1),
    "jar": ("jar", 1), "jars": ("jar", 1), "bunch": ("bunch", 1), "bunches": ("bunch", 1),
    "slice": ("slices", 1), "slices": ("slices", 1), "head": ("head", 1), "heads": ("head", 1),
    "stalk": ("stalks", 1), "stalks": ("stalks", 1), "sprig": ("sprigs", 1), "sprigs": ("sprigs", 1),
    "bottle": ("bottle", 1), "bottles": ("bottles", 1), "envelope": ("pack", 1), "envelopes": ("packs", 1),
    "container": ("pack", 1), "containers": ("packs", 1), "box": ("pack", 1), "boxes": ("packs", 1),
    "loaf": ("loaf", 1), "loaves": ("loaves", 1), "piece": ("pieces", 1), "pieces": ("pieces", 1),
}
FLUID_OZ = re.compile(r"^fluid\s+ounces?\b")


def to_num(s: str) -> float:
    s = s.strip()
    if " " in s:
        a, b = s.split(None, 1)
        return float(a) + float(Fraction(b))
    return float(Fraction(s))


def nice(q: float, unit: str) -> float:
    if unit in ("g", "ml"):
        if q >= 100:
            return round(q / 5) * 5
        return round(q)
    return round(q, 2)


def parse_line(raw: str):
    """'1 ½ cups white sugar' -> [360, 'ml', 'white sugar']"""
    s = raw.strip()
    for k, v in UNICODE_FRAC.items():
        s = re.sub(rf"(\d)\s*{k}", rf"\1 {v}", s).replace(k, v)
    m = QTY_RE.match(s)
    if not m:
        return [0, "", s]
    q = to_num(m.group(2) or m.group(1))
    rest = s[m.end():]
    size_note = ""
    pm = PAREN_RE.match(rest)
    if pm:  # "1 (14.5 ounce) can tomatoes"
        size_note = pm.group(1)
        rest = rest[pm.end():]
    if FLUID_OZ.match(rest.lower()):
        rest = FLUID_OZ.sub("", rest.lower(), count=1)
        unit, factor = "ml", 29.57
    else:
        word = rest.split(" ", 1)[0].lower() if rest else ""
        if word in UNIT_MAP:
            unit, factor = UNIT_MAP[word]
            rest = rest[len(word):]
        else:
            unit, factor = "", 1
    q = q * factor
    name = rest.strip(" ,")
    if size_note:
        sm = re.match(rf"({NUM})\s*(ounce|oz|pound|lb)s?", size_note)
        if sm and unit in ("can", "cans", "pack", "packs", "jar", "bottle", "bottles"):
            g = to_num(sm.group(1)) * (28.35 if sm.group(2).startswith(("o")) else 454)
            name = f"{name} ({round(g)} g each)"
        else:
            name = f"{name} ({size_note})"
    if unit == "ml" and q >= 1000:
        q, unit = q / 1000, "l"
    if unit == "g" and q >= 1000:
        q, unit = q / 1000, "kg"
    return [nice(q, unit), unit, name or s]


# ------------------------------------------------------------------ time estimate
TIME_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*(?:to|-|–)\s*(\d+(?:\.\d+)?))?\s*(minutes?|mins?|hours?|hrs?)\b", re.I)


def estimate_minutes(steps) -> int | None:
    total, found = 0.0, False
    text = " ".join(steps)
    for m in TIME_RE.finditer(text):
        v = float(m.group(2) or m.group(1))
        if m.group(3).lower().startswith("h"):
            v *= 60
        total += v
        found = True
    if re.search(r"overnight", text, re.I):
        total += 8 * 60
        found = True
    if not found:
        return None
    return int(round((total + 10) / 5) * 5)  # + ~10 min prep


# ------------------------------------------------------------------ build
PART_SIZE = 2_000_000  # bytes per file


def main(src: str, out_dir: str):
    groups: dict[tuple, dict] = {}
    with open(src, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["recipe_title"].strip(), row["ingredients"])
            g = groups.get(key)
            labels = {row["category"].strip(), row["subcategory"].strip()}
            if g:
                g["labels"] |= labels
            else:
                groups[key] = {"row": row, "labels": labels}

    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "recipes.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE recipes (
        id TEXT PRIMARY KEY, name TEXT, description TEXT, course TEXT, cuisine TEXT, diet TEXT,
        allergens TEXT, time INTEGER, categories TEXT, ingredients TEXT, steps TEXT)""")
    counts = {"kept": 0, "excluded": 0}
    rows = []
    for n, g in enumerate(groups.values(), 1):
        r = g["row"]
        labels = sorted(g["labels"])
        lab = " | ".join(labels).lower()
        course = next(c for c, p in COURSE_RULES if p.search(lab))
        if course == "exclude":
            counts["excluded"] += 1
            continue
        ing_raw = json.loads(r["ingredients"])
        steps = json.loads(r["directions"])
        lines = [x.lower() for x in ing_raw]
        title = r["recipe_title"].strip()
        cuis_text = lab + " | " + title.lower()
        cuisine = next((c for c, p in CUISINE_RULES if p.search(cuis_text)), "")
        allergens = [a for a, rule in ALLERGEN_RULES if hits(rule, lines)]
        ingredients = [parse_line(x) for x in ing_raw]
        rows.append((f"ar-{n}", title, r["description"].strip(), course, cuisine, diet_of(lines), ",".join(allergens),
                     estimate_minutes(steps), json.dumps(labels, ensure_ascii=False, separators=(',', ':')),
                     json.dumps(ingredients, ensure_ascii=False, separators=(',', ':')),
                     json.dumps(steps, ensure_ascii=False, separators=(',', ':'))))
        counts["kept"] += 1
    con.executemany("INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.execute("CREATE INDEX idx_course ON recipes(course)")
    con.commit()
    con.execute("VACUUM")
    con.close()
    with open(db_path, "rb") as fi:
        packed = lzma.compress(fi.read(), preset=9 | lzma.PRESET_EXTREME)
    os.remove(db_path)
    for old in glob.glob(os.path.join(out_dir, "recipes.db.*")):  # remove earlier builds
        os.remove(old)
    parts = [packed[i:i + PART_SIZE] for i in range(0, len(packed), PART_SIZE)]
    for k, chunk in enumerate(parts, 1):
        with open(os.path.join(out_dir, f"recipes.db.xz.part{k:02d}"), "wb") as fo:
            fo.write(chunk)
    print(f"{counts['kept']} recipes written, {counts['excluded']} excluded; "
          f"{len(packed) / 1e6:.1f} MB in {len(parts)} parts -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1], os.path.join(here, "..", "data"))
