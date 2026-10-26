# WG Organizer – a flat-share app in Python (Streamlit)

Three sections in one app:

- **Fair Split**: expenses with category, amount, who paid and who shares, balances, "Who pays whom?" and daily ECB exchange rates.
- **House Organizer**: month calendar (days when someone is away are marked red), cleaning rota that passes chores on automatically, tasks and a shared shopping list.
- **Cooking**: food preferences per person, a recipe database of about 25,000 recipes plus 45 hand-picked ones, filtered to what suits everyone at home, servings scaled to the number of people, search and filters (course, cuisine, time), and one-click transfer to the shopping list and Fair Split.

## Run it

You need Python 3.10 or newer.

```bash
cd wg-organizer
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (the `.gitignore` keeps your data out of it).
2. On https://share.streamlit.io choose **Create app → Deploy a public app from GitHub**.
3. Pick the repository and branch, set **Main file path** to `app.py` (or `wg-organizer/app.py` if the files sit in a subfolder) and click **Deploy**.

Note: on Streamlit Community Cloud the `wg.sqlite` file is wiped whenever the app restarts (new code pushed, or after the app has been asleep). It's fine for trying things out; for long-term use, run it on a machine at home or connect an external database.

## Recipe database

The `data` folder holds 24,979 recipes from the Allrecipes dataset (`1_Recipe_csv.csv`) as one SQLite database,
compressed with xz (5.7 MB) and split into three files of at most 2 MB: `recipes.db.xz.part01` to `part03`.
All parts are needed. The app joins and unpacks them on first start (`data/recipes.db`, not committed).

It was built with:

```bash
python tools/build_recipe_db.py path/to/1_Recipe_csv.csv
```

The script:
- removes duplicates (62,126 rows → 24,979 recipes; the same recipe was listed under several categories) and pet food,
- derives the fields the dataset doesn't have from categories, titles, ingredients and steps:
  course, cuisine, diet (meat / fish / vegetarian / vegan), allergens and an estimated cooking time,
- parses every ingredient line into amount, unit and name and converts US units to metric (cups → ml, pounds/ounces → g).

These derived fields are keyword-based estimates. The dataset has no serving sizes, so the app assumes 4 servings for
database recipes. Recipes show no prices; "Cooked it: split the cost" asks for the amount from the receipt.
The raw CSV/JSON files are too large for GitHub and aren't needed to run the app.

## Machine learning: the app learns what you like

After you press **Cooked it**, the Cooking page asks everyone who ate: **How was it?** Each person rates the dish
from 1 to 10 for **taste** and for **price** (value for money). You can also rate any recipe in its detail view.
The **overall rating = 50 % taste + 50 % price**. Favourites count as an 8.5 until they are rated.

For each flatmate, `ml.py` trains a small **ridge regression** (scikit-learn) that predicts this overall rating:
- **Features:** the words of the recipe title and ingredients (via a `HashingVectorizer`) plus its cuisine, course,
  diet and cooking-time bucket.
- **Training data:** the person's rated dishes (target: their overall rating) and favourites. A fixed random sample of
  200 other recipes is added as lightly weighted "neutral" examples (5.5), so the model learns what sets a person's
  well-rated and badly-rated dishes apart from a typical recipe.
- **Prediction:** a rating from 1 to 10 for every recipe. The three best-ranked dishes for the people at home get the
  label **"★ Daily recommendation"** on their recipe card.
- **Ranking:** the prediction adds up to about ±5 points per person on top of the fixed rules (favourites, likes, cuisines).
  How much it counts grows with the number of ratings (full weight from 8 ratings/favourites).
- **Safety:** diet, allergies and dislikes stay hard rules; the model only re-ranks dishes that are allowed.

Because price counts half, a dish that tasted great but was far too expensive (taste 9, price 2 → 5.5) ends up neutral,
so similar expensive dishes are not pushed up. The Cooking page shows, for each person, what the model has learned so far
(e.g. "Leans towards: indian food, ginger, curry · Less keen on: cheese, cream"). Models retrain automatically whenever a
rating or favourite changes (well under a second).

## Exchange rates

The app fetches the European Central Bank reference rates itself
(https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml) and reloads them at least every 3 hours.
If the ECB can't be reached, the last saved rates are used. Every expense stores its value at the rate of the day it was entered.

## Data

Everything is stored in `wg.sqlite` next to `app.py`. To back up, copy the file.
Set the environment variable `WG_DB` to use a different location.

## Sharing with your flatmates

Everyone has to open the same running app to see the same data:

- **On your home Wi-Fi**: start `streamlit run app.py --server.address 0.0.0.0` on a computer that stays on
  (e.g. a Raspberry Pi). The others open `http://<that-computer's-IP>:8501`.
- **On the internet**: e.g. PythonAnywhere, Render or Streamlit Community Cloud (see the note above).

There is no login. Anyone with the link can see and change everything.

## Files

| File | Contents |
|---|---|
| `app.py` | User interface (Streamlit) |
| `logic.py` | Logic: balances, settle-up transfers, chore rotation, calendar, recipe matching |
| `recipes.py` | 45 built-in recipes and option lists |
| `ml.py` | Machine learning: rating model (taste + price) per flatmate |
| `recipe_db.py` | Reading the recipe database |
| `tools/build_recipe_db.py` | Builds the recipe database from the raw dataset |
| `data/recipes.db.xz.part01–03` | The recipe database (compressed SQLite, split into 3 parts) |
| `fx.py` | Fetching ECB exchange rates |
| `store.py` | Saving to SQLite |
