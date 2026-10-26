"""Machine learning: learn what each flatmate likes to eat.

After cooking, everyone who ate rates the dish from 1 to 10 for taste and for price (value for money).
The overall rating is 50 % taste + 50 % price. For every person the app trains a small ridge-regression
model (scikit-learn) that predicts this overall rating for any recipe, from
  - the dishes they rated (target: their overall rating),
  - their favourite recipes (counted as a good rating, FAVOURITE_SCORE, unless they rated them),
  - background: a fixed random sample of other recipes counted as neutral (5.5) and weighted lightly, so the
    model learns what sets a person's well-rated and badly-rated dishes apart from a typical recipe.

Recipes are turned into features with a HashingVectorizer: the words of the title and ingredients plus
tokens for cuisine, course, diet and cooking-time bucket. How much the prediction counts in the ranking
grows with the number of ratings (`confidence`), so the app starts with the fixed rules and relies more
on learning over time. Hard constraints (diet, allergies, dislikes) stay rules; the model only re-ranks
dishes that are allowed.

No Streamlit and no database here, so this module can be tested on its own.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer
from sklearn.linear_model import Ridge

import logic as L

N_FEATURES = 2 ** 15
BACKGROUND = 200          # random recipes used as "typical" examples
FULL_CONFIDENCE_AT = 8    # ratings + favourites needed before the model counts fully
MAX_BONUS = 5.0           # points a fully confident model can add or remove per person
NEUTRAL = 5.5             # middle of the 1–10 scale
FAVOURITE_SCORE = 8.5     # an unrated favourite counts like this rating
TASTE_WEIGHT = 0.5        # overall rating = 50 % taste + 50 % price
RIDGE_ALPHA = 0.3         # regularisation: higher = more cautious predictions

COOKING_WORDS = {
    "cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons", "tbsp", "tsp", "pound", "pounds", "ounce",
    "ounces", "chopped", "sliced", "diced", "minced", "fresh", "freshly", "ground", "large", "medium", "small",
    "taste", "divided", "needed", "optional", "finely", "coarsely", "thinly", "cut", "inch", "pieces", "piece",
    "salt", "pepper", "black", "water", "oil", "whole", "recipes", "recipe", "dishes", "dish", "main", "peeled",
    "drained", "rinsed", "packed", "softened", "melted", "room", "temperature", "beaten", "halved", "cubed",
    "trimmed", "plus", "more", "serving", "garnish", "each", "package", "can", "cans", "jar", "clove", "cloves",
    "allrecipes", "allstar", "allstars", "easy", "best", "quick", "homemade", "simple", "healthy", "low",
    "calorie", "fat", "sodium", "cholesterol", "high", "fiber", "kosher", "christmas", "easter", "day",
    "mother", "father", "july", "labor", "new", "year", "halloween", "chef", "john", "style", "white", "cooking",
    "purpose", "grated", "shredded", "unsalted", "boneless", "skinless", "frozen", "dried", "uncooked", "cooked",
    "light", "brown", "red", "green", "yellow", "extra", "virgin", "once", "twice", "cook", "eat", "make", "ahead",
    "leftovers", "dinner", "lunch", "recipes", "family", "favorite", "favorites", "ideas", "food",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | COOKING_WORDS)

_vectorizer = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, norm="l2", binary=True,
                                token_pattern=r"(?u)\b[a-z][a-z_]+\b", stop_words=STOP_WORDS)
_analyzer = _vectorizer.build_analyzer()


def time_bucket(minutes) -> str:
    if not minutes:
        return "time_unknown"
    return "time_quick" if minutes <= 30 else "time_medium" if minutes <= 60 else "time_long"


def document(recipe: dict) -> str:
    """The text the model sees for one recipe."""
    tokens = [L.recipe_text(recipe), f"course_{recipe.get('course') or 'main'}", f"diet_{recipe.get('diet') or 'meat'}",
              time_bucket(recipe.get("time"))]
    if recipe.get("cuisine"):
        tokens.append("cuisine_" + recipe["cuisine"].replace(" ", "_"))
    return " ".join(tokens)


def featurize(recipes: list[dict]) -> sparse.csr_matrix:
    return _vectorizer.transform([document(r) for r in recipes])


class Features:
    """Feature rows for a fixed list of recipes (the recipe database), built once and reused."""

    def __init__(self, recipes: list[dict]):
        self.ids = [r["id"] for r in recipes]
        self.row = {rid: i for i, rid in enumerate(self.ids)}
        self.X = featurize(recipes) if recipes else sparse.csr_matrix((0, N_FEATURES))

    def matrix(self, recipes: list[dict]) -> sparse.csr_matrix:
        """Rows for any list of recipes: cached rows where possible, the rest computed now."""
        if not recipes:
            return sparse.csr_matrix((0, N_FEATURES))
        known = [self.row.get(r["id"]) for r in recipes]
        if all(k is not None for k in known):
            return self.X[known]
        rows = []
        for r, k in zip(recipes, known):
            rows.append(self.X[k] if k is not None else featurize([r]))
        return sparse.vstack(rows).tocsr()


@dataclass
class TasteModel:
    member: str
    reg: Ridge
    n_rated: int
    n_favs: int
    confidence: float
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    reference: np.ndarray = field(default_factory=lambda: np.array([]))  # sorted predictions for typical recipes

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        """Predicted overall rating (1–10) of each recipe for this person."""
        if X.shape[0] == 0:
            return np.zeros(0)
        return np.clip(NEUTRAL + self.reg.predict(X), 1, 10)

    def percentile(self, score: float) -> float:
        """Share of typical recipes this person would rate lower than this one (0 = lowest, 1 = highest)."""
        if not len(self.reference):
            return 0.5
        return float(np.searchsorted(self.reference, score)) / len(self.reference)

    def bonus(self, score: float) -> float:
        """Points added to (or removed from) a recipe's ranking score: + for dishes the person is predicted to rate
        above their typical recipe, - below it, scaled by how much the model has learned so far."""
        return self.confidence * (MAX_BONUS * (2 * self.percentile(score) - 1) + (score - NEUTRAL) / 4.5)


def overall(taste: float, price: float) -> float:
    """The overall rating the model learns from: taste and price count 50 % each."""
    return TASTE_WEIGHT * taste + (1 - TASTE_WEIGHT) * price


def train(member_id: str, ratings: dict[str, float], favs: list[str], recipes_by_id: dict[str, dict],
          features: Features, background_pool: list[str]) -> TasteModel | None:
    """Train one person's model. `ratings` maps recipe id -> overall rating 1–10 (see `overall`).
    Favourites that haven't been rated count as FAVOURITE_SCORE. A rating always beats a favourite."""
    rated = {rid: float(v) for rid, v in ratings.items() if rid in recipes_by_id}
    fav_only = [rid for rid in favs if rid in recipes_by_id and rid not in rated]
    labelled = list(rated) + fav_only
    if not labelled:
        return None  # nothing rated yet: nothing to learn from
    rng = random.Random(zlib.crc32(member_id.encode()))
    pool = [rid for rid in background_pool if rid not in set(labelled)]
    bg = rng.sample(pool, min(BACKGROUND, len(pool)))

    ids = labelled + bg
    X = features.matrix([recipes_by_id[i] for i in ids])
    # target: how far above or below a neutral 5.5 the person rated the dish; typical recipes count as neutral
    y = np.array([rated[i] - NEUTRAL for i in rated] + [FAVOURITE_SCORE - NEUTRAL] * len(fav_only) + [0.0] * len(bg))
    # each rated dish counts fully; together the background weighs half as much as the rated dishes
    w = np.array([1.0] * len(labelled) + [0.5 * len(labelled) / max(len(bg), 1)] * len(bg))
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X, y, sample_weight=w)

    model = TasteModel(member=member_id, reg=reg, n_rated=len(rated), n_favs=len(fav_only),
                       confidence=min(1.0, len(labelled) / FULL_CONFIDENCE_AT))
    model.likes, model.dislikes = explain(reg, [recipes_by_id[i] for i in labelled])
    if bg:
        model.reference = np.sort(model.predict(X[len(labelled):]))
    return model


def _pretty(token: str) -> str:
    diets = {"diet_meat": "meat dishes", "diet_pesce": "fish dishes", "diet_veg": "vegetarian dishes",
             "diet_vegan": "vegan dishes"}
    if token in diets:
        return diets[token]
    for prefix, fmt in (("cuisine_", "{} food"), ("course_", "{}")):
        if token.startswith(prefix):
            return fmt.format(token[len(prefix):].replace("_", " "))
    return {"time_quick": "quick dishes", "time_medium": "30–60 min dishes", "time_long": "long cooking",
            "time_unknown": ""}.get(token, token)


def explain(reg: Ridge, recipes: list[dict], k: int = 6) -> tuple[list[str], list[str]]:
    """The words the model weighs most for and against, taken from the rated dishes themselves."""
    tokens = sorted({t for r in recipes for t in _analyzer(document(r))})
    if not tokens:
        return [], []
    idx = _vectorizer.transform(tokens).indices  # one hashed column per token (same order)
    coef = np.ravel(reg.coef_)[idx]
    order = np.argsort(coef)
    likes = [_pretty(tokens[i]) for i in order[::-1] if coef[i] > 0.05][:k]
    dislikes = [_pretty(tokens[i]) for i in order if coef[i] < -0.05][:k]
    return [x for x in likes if x], [x for x in dislikes if x]
