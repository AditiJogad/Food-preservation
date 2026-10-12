"""Reusable prediction functions for the existing Streamlit interface."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["food_type", "temperature_c", "humidity_percent", "storage_days", "ph_level",
            "packaging_type", "storage_condition", "microbial_count_cfu", "preservative_used",
            "expiration_days_remaining"]


@lru_cache(maxsize=1)
def load_model(path=None):
    """Load the complete preprocessing-and-model pipeline once per process."""
    return joblib.load(Path(path) if path else ROOT / "food_preservation_model.pkl")


def predict_food_preservation(food_type, temperature_c, humidity_percent, storage_days, ph_level,
                              packaging_type, storage_condition, microbial_count_cfu,
                              preservative_used, expiration_days_remaining):
    """Return a live model prediction and both class probabilities."""
    values = dict(food_type=food_type, temperature_c=temperature_c, humidity_percent=humidity_percent,
        storage_days=storage_days, ph_level=ph_level, packaging_type=packaging_type,
        storage_condition=storage_condition, microbial_count_cfu=microbial_count_cfu,
        preservative_used=preservative_used, expiration_days_remaining=expiration_days_remaining)
    model = load_model()
    row = pd.DataFrame([values])[FEATURES]
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    p_preserved = float(probabilities[classes.index(1)])
    p_not_preserved = float(probabilities[classes.index(0)])
    return {"prediction": "Preserved" if model.predict(row)[0] == 1 else "Not Preserved",
            "probability_preserved": p_preserved, "probability_not_preserved": p_not_preserved}


def run_sanity_scenarios():
    """Evaluate controlled, model-generated scenarios; no outcome is hard-coded."""
    base = dict(food_type="Milk", temperature_c=4, humidity_percent=60, storage_days=2, ph_level=6.6,
        packaging_type="Sealed", storage_condition="Refrigerated", microbial_count_cfu=300,
        preservative_used="No", expiration_days_remaining=5)
    scenarios = {"Good storage": base, "High temperature": {**base, "temperature_c": 22},
        "Long storage": {**base, "storage_days": 10}, "High microbial count": {**base, "microbial_count_cfu": 10_000_000},
        "Poor storage": {**base, "storage_condition": "Poor Storage", "packaging_type": "Open", "temperature_c": 30}}
    rows = []
    for name, inputs in scenarios.items():
        result = predict_food_preservation(**inputs)
        rows.append({"Scenario": name, "Prediction": result["prediction"],
                     "Probability Preserved": result["probability_preserved"],
                     "Probability Not Preserved": result["probability_not_preserved"]})
    return pd.DataFrame(rows)
