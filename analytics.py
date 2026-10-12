"""Efficient dataset analytics for Food Preservation Safety Prediction."""
import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ["temperature_c", "humidity_percent", "storage_days", "ph_level",
                   "microbial_count_cfu", "expiration_days_remaining", "preservation_score"]


def add_categories(df):
    """Return a shallow analysis copy with expiration and score categories."""
    result = df.copy()
    result["expiration_status"] = np.select(
        [result.expiration_days_remaining.lt(0), result.expiration_days_remaining.eq(0), result.expiration_days_remaining.le(3)],
        ["Expired", "Expires Today", "Expiring Soon"], default="Valid")
    result["risk_category"] = pd.cut(result.preservation_score, bins=[-np.inf, 30, 60, 80, np.inf],
        labels=["Low", "Moderate", "High", "Critical"])
    return result


def get_kpis(df):
    """Return structured overall KPIs; preservation_score is analytics-only."""
    target = df.preserved.eq("Yes")
    return {"total_records": int(len(df)), "total_features": int(df.shape[1] - 1),
        "food_types": int(df.food_type.nunique()), "preserved_count": int(target.sum()),
        "not_preserved_count": int((~target).sum()), "preservation_rate": float(target.mean()),
        "average_temperature_c": float(df.temperature_c.mean()), "average_humidity_percent": float(df.humidity_percent.mean()),
        "average_storage_days": float(df.storage_days.mean()), "average_microbial_count_cfu": float(df.microbial_count_cfu.mean()),
        "average_ph": float(df.ph_level.mean()), "average_preservation_score": float(df.preservation_score.mean())}


def preservation_rate(df, group):
    """Return count and preserved percentage by any categorical grouping column."""
    output = df.assign(_preserved=df.preserved.eq("Yes")).groupby(group, observed=True).agg(
        records=("preserved", "size"), preservation_rate=("_preserved", "mean")).reset_index()
    return output.assign(preservation_rate=lambda x: x.preservation_rate * 100).sort_values("preservation_rate", ascending=False)
