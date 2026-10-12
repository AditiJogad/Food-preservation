"""Train and evaluate Food Preservation Safety Prediction classifiers.

This educational machine-learning project is trained on generated food-storage
records and estimates preservation outcomes from simulated conditions.

Run from this directory: .venv313/bin/python model_training.py
PySpark is required for ingestion, validation, summaries, and feature selection.
Requires Java 17 or 21 and the full requirements.txt dependency set.
Scikit-learn trains the classification pipelines on the Spark-prepared inputs.
The full dataset is collected into local memory, appropriate for this project size.
Outputs: food_preservation_model.pkl, model_evaluation_report.csv, and four PNGs.
"""

from pathlib import Path
import json
import os
import gc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/food-preservation-mpl")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "food_preservation_dataset.csv"
MODEL_PATH = ROOT / "food_preservation_model.pkl"
REPORT_PATH = ROOT / "model_evaluation_report.csv"
RANDOM_STATE = 42
CATEGORICAL = ["food_type", "packaging_type", "storage_condition", "preservative_used"]
NUMERICAL = ["temperature_c", "humidity_percent", "storage_days", "ph_level",
             "microbial_count_cfu", "expiration_days_remaining"]
FEATURES = CATEGORICAL + NUMERICAL
EXPECTED = ["food_type", "temperature_c", "humidity_percent", "storage_days", "ph_level",
            "packaging_type", "storage_condition", "microbial_count_cfu", "preservative_used",
            "expiration_days_remaining", "preservation_score", "preserved"]


def load_with_pyspark(path):
    """Prepare and validate the training dataset using required local Spark."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError as exc:
        raise RuntimeError("Training requires PySpark and Java 17 or 21. Install the project requirements first.") from exc
    # Start Spark locally: no cluster is needed.
    spark = (SparkSession.builder.appName("FoodPreservationTraining")
             .master("local[2]").config("spark.ui.enabled", "false").getOrCreate())
    try:
        spark.sparkContext.setLogLevel("ERROR")
        # header uses column names; inferSchema identifies numeric columns.
        foods = spark.read.option("mode", "FAILFAST").csv(str(path), header=True, inferSchema=True).cache()
        if foods.columns != EXPECTED:
            raise ValueError(f"Expected columns: {EXPECTED}; got {foods.columns}")
        total = foods.count()
        if not total:
            raise ValueError("Dataset is empty.")
        # Compute missing counts in one distributed aggregation.
        missing = foods.select([
            F.sum(F.when(F.col(c).isNull() | F.isnan(F.col(c).cast("double")), 1).otherwise(0)).alias(c)
            if c in NUMERICAL + ["preservation_score"] else
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in EXPECTED
        ]).first().asDict()
        if any(missing.values()):
            raise ValueError(f"Missing values: {missing}")
        if foods.dropDuplicates().count() != total:
            raise ValueError("Dataset has exact duplicate rows.")
        if foods.filter(~F.col("preserved").isin("Yes", "No")).limit(1).count():
            raise ValueError("Target must contain only Yes and No.")
        for c in NUMERICAL:
            if foods.filter(F.abs(F.col(c)) == float("inf")).limit(1).count():
                raise ValueError(f"Infinite numeric values in {c}.")
        print("Spark dataset shape:", (total, len(foods.columns)))
        foods.printSchema()
        print("Spark missing-value counts:", missing)
        print("Spark target distribution:")
        foods.groupBy("preserved").count().withColumn("percent", F.col("count") / total * 100).show()
        print("Spark numerical summaries:")
        foods.select(NUMERICAL + ["preservation_score"]).summary().show()
        for c in CATEGORICAL:
            foods.groupBy(c).count().orderBy(c).show()
        # Exclude the derived score before crossing into the ML training stage.
        # toPandas collects the prepared inputs locally for scikit-learn.
        return foods.select(*FEATURES, "preserved").toPandas()
    finally:
        spark.stop()


def load_and_validate(path=DATA_PATH):
    df = load_with_pyspark(path)
    if df.isna().any().any():
        raise ValueError("Dataset has missing values.")
    if df.duplicated().any():
        raise ValueError("Dataset has exact duplicate rows.")
    if not set(df["preserved"]).issubset({"Yes", "No"}):
        raise ValueError("Target must contain only Yes and No.")
    if not np.isfinite(df[NUMERICAL].to_numpy()).all():
        raise ValueError("Numeric feature contains a non-finite value.")
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("Duplicate count:", int(df.duplicated().sum()))
    print("\nTarget distribution:\n", df["preserved"].value_counts(normalize=True).mul(100).round(2))
    print("\nPrepared numerical summary:\n", df[NUMERICAL].describe().round(3))
    for col in CATEGORICAL:
        print(f"\n{col} counts:\n{df[col].value_counts()}")
    return df


def make_preprocessor():
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", StandardScaler(), NUMERICAL),
    ])


def make_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=.08,
            subsample=.85, colsample_bytree=.85, objective="binary:logistic",
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    except ImportError:
        print("\nXGBoost is unavailable; continuing with Logistic Regression and Random Forest.")
    return models


def pipeline(estimator):
    return Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])


def metrics(y_true, prediction, probability):
    # Positive class is preserved (1); the No-class values track the project's safety concern.
    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision (Preserved)": precision_score(y_true, prediction, pos_label=1, zero_division=0),
        "Recall (Preserved)": recall_score(y_true, prediction, pos_label=1, zero_division=0),
        "F1 (Preserved)": f1_score(y_true, prediction, pos_label=1, zero_division=0),
        "Precision (Not Preserved)": precision_score(y_true, prediction, pos_label=0, zero_division=0),
        "Recall (Not Preserved)": recall_score(y_true, prediction, pos_label=0, zero_division=0),
        "F1 (Not Preserved)": f1_score(y_true, prediction, pos_label=0, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probability),
    }


def get_feature_importance(model, top_n=15):
    """Return actual tree-model importances with post-preprocessing feature names."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("The supplied model does not expose feature_importances_.")
    return pd.DataFrame({"feature": model.named_steps["preprocessor"].get_feature_names_out(),
                         "importance": estimator.feature_importances_}).nlargest(top_n, "importance").reset_index(drop=True)


def predict_food_preservation(model, food_type, temperature_c, humidity_percent, storage_days,
                              ph_level, packaging_type, storage_condition,
                              microbial_count_cfu, preservative_used, expiration_days_remaining):
    """Return the saved pipeline's live prediction and both class probabilities."""
    row = pd.DataFrame([dict(food_type=food_type, temperature_c=temperature_c,
        humidity_percent=humidity_percent, storage_days=storage_days, ph_level=ph_level,
        packaging_type=packaging_type, storage_condition=storage_condition,
        microbial_count_cfu=microbial_count_cfu, preservative_used=preservative_used,
        expiration_days_remaining=expiration_days_remaining)])[FEATURES]
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    p_yes = float(probabilities[classes.index(1)])
    p_no = float(probabilities[classes.index(0)])
    return {"prediction": "PRESERVED" if model.predict(row)[0] == 1 else "NOT PRESERVED",
            "probability_preserved": p_yes, "probability_not_preserved": p_no}


def plot_outputs(fitted, curve_data, X_test, y_test, all_results, selected_name):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    table = pd.DataFrame(all_results).set_index("Model")
    table[["Accuracy", "F1 (Not Preserved)", "ROC-AUC"]].plot.bar(ax=ax, ylim=(0, 1), rot=0)
    ax.set_ylabel("Score"); ax.set_title("Test-set model metrics")
    fig.tight_layout(); fig.savefig(ROOT / "model_metrics.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, probability in curve_data.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, probability):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1); ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC curves")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(ROOT / "roc_curve.png", dpi=180); plt.close(fig)

    prediction = fitted.predict(X_test)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, prediction, labels=[0, 1]),
        display_labels=["Not Preserved", "Preserved"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {selected_name}")
    fig.tight_layout(); fig.savefig(ROOT / "confusion_matrix.png", dpi=180); plt.close(fig)

    estimator = fitted.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        names = fitted.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.DataFrame({"Feature": names, "Importance": estimator.feature_importances_}).nlargest(15, "Importance")
        print("\nTop feature importances:\n", importance.to_string(index=False))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(importance["Feature"][::-1], importance["Importance"][::-1])
        ax.set(xlabel="Importance", title=f"Top features: {selected_name}")
        fig.tight_layout(); fig.savefig(ROOT / "feature_importance.png", dpi=180); plt.close(fig)


def main():
    df = load_and_validate()
    # preservation_score is deliberately excluded: it summarizes the simulator's inputs.
    X, y = df[FEATURES], df["preserved"].map({"No": 0, "Yes": 1}).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20,
        random_state=RANDOM_STATE, stratify=y)
    print(f"\nTrain/test rows: {len(X_train):,} / {len(X_test):,}")
    models = make_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows, result_rows = [], []
    final_model, curve_data = None, {}
    for name, estimator in models.items():
        print(f"\nCross-validating {name}...")
        # Calculate No-class F1 explicitly, since identifying spoiled food matters most.
        from sklearn.metrics import make_scorer
        scores = cross_validate(pipeline(estimator), X_train, y_train, cv=cv, n_jobs=1,
            scoring={"f1_no": make_scorer(f1_score, pos_label=0), "roc_auc": "roc_auc"})
        cv_row = {"Model": name, "CV Mean F1 (Not Preserved)": scores["test_f1_no"].mean(),
                  "CV Std F1 (Not Preserved)": scores["test_f1_no"].std(),
                  "CV Mean ROC-AUC": scores["test_roc_auc"].mean(),
                  "CV Std ROC-AUC": scores["test_roc_auc"].std()}
        cv_rows.append(cv_row)
    # Selection uses training-only CV: F1 No first, ROC-AUC next, then recall No in final test report.
    selected_name = sorted(cv_rows, key=lambda r: (r["CV Mean F1 (Not Preserved)"], r["CV Mean ROC-AUC"]), reverse=True)[0]["Model"]
    print("\nSelected from training-only CV:", selected_name)
    for name, estimator in models.items():
        print(f"Fitting and evaluating {name} on untouched test set...")
        fit = pipeline(estimator).fit(X_train, y_train)
        prediction = fit.predict(X_test)
        probability = fit.predict_proba(X_test)[:, list(fit.classes_).index(1)]
        curve_data[name] = probability
        row = {"Model": name, **metrics(y_test, prediction, probability),
               **next(r for r in cv_rows if r["Model"] == name), "Selected": name == selected_name}
        result_rows.append(row)
        print(f"\n{name} classification report:\n", classification_report(y_test, prediction,
            labels=[0, 1], target_names=["Not Preserved", "Preserved"], zero_division=0))
        print("Confusion matrix [TN, FP; FN, TP]:\n", confusion_matrix(y_test, prediction, labels=[0, 1]))
        if name == selected_name:
            final_model = fit
        else:
            del fit
            gc.collect()
    report = pd.DataFrame(result_rows)
    report.to_csv(REPORT_PATH, index=False)
    report.to_csv(ROOT / "model_metrics.csv", index=False)
    plot_outputs(final_model, curve_data, X_test, y_test, result_rows, selected_name)
    joblib.dump(final_model, MODEL_PATH)
    print("\nFinal comparison:\n", report.round(4).to_string(index=False))
    print("\nConfusion matrix meaning: TN=correctly Not Preserved; FP=food predicted Preserved when it is Not Preserved; FN=food predicted Not Preserved when it is Preserved; TP=correctly Preserved.")
    print("\nExample predictions:")
    examples = {
        "Good storage": dict(food_type="Milk", temperature_c=4, humidity_percent=60, storage_days=2, ph_level=6.6, packaging_type="Sealed", storage_condition="Refrigerated", microbial_count_cfu=300, preservative_used="No", expiration_days_remaining=5),
        "Poor storage": dict(food_type="Fish", temperature_c=31, humidity_percent=90, storage_days=5, ph_level=6.3, packaging_type="Open", storage_condition="Poor Storage", microbial_count_cfu=50_000_000, preservative_used="No", expiration_days_remaining=-3),
    }
    for name, values in examples.items(): print(name + ":", json.dumps(predict_food_preservation(final_model, **values), indent=2))
    base = examples["Good storage"].copy()
    checks = {"higher temperature": {**base, "temperature_c": 18}, "longer storage": {**base, "storage_days": 8},
              "higher microbial count": {**base, "microbial_count_cfu": 10_000_000}, "poor -> refrigerated": {**examples["Poor storage"], "storage_condition": "Refrigerated", "temperature_c": 4}}
    print("\nControlled sanity checks (model outputs; not forced):")
    for name, values in checks.items(): print(name + ":", json.dumps(predict_food_preservation(final_model, **values)))
    print(f"\nSaved complete pipeline: {MODEL_PATH.name}\nSaved evaluation report: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
