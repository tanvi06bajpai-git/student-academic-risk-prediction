"""Add `likely_result` (model prediction) and `actual_result` columns to the dataset.

likely_result comes from 5-fold out-of-fold predictions, so every student is
predicted by a model that never saw their outcome. This keeps the mismatches
(e.g. predicted Fail but actually Pass) honest instead of memorised.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SRC = "student_performance_feature_engineered.csv"
OUT = "student_performance_with_results.csv"

df = pd.read_csv(SRC)

# Actual outcome: Pass/Distinction -> Pass, Fail/Withdrawn -> Fail (same split as academic_risk)
df["actual_result"] = df["academic_risk"].map({0: "Pass", 1: "Fail"})

# Features: everything except the outcome columns and the student id
X = df.drop(columns=["final_result", "academic_risk", "actual_result", "id_student"])
cat_cols = X.select_dtypes(include=["object", "string"]).columns
X[cat_cols] = X[cat_cols].astype("category")
y = df["academic_risk"]

model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(model, X, y, cv=cv)

df["likely_result"] = pd.Series(pred, index=df.index).map({0: "Pass", 1: "Fail"})

df.to_csv(OUT, index=False)

print(f"Saved {OUT} ({len(df)} rows)")
print(f"Accuracy: {(df.likely_result == df.actual_result).mean():.3f}")
print(pd.crosstab(df.likely_result, df.actual_result,
                  rownames=["likely_result"], colnames=["actual_result"]))
