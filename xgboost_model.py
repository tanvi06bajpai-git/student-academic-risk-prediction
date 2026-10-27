"""Tuned XGBoost model for predicting academic risk (Fail/Withdrawn vs Pass/Distinction).

Hyperparameters come from a 40-candidate randomized search with 5-fold CV
(ROC-AUC 0.845, vs 0.830 for default XGBoost).
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("student_performance_feature_engineered.csv")

# Target: 1 = at risk (Fail/Withdrawn), 0 = Pass/Distinction.
# Drop the outcome columns and the id so the model can't cheat.
y = df["academic_risk"]
X = df.drop(columns=["final_result", "academic_risk", "id_student"])
cat_cols = X.select_dtypes(include=["object", "string"]).columns
X[cat_cols] = X[cat_cols].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(
    n_estimators=1128,
    learning_rate=0.0168,
    max_depth=6,
    min_child_weight=9,
    subsample=0.935,
    colsample_bytree=0.41,
    gamma=2.56,
    reg_alpha=0.0296,
    reg_lambda=1.55,
    max_cat_to_onehot=7,
    enable_categorical=True,
    tree_method="hist",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.4f}\n")
print(classification_report(y_test, pred, target_names=["Pass", "At risk"]))
print("Confusion matrix (rows = actual, cols = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, pred),
                   index=["Actual Pass", "Actual At risk"],
                   columns=["Pred Pass", "Pred At risk"]))

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features:")
print(importance.head(10).round(4).to_string())

model.save_model("xgboost_academic_risk.json")
print("\nSaved model to xgboost_academic_risk.json")
