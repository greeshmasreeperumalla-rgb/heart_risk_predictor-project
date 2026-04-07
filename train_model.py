import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import json

# 1. Load data
df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. Train multiple models & compare
models = {
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_pestimators=100, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=5000, random_state=42),
}

results = {}
best_model_name = None
best_auc = 0

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    cv  = cross_val_score(model, X_train, y_train, cv=5).mean()

    results[name] = {"accuracy": round(acc, 4), "auc": round(auc, 4), "cv": round(cv, 4)}
    print(f"{name}: Accuracy={acc:.4f}, AUC={auc:.4f}, CV={cv:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = model

print(f"\nBest Model: {best_model_name} (AUC={best_auc:.4f})")

# 5. Feature importance (for tree models)
feature_importance = {}
if hasattr(best_model, 'feature_importances_'):
    fi = best_model.feature_importances_
    feature_importance = dict(zip(X.columns.tolist(), [round(float(v), 4) for v in fi]))

# 6. Save model, scaler, metadata
joblib.dump(best_model, 'model/heart_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

meta = {
    "best_model": best_model_name,
    "features": X.columns.tolist(),
    "results": results,
    "feature_importance": feature_importance
}
with open('model/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("Model saved to model/heart_model.pkl")
print("Metadata saved to model/meta.json")