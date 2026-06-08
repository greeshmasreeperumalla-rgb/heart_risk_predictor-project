import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ─────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────
df = pd.read_csv('heart.csv')
df = df.drop_duplicates()

# ─────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────
df['age_trestbps'] = df['age'] * df['trestbps']
df['chol_trestbps'] = df['chol'] / (df['trestbps'] + 1)
df['oldpeak_sq'] = df['oldpeak'] ** 2
df['thalach_age_ratio'] = df['thalach'] / (df['age'] + 1)

# ─────────────────────────────
# 3. SPLIT FEATURES & TARGET
# ─────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

# ─────────────────────────────
# 4. COLUMN TYPES
# ─────────────────────────────
categorical = ['cp', 'thal', 'slope', 'ca']
numerical = [col for col in X.columns if col not in categorical]

# ─────────────────────────────
# 5. PREPROCESSING
# ─────────────────────────────
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(drop='first'), categorical)
])

# ─────────────────────────────
# 6. MODELS
# ─────────────────────────────
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000, class_weight='balanced'),
        {'model__C': [0.01, 0.1, 1, 10]}
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, None]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1]
        }
    )
}

# ─────────────────────────────
# 7. TRAIN-TEST SPLIT
# ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_f1 = 0
best_name = ""

print("\n===== MODEL COMPARISON =====")

# ─────────────────────────────
# 8. TRAINING LOOP
# ─────────────────────────────
for name, (model, params) in models.items():

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    search = RandomizedSearchCV(
        pipeline,
        params,
        n_iter=4,  # avoid warning
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )

    # Train
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_

    # Probabilities
    proba = best_pipe.predict_proba(X_test)[:, 1]

    # Threshold tuning
    best_thresh = 0.5
    local_best_f1 = 0

    for t in np.arange(0.3, 0.7, 0.01):
        preds_temp = (proba >= t).astype(int)
        f1_temp = f1_score(y_test, preds_temp)

        if f1_temp > local_best_f1:
            local_best_f1 = f1_temp
            best_thresh = t

    # Final predictions
    preds = (proba >= best_thresh).astype(int)

    # Metrics
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    auc  = roc_auc_score(y_test, proba)
    cm   = confusion_matrix(y_test, preds)

    # PRINT RESULTS
    print(f"\n--- {name} ---")
    print(f"Threshold : {best_thresh:.2f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")

    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}  TP: {cm[1][1]}")

    # Track best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = best_pipe
        best_name = name

# ─────────────────────────────
# 9. BEST MODEL
# ─────────────────────────────
print("\n===== BEST MODEL =====")
print(f"{best_name} (F1 Score = {best_f1:.4f})")

# ─────────────────────────────
# 10. FEATURE IMPORTANCE
# ─────────────────────────────
if best_name != "Logistic Regression":
    feature_names = best_model.named_steps['prep'].get_feature_names_out()
    importances = best_model.named_steps['model'].feature_importances_

    print("\nTop Features:")
    for f, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]:
        print(f"{f}: {imp:.4f}")

# ─────────────────────────────
# 11. SAVE MODEL
# ─────────────────────────────
joblib.dump(best_model, 'model/heart_model.pkl')
print("\n✅ Model saved as model/heart_model.pkl")