"""train.py - 80/20 split, grid search + 5-fold CV for RF, GB and SVM."""
import json

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("features.csv")
X, y = df.drop(columns="label"), df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

SEARCH = {
    "RandomForest": (RandomForestClassifier(random_state=42),
                     {"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20]}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=42),
                         {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                          "max_depth": [2, 3]}),
    "SVM": (make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
            {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01]}),
}

models, rows = {}, []
for name, (estimator, grid) in SEARCH.items():
    gs = GridSearchCV(estimator, grid, cv=5, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    p, r, f1, _ = precision_recall_fscore_support(
        y_te, gs.predict(X_te), average="macro", zero_division=0)
    models[name] = gs.best_estimator_
    rows.append({"model": name, "cv_f1": gs.best_score_, "precision": p,
                 "recall": r, "f1": f1, "params": json.dumps(gs.best_params_)})

results = pd.DataFrame(rows)
results.to_csv("model_comparison.csv", index=False)
best = results.sort_values("cv_f1", ascending=False).iloc[0]["model"]
joblib.dump({"models": models, "best": best, "features": list(X.columns)},
            "grail_models.joblib")
X_tr.assign(label=y_tr).to_csv("train_set.csv", index=False)
X_te.assign(label=y_te).to_csv("test_set.csv", index=False)
print(results.drop(columns="params"), "\nSelected model:", best)
