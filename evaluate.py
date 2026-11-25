"""evaluate.py - produces the figures used in Chapter 8 of the report."""
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

bundle = joblib.load("grail_models.joblib")
models, best = bundle["models"], bundle["best"]
train, test = pd.read_csv("train_set.csv"), pd.read_csv("test_set.csv")
X_tr, y_tr = train.drop(columns="label"), train["label"]
X_te, y_te = test.drop(columns="label"), test["label"]
compare = pd.read_csv("model_comparison.csv").set_index("model")


def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=200)
    plt.close()


compare[["precision", "recall", "f1"]].plot.bar(rot=0, figsize=(6, 4), ylim=(0, 1))
plt.title("Model comparison (macro average, 20% test set)")
save("fig_8_2_model_comparison.png")

ConfusionMatrixDisplay.from_estimator(models[best], X_te, y_te,
                                      xticks_rotation=30, cmap="Blues")
plt.title(f"Confusion matrix - {best}")
save("fig_8_3_confusion_matrix.png")
print(classification_report(y_te, models[best].predict(X_te)))

rf = models["RandomForest"]
pd.Series(rf.feature_importances_, index=X_te.columns).sort_values().plot.barh(
    figsize=(6, 4))
plt.title("Random Forest feature importance")
save("fig_8_4_feature_importance.png")

GROUPS = {
    "all features": list(X_tr.columns),
    "traffic only": ["query_len", "req_rate_60s"],
    "structure only": [c for c in X_tr.columns if c != "req_rate_60s"],
    "no introspection": [c for c in X_tr.columns if not c.startswith("introspection")],
}
scores = {}
for name, cols in GROUPS.items():
    m = clone(rf).fit(X_tr[cols], y_tr)
    scores[name] = f1_score(y_te, m.predict(X_te[cols]), average="macro")
pd.Series(scores).plot.bar(rot=15, figsize=(6, 4), ylim=(0, 1))
plt.title("Ablation: macro F1 by feature set (Random Forest)")
save("fig_8_5_ablation.png")

lat = []
for i in range(min(500, len(X_te))):
    t0 = time.perf_counter()
    models[best].predict_proba(X_te.iloc[[i]])
    lat.append((time.perf_counter() - t0) * 1000)
plt.hist(lat, bins=40)
plt.xlabel("ms per request")
plt.title(f"Inference latency ({best}, median {np.median(lat):.2f} ms)")
save("fig_8_6_latency.png")
