"""detector.py - request-path detector: parse -> features -> classify -> alert."""
import time

import joblib
import pandas as pd

from features import extract_features

bundle = joblib.load("grail_models.joblib")
MODEL, FEATURES = bundle["models"][bundle["best"]], bundle["features"]
THRESHOLD = 0.80                 # alert only if class confidence >= threshold
HARD_LIMITS = {"max_depth": 50, "batch_size": 1000}   # heuristic safety guard


def classify(raw_body, req_rate_60s):
    t0 = time.perf_counter()
    feats = extract_features(raw_body)
    if feats is None:
        return {"label": "malformed", "confidence": 1.0, "alert": False}
    feats["req_rate_60s"] = req_rate_60s
    if any(feats[k] > v for k, v in HARD_LIMITS.items()):
        return {"label": "hard_limit", "confidence": 1.0, "alert": True}
    proba = MODEL.predict_proba(pd.DataFrame([feats])[FEATURES])[0]
    k = proba.argmax()
    label = str(MODEL.classes_[k])
    return {"label": label, "confidence": float(proba[k]),
            "alert": label != "benign" and proba[k] >= THRESHOLD,
            "latency_ms": (time.perf_counter() - t0) * 1000}


if __name__ == "__main__":
    body = '{"query": "query { __schema { types { name } } }"}'
    print(classify(body, req_rate_60s=1))
