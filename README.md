# GRAIL — GraphQL Request Analysis & Intrusion Layer

A query-structure-aware machine learning framework for detecting **GraphQL introspection** and **resource-exhaustion abuse** in the live request path.

---

## Overview

Traditional API defenses (WAFs, rate limiters, and traffic-volume monitors) inspect request frequency or flat payload text. In GraphQL, a **single low-volume HTTP request** can exhaust server CPU and memory via deeply nested relationships, alias overloading, batched operations, or schema introspection.

**GRAIL** parses incoming GraphQL payloads into their **Abstract Syntax Tree (AST)**, extracts structural, cost, introspection, argument-entropy, and temporal features, and classifies requests into 5 categories:

1. `benign` — Normal, low-complexity queries
2. `deep_nesting` — Recursive relationship depth abuse
3. `alias_overload` — Repeated field execution via GraphQL aliases
4. `batch_abuse` — Multiple operations packed into a single JSON array request
5. `introspection_abuse` — Schema enumeration via `__schema` and `__type`

---

## Repository Structure

| File | Role |
| :--- | :--- |
| `features.py` | Single-pass `graphql-core` AST visitor (`QueryProfile`), resolver cost estimator (`estimate_cost`), Shannon entropy, and 12-feature extractor (`extract_features`) |
| `capture_addon.py` | `mitmproxy` addon (`GraphQLLogger`) that intercepts `/graphql` traffic and logs labeled requests to `graphql_traffic.jsonl` |
| `traffic_gen.py` | Labeled synthetic traffic generator targeting a local DVGA instance across all 5 classes |
| `build_dataset.py` | ETL pipeline that parses captured logs, removes malformed queries, computes `req_rate_60s` per client IP, and exports `features.csv` |
| `train.py` | 80/20 stratified split + 5-fold `GridSearchCV` for Random Forest, Gradient Boosting, and SVM; exports `grail_models.joblib` and `model_comparison.csv` |
| `evaluate.py` | Generates model comparison, confusion matrix, feature importance, 4-way feature ablation, and inference latency plots |
| `detector.py` | Real-time request-path classifier (`classify`) combining deterministic `HARD_LIMITS` guardrails with confidence-thresholded ML alerting (`THRESHOLD = 0.80`) |
| `requirements.txt` | Python package dependencies |

---

## Extracted Features (13 Total)

- **Structural & Cost (AST):** `batch_size`, `query_len`, `max_depth`, `field_count`, `alias_count`, `alias_density`, `dup_field_max`, `directive_count`, `est_cost`
- **Introspection:** `introspection_count`, `introspection_flag`
- **Payload:** `arg_entropy` (Shannon entropy of argument values)
- **Temporal:** `req_rate_60s` (rolling 60-second request count per client IP)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Target GraphQL Server (DVGA)

```bash
docker run -d -t -p 5013:5013 -e WEB_HOST=0.0.0.0 --name dvga dolevf/dvga
```

### 3. Capture Labeled Traffic

In terminal 1, start the `mitmproxy` capture addon on port `8080`:

```bash
mitmdump -s capture_addon.py -p 8080
```

In terminal 2, run the traffic generator (100 rounds × 5 classes = 500 requests):

```bash
python traffic_gen.py
```

### 4. Build Dataset, Train & Evaluate

```bash
python build_dataset.py
python train.py
python evaluate.py
```

### 5. Run Real-Time Inference

```bash
python detector.py
```

Or import `classify` in your middleware/proxy:

```python
from detector import classify

payload = '{"query": "query { __schema { types { name } } }"}'
result = classify(payload, req_rate_60s=1)
print(result)
```

---

## Key Lab Results

- **Model Performance:** Random Forest, Gradient Boosting, and SVM achieved **1.00 macro F1** on the 100-request held-out test split (20 per class) in the synthetic DVGA lab benchmark.
- **Ablation Study:** AST structure-only features matched all features (**1.00 F1**), whereas traffic-only features (`query_len`, `req_rate_60s`) dropped to **~0.89 F1**.
- **Latency:** Median single-request model inference latency of **9.83 ms** (Random Forest) on a commodity CPU.
