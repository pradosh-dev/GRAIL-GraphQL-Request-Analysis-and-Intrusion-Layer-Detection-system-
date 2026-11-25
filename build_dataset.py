"""build_dataset.py - turn the captured log into the labelled feature matrix."""
import json

import pandas as pd

from features import extract_features

rows = []
with open("graphql_traffic.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        feats = extract_features(rec["body"])
        if feats is None:                       # noise removal: malformed requests
            continue
        rows.append({"ts": rec["ts"], "client": rec["client"],
                     "label": rec["label"], **feats})

df = pd.DataFrame(rows)
df["t"] = pd.to_datetime(df["ts"], unit="s")
df = df.sort_values("t").set_index("t")
# temporal feature: requests from the same client in the previous 60 seconds
df["req_rate_60s"] = df.groupby("client")["ts"].transform(
    lambda s: s.rolling("60s").count())
df = df.reset_index(drop=True).drop(columns=["ts", "client"])
df.to_csv("features.csv", index=False)
print(df["label"].value_counts())
