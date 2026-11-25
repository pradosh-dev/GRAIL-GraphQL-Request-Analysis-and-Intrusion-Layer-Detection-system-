"""capture_addon.py - mitmproxy addon that logs every GraphQL request (GRAIL).
Run:  mitmdump -s capture_addon.py -p 8080
"""
import json
import time

from mitmproxy import http

LOG = open("graphql_traffic.jsonl", "a", encoding="utf-8")


class GraphQLLogger:
    def response(self, flow: http.HTTPFlow) -> None:
        if "/graphql" not in flow.request.path:
            return
        record = {
            "ts": time.time(),
            "client": flow.client_conn.peername[0],
            "status": flow.response.status_code,
            "latency_ms": (flow.response.timestamp_end
                           - flow.request.timestamp_start) * 1000,
            "body": flow.request.get_text(),
            # the traffic generators tag every request with its ground-truth class
            "label": flow.request.headers.get("X-Label", "unlabeled"),
        }
        LOG.write(json.dumps(record) + "\n")
        LOG.flush()


addons = [GraphQLLogger()]
