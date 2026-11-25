"""features.py - AST-based feature extraction for GraphQL requests (GRAIL)."""
import json
import math
from collections import Counter

from graphql import GraphQLError, Visitor, parse, visit
from graphql.language import FieldNode, InlineFragmentNode, IntValueNode

PAGE_ARGS = {"first", "last", "limit"}


def shannon_entropy(text):
    if not text:
        return 0.0
    counts, n = Counter(text), len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


class QueryProfile(Visitor):
    """Walks the AST once and collects the structural counters."""

    def __init__(self):
        super().__init__()
        self.depth = self.max_depth = 0
        self.fields = self.aliases = self.directives = self.introspection = 0
        self.names = Counter()
        self.arg_text = []

    def enter_selection_set(self, node, *_):
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)

    def leave_selection_set(self, node, *_):
        self.depth -= 1

    def enter_field(self, node, *_):
        name = node.name.value
        self.fields += 1
        self.names[name] += 1
        self.aliases += node.alias is not None
        self.introspection += name.startswith("__")

    def enter_directive(self, node, *_):
        self.directives += 1

    def enter_argument(self, node, *_):
        self.arg_text.append(str(getattr(node.value, "value", "")))


def estimate_cost(selection_set):
    """Rough resolver-call estimate: every field costs 1, list arguments multiply."""
    total = 0
    for sel in selection_set.selections:
        if isinstance(sel, InlineFragmentNode):
            total += estimate_cost(sel.selection_set)
        elif isinstance(sel, FieldNode):
            mult = 1
            for arg in (sel.arguments or ()):
                if arg.name.value in PAGE_ARGS and isinstance(arg.value, IntValueNode):
                    mult = max(mult, int(arg.value.value))
            child = estimate_cost(sel.selection_set) if sel.selection_set else 0
            total += 1 + mult * child
    return total


def extract_features(raw_body):
    """Return a feature dict for one HTTP body, or None if it is malformed."""
    try:
        payload = json.loads(raw_body)
    except (TypeError, ValueError):
        return None
    batch = payload if isinstance(payload, list) else [payload]
    prof, cost, qlen = QueryProfile(), 0, 0
    for item in batch:
        query = item.get("query") if isinstance(item, dict) else None
        if not query:
            return None
        try:
            doc = parse(query)
        except GraphQLError:
            return None
        visit(doc, prof)
        qlen += len(query)
        for definition in doc.definitions:
            if getattr(definition, "selection_set", None):
                cost += estimate_cost(definition.selection_set)
    n_fields = max(prof.fields, 1)
    return {
        "batch_size": len(batch),
        "query_len": qlen,
        "max_depth": prof.max_depth,
        "field_count": prof.fields,
        "alias_count": prof.aliases,
        "alias_density": prof.aliases / n_fields,
        "dup_field_max": max(prof.names.values(), default=0),
        "directive_count": prof.directives,
        "est_cost": min(cost, 10**9),
        "introspection_count": prof.introspection,
        "introspection_flag": int(prof.introspection > 0),
        "arg_entropy": shannon_entropy(" ".join(prof.arg_text)),
    }


if __name__ == "__main__":                      # quick demo used for Figure 8.1
    demo = {
        "benign": '{"query": "query { pastes { id title owner { name } } }"}',
        "deep_nesting": json.dumps({"query": "query { " + "pastes { owner { " * 8
                                    + "id" + " } }" * 8 + " }"}),
        "introspection": '{"query": "query { __schema { types { name } } }"}',
    }
    for label, body in demo.items():
        print(label, extract_features(body))
