"""Minimal s-expression parser for KiCad files.

Lists become Python lists; quoted strings and bare atoms both become str.
Quoted strings are wrapped in `Q` (a str subclass) so callers can tell
`(net "3")` from `(net 3)` when they need to.
"""
import re

_TOKEN = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')


class Q(str):
    """A quoted string atom."""


def parse(text):
    stack, cur = [], []
    for m in _TOKEN.finditer(text):
        t = m.group()
        if t == "(":
            stack.append(cur)
            cur = []
        elif t == ")":
            done, cur = cur, stack.pop()
            cur.append(done)
        elif t[0] == '"':
            cur.append(Q(t[1:-1].replace('\\"', '"').replace("\\\\", "\\")))
        else:
            cur.append(t)
    if stack:
        raise ValueError("unbalanced parentheses")
    return cur[0] if len(cur) == 1 else cur


def head(node):
    return node[0] if isinstance(node, list) and node else None


def find(node, key):
    """First child list whose head is `key`, or None."""
    for c in node[1:]:
        if isinstance(c, list) and c and c[0] == key:
            return c
    return None


def findall(node, key):
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == key]


def value(node, key, default=None, idx=1):
    c = find(node, key)
    return c[idx] if c is not None and len(c) > idx else default


def floats(node, key, n=2, default=None):
    c = find(node, key)
    if c is None:
        return default
    out = []
    for a in c[1:]:
        if isinstance(a, list):
            break
        try:
            out.append(float(a))
        except ValueError:
            continue
        if len(out) == n:
            break
    return tuple(out) if out else default
