"""TOML for the tests file: read with the standard library (tomllib), written here.

The writer keeps the file readable by hand: nested dicts become [table] sections, lists of dicts
become [[array]] sections (also inside other arrays: a header always refers to the latest item),
everything else is written inline. TOML has no null: None is written as the string "null" and read
back as None (so a hand-written "null" means 'not set' too).
"""
import math
import re

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    tomllib = None

NULL = "null"
_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k):
    k = str(k)
    return k if _BARE.match(k) else _string(k)


def _string(s):
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _scalar(v):
    if v is None:
        return _string(NULL)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        r = f"{v:.12g}"
        if float(r) != v and abs(float(r) - v) > 1e-12 * abs(v):
            r = repr(v)
        return r if ("." in r or "e" in r or "n" in r) else r + ".0"
    if isinstance(v, str):
        return _string(v)
    if hasattr(v, "item"):                       # numpy scalar
        return _scalar(v.item())
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _inline(v):
    if isinstance(v, dict):
        if not v:
            return "{}"
        return "{ " + ", ".join(f"{_key(k)} = {_inline(x)}" for k, x in v.items()) + " }"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_inline(x) for x in v) + "]"
    if hasattr(v, "tolist"):                     # numpy array
        return _inline(v.tolist())
    return _scalar(v)


def _is_table_list(v):
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, dict) for x in v)


def _small(v):
    """A dict short and flat enough to stay on one line."""
    return (isinstance(v, dict) and all(not isinstance(x, (dict, list, tuple)) for x in v.values())
            and len(_inline(v)) <= 80)


def _table(lines, path, d):
    """Write the dict d whose header (at path) has been written: plain keys first, then sub-tables."""
    later = []
    for k, v in d.items():
        if (isinstance(v, dict) and v and not _small(v)) or _is_table_list(v):
            later.append((k, v))
        else:
            lines.append(f"{_key(k)} = {_inline(v)}")
    for k, v in later:
        p = path + [_key(k)]
        if isinstance(v, dict):
            lines.append("")
            lines.append(f"[{'.'.join(p)}]")
            _table(lines, p, v)
        else:
            for item in v:
                lines.append("")
                lines.append(f"[[{'.'.join(p)}]]")
                _table(lines, p, item)


def dumps(data, header=()):
    """TOML text of the dict data; header lines become '#' comments at the top."""
    lines = [f"# {h}" if h else "#" for h in header]
    if lines:
        lines.append("")
    _table(lines, [], data)
    return "\n".join(lines).strip("\n") + "\n"


def _nulls(x):
    if isinstance(x, dict):
        return {k: _nulls(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nulls(v) for v in x]
    return None if x == NULL else x


def loads(text):
    if tomllib is None:
        raise RuntimeError("reading TOML needs Python 3.11 or newer")
    return _nulls(tomllib.loads(text))


def load(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        return loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from None


def dump(path, data, header=()):
    import os
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(dumps(data, header))
    os.replace(tmp, path)
