"""Saved switch models: a MOSFET's, diode's or converter IC's datasheet values, stored once by part
number and applied to every board where a footprint has that value.

The store is a JSON file next to the component library ({key: {"kind": "fet" | "diode", "model": {...}}}).
A key is the part number (the footprint's Value, case-insensitive); an IC's internal switches are
saved as "<IC value> high-side" / "<IC value> low-side".
"""
import json
import os

FILE = "switch_models.json"


def path_for(library_path):
    return os.path.join(os.path.dirname(library_path) or ".", FILE)


def load(path):
    try:
        with open(path) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save(path, key, kind, model):
    d = load(path)
    m = {k: v for k, v in model.items() if k != "assumed"}
    d[key.strip()] = {"kind": kind, "model": m}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def remove(path, key):
    d = load(path)
    if d.pop(key, None) is not None:
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)


def key_of(board, d):
    """The store key of a cell part: its footprint's Value, or '<IC value> high-side' for an IC's switch."""
    fp = next((f for f in board.footprints if f.ref == d.ref), None)
    value = (fp.value if fp else d.ref).strip()
    if d.internal:
        return f"{value} {'high' if d.role == 'upper' else 'low'}-side"
    return value


def find(models, key):
    k = key.lower()
    return next((v for n, v in models.items() if n.lower() == k), None)


def apply(board, s, models):
    """Give every switch / diode of cell s whose part has a saved model that model.  Returns the keys
    of the parts it changed."""
    done = []
    for d in s.devices:
        if d.kind not in ("fet", "diode"):
            continue
        hit = find(models, key_of(board, d))
        if hit is None or hit.get("kind") != d.kind:
            continue
        keep = {k: d.model[k] for k in ("channel",) if k in d.model and "channel" not in hit["model"]}
        d.model = {**d.model, **hit["model"], **keep, "assumed": False}
        done.append(d.key)
    return done
