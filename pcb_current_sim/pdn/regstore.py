"""Saved regulator fits: a regulator's control loop fitted to its datasheet's load-transient plot,
stored once by part number and offered on every board that has that part.

The store is a JSON file next to the component library ({part number: RegulatorModel dict}); keys
match a footprint's Value, case-insensitive. The model transfers between boards because it is the
regulator's loop (an admittance), not a fit of one board's output impedance: the datasheet's test
capacitance is part of the model.
"""
import json
import os

FILE = "regulator_models.json"


def path_for(library_path):
    return os.path.join(os.path.dirname(library_path) or ".", FILE)


def load(path):
    try:
        with open(path) as fh:
            d = json.load(fh)
        return {k: v for k, v in d.items() if isinstance(v, dict)} if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path, d):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def save(path, key, model):
    """Store model (a RegulatorModel dict) under key; an existing key with other case is replaced."""
    key = key.strip()
    if not key:
        raise ValueError("A part number is needed.")
    d = {k: v for k, v in load(path).items() if k.lower() != key.lower()}
    d[key] = {**model, "part": model.get("part") or key}
    _write(path, d)


def remove(path, key):
    d = load(path)
    if d.pop(key, None) is not None:
        _write(path, d)


def find(models, key):
    k = key.strip().lower()
    return next((v for n, v in models.items() if n.lower() == k), None)


def on_board(board, s, models):
    """Stored part numbers among the footprints near the rail's source: parts with a pad on the rail,
    or on any net of the regulator output part (a buck's inductor: the switch node reaches the IC).
    [(key, ref)], the regulator output part's own value first."""
    nets = set(s.rail)
    vref = s.vrm.ref
    nets |= {p.net for p in board.pads if p.ref == vref and p.net}
    out = []
    for fp in board.footprints:
        if not fp.value or find(models, fp.value) is None:
            continue
        if fp.ref == vref or any(p.ref == fp.ref and p.net in nets for p in board.pads):
            out.append((next(k for k in models if k.lower() == fp.value.strip().lower()), fp.ref))
    out.sort(key=lambda kr: (kr[1] != vref, kr[1]))
    return out


def apply(board, s, models):
    """If the rail's regulator has no fitted loop yet and a stored part is on the board next to it,
    use that part's fit.  Returns the key used, or ''."""
    if s.vrm.model:
        return ""
    hits = on_board(board, s, models)
    if not hits:
        return ""
    key = hits[0][0]
    s.vrm.model = dict(models[key])
    s.vrm.kind = "loop"
    return key
