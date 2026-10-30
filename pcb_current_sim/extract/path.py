"""Trace the copper between pads as a graph of track pieces, via barrels, zones and elements.

Nodes are track ends, pads, via layers and zones; edges are straight track pieces
(arcs become polylines), via barrel pieces between consecutive layers, links from a
zone to each node that touches it ("zonelink": the zone is a multi-terminal component
meshed later), and user elements between two pads (which may join different nets).
Track ends that touch (within half a width), T-junctions on the middle of another
track, and ends inside a pad or a via ring are joined.
"""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np

SNAP = 1e-3          # mm: ends closer than this are the same point


class SimpleUF:
    def __init__(self):
        self.p = []

    def add(self):
        self.p.append(len(self.p))
        return len(self.p) - 1

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


@dataclass
class Node:
    pos: tuple
    layer: str = None
    kind: str = "pt"            # pt | pad | via | zone | ref (the return, for elements to a return-net pad)
    label: str = ""
    pad: object = None          # board Pad for pad nodes
    via: int = None             # via index for via nodes
    zone: int = None            # board zone index for zone nodes
    net: str = ""


@dataclass
class Edge:
    a: int
    b: int
    kind: str                   # track | barrel | zonelink | element
    layer: str = None
    width: float = 0.0          # mm
    p0: tuple = None
    p1: tuple = None
    length: float = 0.0         # mm (barrel: z length)
    track: int = None           # board track index
    via: int = None             # board via index (barrel)
    layers: tuple = None        # barrel: (upper layer, lower layer)
    term: tuple = None          # zonelink: how the node touches the zone: ("pad", Pad) | ("via", r) | ("end", r)
    el: object = None           # element: elements.Element


@dataclass
class Graph:
    net: str                    # net of port 1
    nodes: list
    edges: list
    ports: list                 # node per port pad id
    port_ids: list
    nets: list = field(default_factory=list)      # every net in the graph
    warnings: list = field(default_factory=list)

    def adjacency(self):
        adj = {}
        for k, e in enumerate(self.edges):
            adj.setdefault(e.a, []).append((e.b, k))
            adj.setdefault(e.b, []).append((e.a, k))
        return adj

    def shortest(self, src, dst):
        """Edge indices of the shortest path src -> dst (by length), or None."""
        adj, dist, prev = self.adjacency(), {src: 0.0}, {}
        q = [(0.0, src)]
        while q:
            d, u = heapq.heappop(q)
            if u == dst:
                break
            if d > dist.get(u, math.inf):
                continue
            for v, k in adj.get(u, []):
                nd = d + self.edges[k].length
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, (u, k)
                    heapq.heappush(q, (nd, v))
        if dst not in dist:
            return None
        out, u = [], dst
        while u != src:
            u, k = prev[u]
            out.append(k)
        return out[::-1]


def _pieces(board, nets):
    """Straight pieces (layer, width, p0, p1, track index) of the nets' tracks."""
    out = []
    for k, t in enumerate(board.tracks):
        if t.net not in nets:
            continue
        if t.kind == "segment":
            pts = [t.start, t.end]
        else:
            g = t.geom
            pts = [g.parts[0].a] + [c.b for c in g.parts]
            pts[0], pts[-1] = t.start, t.end
        for a, b in zip(pts[:-1], pts[1:]):
            if math.dist(a, b) > SNAP:
                out.append((t.layer, t.width, tuple(a), tuple(b), k))
    return out


def _seg_dist(P, a, b):
    """Distance of points P (n, 2) to segment a-b, and the projection parameter."""
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    L2 = float(d @ d)
    s = np.clip(((P - a) @ d) / L2, 0, 1) if L2 > 0 else np.zeros(len(P))
    return np.hypot(*(P - a - s[:, None] * d).T), s


_RING = np.linspace(0, 2 * np.pi, 16, endpoint=False)


def touches(geom, pos, r):
    """True if the disc of radius r at pos (a track end with half its width, a via ring) overlaps
    the shape: its centre or a point on its rim lies inside."""
    x = np.concatenate([[pos[0]], pos[0] + r * np.cos(_RING)])
    y = np.concatenate([[pos[1]], pos[1] + r * np.sin(_RING)])
    return bool(geom.inside(x, y).any())


def reachable_nets(board, start, elements, return_nets=()):
    """Nets reached from `start` through enabled elements (both pads must have a net).  Return
    nets are never entered: an element to a return-net pad ends at the reference (REF)."""
    pad_net = {p.id: p.net for p in board.pads if p.layers}
    nets, grown = {start}, True
    while grown:
        grown = False
        for e in elements:
            na, nb = pad_net.get(e.a), pad_net.get(e.b)
            if e.enabled and na and nb and (na in nets) != (nb in nets) and not ({na, nb} & set(return_nets)):
                nets |= {na, nb}
                grown = True
    return nets


def build(board, port_ids, stubs=True, layer_z=None, elements=(), return_nets=()):
    """Graph of the copper reachable from the first port pad (its net, plus nets joined by
    `elements`).  stubs: keep every branch reachable from port 1 (branches to unselected pads
    become open stubs); else only the shortest paths from port 1 to the other ports.
    layer_z: {layer: z (m)} for barrel lengths."""
    pads = {}
    for p in board.pads:
        if p.layers and p.kind != "np_thru_hole":
            pads.setdefault(p.id, []).append(p)
    for pid in port_ids:
        if pid not in pads:
            raise ValueError(f"Pad {pid} not found on the board.")
    net = pads[port_ids[0]][0].net
    if not net:
        raise ValueError(f"Pad {port_ids[0]} has no net.")
    if len(set(port_ids)) < 2:
        raise ValueError("Pick at least two different pads.")
    ret = set(return_nets)
    for pid in port_ids:
        if pads[pid][0].net in ret:
            raise ValueError(f"Pad {pid} is on the return net {pads[pid][0].net}: a port must be on the signal "
                             "side (the return is the subcircuit's REF pin).")
    elements = [e for e in elements if e.enabled]
    nets = reachable_nets(board, net, elements, ret)
    for pid in port_ids[1:]:
        if pads[pid][0].net not in nets:
            raise ValueError(f"Pad {pid} is on net {pads[pid][0].net or '(none)'}, which no element joins to "
                             f"{net}: add the part between them as an element.")

    uf, nodes = SimpleUF(), []

    def node(**kw):
        nodes.append(Node(**kw))
        return uf.add()

    # track pieces, split at T-junctions
    raw = _pieces(board, nets)
    nets_of = {k: board.tracks[k].net for *_, k in raw}
    # split points: other tracks' ends, and vias / pads sitting on the middle of a track
    ends = [(L, a, nets_of[k], 0.0) for L, w, a, b, k in raw] + [(L, b, nets_of[k], 0.0) for L, w, a, b, k in raw]
    ends += [(L, v.pos, v.net, v.size / 2) for v in board.vias if v.net in nets for L in v.layers]
    ends += [(L, p.pos, p.net, 0.0) for ps in pads.values() for p in ps if p.net in nets for L in p.layers]
    splits = {i: [] for i in range(len(raw))}
    by_layer = {}
    for i, (L, w, a, b, _) in enumerate(raw):
        by_layer.setdefault(L, []).append(i)
    for L, ids in by_layer.items():
        sel = [e for e in ends if e[0] == L]
        P = np.array([e[1] for e in sel])
        enet = np.array([e[2] for e in sel])
        rad = np.array([e[3] for e in sel])
        if not len(P):
            continue
        for i in ids:
            _, w, a, b, tk = raw[i]
            d, s = _seg_dist(P, a, b)
            Ls = math.dist(a, b)
            hit = (d <= w / 2 + rad) & (s * Ls > w / 2) & ((1 - s) * Ls > w / 2) & (enet == nets_of[tk])
            for k in np.flatnonzero(hit):
                splits[i].append(float(s[k]))
    pieces = []
    for i, (L, w, a, b, tk) in enumerate(raw):
        ss = sorted(set([0.0, 1.0] + splits[i]))
        pts = [(a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])) for s in ss]
        for p, q in zip(pts[:-1], pts[1:]):
            if math.dist(p, q) > SNAP:
                pieces.append((L, w, p, q, tk))
    pn = [(node(pos=p, layer=L, net=nets_of[tk]), node(pos=q, layer=L, net=nets_of[tk]))
          for L, w, p, q, tk in pieces]

    # join piece ends of one net that touch (within half the narrower width)
    for L in {pc[0] for pc in pieces}:
        items = [(pn[i][j], pieces[i][2 + j], pieces[i][1], nets_of[pieces[i][4]])
                 for i in range(len(pieces)) if pieces[i][0] == L for j in (0, 1)]
        P = np.array([it[1] for it in items])
        W = np.array([it[2] for it in items])
        order = np.argsort(P[:, 0])
        Ps, Ws = P[order], W[order]
        for ii in range(len(order)):
            jj = ii + 1
            while jj < len(order) and Ps[jj, 0] - Ps[ii, 0] <= max(Ws[ii], Ws[jj]) / 2 + SNAP:
                if math.dist(Ps[ii], Ps[jj]) <= min(Ws[ii], Ws[jj]) / 2 + SNAP and \
                        items[order[ii]][3] == items[order[jj]][3]:
                    uf.union(items[order[ii]][0], items[order[jj]][0])
                jj += 1

    end_nodes = [(pn[i][j], pieces[i][2 + j], pieces[i][0], pieces[i][1], nets_of[pieces[i][4]])
                 for i in range(len(pieces)) for j in (0, 1)]

    # pads: one node per pad id; track ends inside the pad join it
    pad_node = {}
    for pid, ps in pads.items():
        if ps[0].net not in nets:
            continue
        p0 = ps[0]
        n = node(pos=p0.pos, layer=p0.layers[0] if len(p0.layers) == 1 else None, kind="pad", label=pid, pad=p0,
                 net=p0.net)
        pad_node[pid] = n
        for p in ps:
            if p.geom is None:
                continue
            for nd, q, L, w, nn in end_nodes:
                if nn == p.net and L in p.layers and touches(p.geom, q, w / 2):
                    uf.union(n, nd)

    # vias: a node per spanned layer, barrel edges between them
    edges, via_nodes = [], []
    for k, v in enumerate(board.vias):
        if v.net not in nets:
            continue
        vn = {L: node(pos=v.pos, layer=L, kind="via", label=f"via #{k} {L}", via=k, net=v.net) for L in v.layers}
        via_nodes += [(n, L, v) for L, n in vn.items()]
        for nd, q, L, w, nn in end_nodes:
            if nn == v.net and L in vn and math.dist(q, v.pos) <= v.size / 2 + w / 2 + SNAP:
                uf.union(vn[L], nd)
        for pid, n in pad_node.items():       # via in pad
            p = nodes[n].pad
            if p.net == v.net and p.geom is not None and touches(p.geom, v.pos, v.size / 2):
                for L in p.layers:
                    if L in vn:
                        uf.union(vn[L], n)
        for L1, L2 in zip(v.layers[:-1], v.layers[1:]):
            dz = abs(layer_z[L2] - layer_z[L1]) * 1e3 if layer_z else 0.0
            edges.append(Edge(vn[L1], vn[L2], "barrel", length=dz, via=k, layers=(L1, L2), p0=v.pos, p1=v.pos))

    # zones: a component linked to every node that touches it
    for zi, z in enumerate(board.zones):
        if z.net not in nets:
            continue
        cands = [(nd, q, ("end", w / 2)) for nd, q, L, w, nn in end_nodes if L == z.layer and nn == z.net]
        cands += [(n, v.pos, ("via", v.size / 2)) for n, L, v in via_nodes if L == z.layer and v.net == z.net]
        cands += [(n, nodes[n].pos, ("pad", nodes[n].pad)) for n in pad_node.values()
                  if nodes[n].pad.net == z.net and z.layer in nodes[n].pad.layers]
        if not cands:
            continue
        P = np.array([q for _, q, _ in cands])
        inside = z.geom.inside(P[:, 0], P[:, 1])
        if not inside.any():
            continue
        c = tuple(z.geom.pts.mean(0))
        zn = node(pos=c, layer=z.layer, kind="zone", label=f"zone {z.net} {z.layer}", zone=zi, net=z.net)
        for (nd, q, term), ins in zip(cands, inside):
            if ins:
                edges.append(Edge(zn, nd, "zonelink", layer=z.layer, p0=q, p1=c, length=math.dist(q, c), term=term))

    # user elements between pads; a pad on a return net is the reference node
    ref = None
    for e in elements:
        ends = []
        for pid in (e.a, e.b):
            if pid in pad_node:
                ends.append(pad_node[pid])
            elif pid in pads and pads[pid][0].net in ret:
                if ref is None:
                    ref = node(pos=pads[pid][0].pos, kind="ref", label="REF", net=pads[pid][0].net)
                ends.append(ref)
        if len(ends) == 2 and ends[0] != ends[1]:
            pa, pb = nodes[ends[0]].pos, nodes[ends[1]].pos
            edges.append(Edge(ends[0], ends[1], "element", p0=pa, p1=pb, length=math.dist(pa, pb), el=e))

    for i, (L, w, p, q, tk) in enumerate(pieces):
        edges.append(Edge(pn[i][0], pn[i][1], "track", layer=L, width=w, p0=p, p1=q, length=math.dist(p, q),
                          track=tk))

    # compact: one node per union-find class (a pad node represents its class)
    rank = {"ref": 4, "pad": 3, "zone": 2, "via": 1, "pt": 0}
    rep = {}
    for n in range(len(nodes)):
        r = uf.find(n)
        if r not in rep or rank[nodes[n].kind] > rank[nodes[rep[r]].kind]:
            rep[r] = n
    newid, out_nodes = {}, []
    for n in range(len(nodes)):
        r = rep[uf.find(n)]
        if r not in newid:
            newid[r] = len(out_nodes)
            out_nodes.append(nodes[r])
    nid = [newid[rep[uf.find(n)]] for n in range(len(nodes))]
    out_edges, links = [], {}
    prio = {"pad": 0, "via": 1, "end": 2}
    for e in edges:
        e.a, e.b = nid[e.a], nid[e.b]
        if e.a == e.b:
            continue
        if e.kind == "zonelink":          # one link per (zone, node): a pad beats a via beats a track end
            k = (e.a, e.b)
            if k in links and prio[links[k].term[0]] <= prio[e.term[0]]:
                continue
            links[k] = e
            continue
        out_edges.append(e)
    out_edges += list(links.values())
    ports = [nid[pad_node[pid]] for pid in port_ids]
    g = Graph(net, out_nodes, out_edges, ports, list(port_ids), sorted(nets))
    if len(set(ports)) < len(ports):
        raise ValueError("Two of the chosen pads are the same copper node; pick different pads.")

    # keep what carries signal: the component of port 1 (with stubs) or the port-to-port paths
    adj = g.adjacency()
    seen, stack = {ports[0]}, [ports[0]]
    while stack:
        u = stack.pop()
        for v, _ in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    missing = [pid for pid, n in zip(port_ids, ports) if n not in seen]
    if missing:
        raise ValueError(f"No copper path from {port_ids[0]} to {', '.join(missing)} (tracks, vias, zones or "
                         "elements). Save the board in KiCad and refill the zones; the copper between them may "
                         "be missing.")
    if stubs:
        keep = {k for k, e in enumerate(g.edges) if e.a in seen}
    else:
        keep = set()
        for n in ports[1:]:
            keep |= set(g.shortest(ports[0], n))
    g.edges = [e for k, e in enumerate(g.edges) if k in keep]
    return g
