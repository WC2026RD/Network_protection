"""Generate undirected graph instances for the RPLS / MILP experiments.

Output format (plain text):
    n m
    u v c        (1-based node ids, integer reinforcing cost c)
    ...

Edge-count convention (as requested for these experiments):
    m = n * d, i.e. d=2 -> 100 edges, d=3 -> 150, d=4 -> 200 for n = 50.
    With that convention the ordinary average degree 2*m/n equals 2*d; both
    numbers are reported so the two readings stay distinguishable.

Supported instance families:
  er  : Erdos-Renyi G(n, m) drawn with exactly m = n*d edges
        (networkx.gnm_random_graph).
  sf  : Barabasi-Albert scale-free graph with attachment parameter
        m0 = max(1, floor(d / 2)); because 2*m0 can only take even values, the
        graph is then topped up with uniformly random missing links until it
        holds exactly m = n*d edges.

Both families are simple undirected graphs; instances are redrawn with
successive seeds until the graph is connected (edge connectivity is only
meaningful for connected graphs).  For n >= 100000 a vectorised path is used:
the graph is drawn once and its components are joined by one extra link each
(an i.i.d. G(n,m) at that density always contains isolated vertices), so the
ER files then hold m + (#components - 1) links.

Example:
    python gen_graph.py --type er -n 40 -d 4 --seed 1 -o inst_er_40_4.txt
    python gen_graph.py --type sf -n 40 -d 4 --seed 1 -o inst_sf_40_4.txt
    python gen_graph.py --batch            # 100 graphs x {er,sf} x d in {2,3,4}
                                           # -> graphs/<type>_d<d>/*.grf
    python gen_graph.py --batch -n 1000000 --count 10 --outdir graphs_large
"""
import argparse
import random

import networkx as nx
import numpy as np

# Graphs with at least this many nodes are drawn with the vectorised generators
# below (networkx cannot build a 10^6-node Barabasi-Albert graph in practice).
LARGE_N = 100000


def target_edges(n, d):
    """Requested edge count for these experiments: m = n * d."""
    return min(int(round(n * d)), n * (n - 1) // 2)


def make_er(n, d, seed):
    return nx.gnm_random_graph(n, target_edges(n, d), seed=seed)


def make_sf(n, d, seed):
    target = target_edges(n, d)
    m0 = max(1, min(int(d // 2), n - 1))
    g = nx.barabasi_albert_graph(n, m0, seed=seed)
    rnd = random.Random(seed * 6151 + 101)
    have = {tuple(sorted(e)) for e in g.edges()}
    while g.number_of_edges() < target:
        u = rnd.randrange(n)
        v = rnd.randrange(n)
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e in have:
            continue
        g.add_edge(u, v)
        have.add(e)
    return g


def connected_instance(kind, n, d, seed, max_try=200):
    """Draw graphs until one is connected; return (graph, seed_used)."""
    for k in range(max_try):
        s = seed + k
        g = make_er(n, d, s) if kind == "er" else make_sf(n, d, s)
        if nx.is_connected(g):
            return g, s
    raise RuntimeError("no connected instance found; increase average degree")


# ------------------------------------------------------- large-graph path
# At n = 10^6 an i.i.d. Erdos-Renyi graph of average degree 2m/n = 2d (d >= 2)
# still has isolated vertices, so "redraw until connected" is hopeless.  The
# vectorised path therefore draws the graph once and then joins the components
# with one extra link per component; the number of added links is reported.

def _er_keys(n, m, rng):
    """m distinct undirected links of G(n, m), packed as keys u*n+v (u<v)."""
    keys = np.empty(0, dtype=np.int64)
    while keys.size < m:
        need = m - keys.size
        k = int(need * 1.02) + 64
        u = rng.integers(0, n, k, dtype=np.int64)
        v = rng.integers(0, n, k, dtype=np.int64)
        keep = u != v
        u, v = u[keep], v[keep]
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        keys = np.union1d(keys, np.unique(lo * n + hi))
    if keys.size > m:
        keys = np.sort(keys[rng.choice(keys.size, m, replace=False)])
    return keys


def _ba_keys(n, m0, rnd):
    """Barabasi-Albert backbone with attachment m0, as packed keys."""
    from array import array

    m0 = max(1, min(int(m0), n - 1))
    us = array("i", [0] * m0)
    vs = array("i", range(1, m0 + 1))
    rep = array("i", [0] * m0)          # each node repeated once per incident link
    rep.extend(range(1, m0 + 1))
    for t in range(m0 + 1, n):
        L = len(rep)
        if m0 == 1:
            picks = (rep[rnd.randrange(L)],)
        else:
            got = set()
            while len(got) < m0:
                got.add(rep[rnd.randrange(L)])
            picks = tuple(got)
        for x in picks:
            us.append(x)
            vs.append(t)
        rep.extend([t] * m0)
        rep.extend(picks)
    u = np.frombuffer(us, dtype=np.int32).astype(np.int64)
    v = np.frombuffer(vs, dtype=np.int32).astype(np.int64)
    return np.sort(np.minimum(u, v) * n + np.maximum(u, v))


def _top_up(n, keys, m, rng):
    """Add uniform random links until `keys` holds exactly m distinct links."""
    keys = np.unique(keys)
    while keys.size < m:
        need = m - keys.size
        k = int(need * 1.05) + 64
        u = rng.integers(0, n, k, dtype=np.int64)
        v = rng.integers(0, n, k, dtype=np.int64)
        keep = u != v
        u, v = u[keep], v[keep]
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        cand = np.unique(lo * n + hi)
        if keys.size:
            pos = np.searchsorted(keys, cand)
            hit = pos < keys.size
            hit[hit] = keys[pos[hit]] == cand[hit]
            cand = cand[~hit]
        if cand.size == 0:
            continue
        if cand.size > need:
            cand = cand[rng.choice(cand.size, need, replace=False)]
        keys = np.union1d(keys, cand)
    return keys


def _join_components(n, keys, rng):
    """Make the graph connected with one added link per extra component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    u, v = keys // n, keys % n
    adj = coo_matrix((np.ones(u.size, dtype=np.int8), (u, v)), shape=(n, n))
    ncomp, labels = connected_components(adj, directed=False)
    if ncomp <= 1:
        return keys, 0
    sizes = np.bincount(labels)
    big = int(np.argmax(sizes))
    reps = np.empty(ncomp, dtype=np.int64)
    reps[labels] = np.arange(n, dtype=np.int64)     # any member per component
    big_nodes = np.flatnonzero(labels == big)
    others = np.flatnonzero(np.arange(ncomp) != big)
    tgt = big_nodes[rng.integers(0, big_nodes.size, others.size)]
    a = reps[others]
    extra = np.minimum(a, tgt) * n + np.maximum(a, tgt)
    return np.union1d(keys, extra), int(ncomp) - 1


def write_instance_large(kind, n, d, seed, out, min_cost, max_cost):
    """Vectorised generation and streaming write of one large instance."""
    import pandas as pd

    rng = np.random.default_rng(seed * 2654435761 % (1 << 63))
    m = target_edges(n, d)
    if kind == "er":
        keys = _er_keys(n, m, rng)
    else:
        m0 = max(1, min(int(d // 2), n - 1))
        keys = _top_up(n, _ba_keys(n, m0, random.Random(seed)), m, rng)
    keys, added = _join_components(n, keys, rng)

    frame = pd.DataFrame({"u": (keys // n) + 1, "v": (keys % n) + 1,
                          "c": rng.integers(min_cost, max_cost + 1, keys.size)})
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"{n} {keys.size}\n")
        frame.to_csv(f, sep=" ", header=False, index=False, lineterminator="\n")

    return dict(nodes=n, edges=int(keys.size), avg_degree=2.0 * keys.size / n,
                seed_used=seed, output=out, added_links=added)


def write_instance(kind, n, d, seed, out, min_cost, max_cost, max_try=200):
    """Draw one connected instance and write it to `out`; return a summary."""
    if n >= LARGE_N:
        return write_instance_large(kind, n, d, seed, out, min_cost, max_cost)
    g, used_seed = connected_instance(kind, n, d, seed, max_try)
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    edges = sorted((min(u, v), max(u, v)) for u, v in g.edges())

    rnd = random.Random(used_seed * 7919 + 17)
    costs = [rnd.randint(min_cost, max_cost) for _ in edges]

    with open(out, "w", encoding="utf-8") as f:
        f.write(f"{g.number_of_nodes()} {len(edges)}\n")
        for (u, v), c in zip(edges, costs):
            f.write(f"{u + 1} {v + 1} {c}\n")

    avg = 2.0 * len(edges) / g.number_of_nodes()
    return dict(nodes=g.number_of_nodes(), edges=len(edges), avg_degree=avg,
                seed_used=used_seed, output=out)


def batch(outdir, count, n, degrees, min_cost, max_cost):
    """Generate `count` .grf instances per (type, average degree) combination."""
    import os

    total = 0
    for kind in ("er", "sf"):
        for d in degrees:
            sub = os.path.join(outdir, f"{kind}_d{d}")
            os.makedirs(sub, exist_ok=True)
            avgs = []
            added = 0
            for i in range(1, count + 1):
                path = os.path.join(sub, f"{kind}_d{d}_{i:03d}.grf")
                s = write_instance(kind, n, d, 7919 * i, path,
                                   min_cost, max_cost, max_try=5000)
                avgs.append(s["avg_degree"])
                added += s.get("added_links", 0)
                total += 1
                if n >= LARGE_N:
                    print(f"  {path}: edges={s['edges']} "
                          f"added={s.get('added_links', 0)}", flush=True)
            print(f"{kind} d={d}: {count} files in {sub}  "
                  f"edges={int(round(n * d))}  "
                  f"avg_degree(2m/n)={sum(avgs) / len(avgs):.3f}  "
                  f"links added for connectivity={added}")
    print(f"total files: {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["er", "sf"], default="er")
    ap.add_argument("-n", type=int, default=50, help="number of nodes")
    ap.add_argument("-d", type=float, default=0.0, help="target average degree")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-cost", type=int, default=1)
    ap.add_argument("--max-cost", type=int, default=10)
    ap.add_argument("-o", "--out")
    ap.add_argument("--batch", action="store_true",
                    help="generate 100 instances per (type, degree) combination")
    ap.add_argument("--count", type=int, default=100, help="instances per combination")
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--outdir", default="graphs")
    args = ap.parse_args()

    if args.batch:
        batch(args.outdir, args.count, args.n, args.degrees,
              args.min_cost, args.max_cost)
        return

    if not args.out or args.d <= 0:
        ap.error("single-instance mode needs -d and -o")

    s = write_instance(args.type, args.n, args.d, args.seed, args.out,
                       args.min_cost, args.max_cost)
    print(f"type={args.type} nodes={s['nodes']} edges={s['edges']} "
          f"(m/n={s['edges'] / s['nodes']:.2f}) "
          f"avg_degree(2m/n)={s['avg_degree']:.3f} (target d={args.d}) "
          f"seed_used={s['seed_used']} output={s['output']}")


if __name__ == "__main__":
    main()
