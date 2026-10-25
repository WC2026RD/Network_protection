/* rpls.c -- Randomized trunk-link safeguarding (RPLS) pipeline.
 *
 * Implements Algorithm 1 (complete procedure) of the paper:
 *   step 1  targeted edge contraction (rules R1, R2, R3a, R3b)
 *   step 2  T randomized passes of Algorithm 2 (randTree) and Algorithm 3
 *           (trunkLink) on the contracted graph
 *   step 3  expansion back to the original graph (generated markings dropped)
 *   step 4  Algorithm 4 (forest + cost-ascending merge) on the original graph
 *
 * The result is checked exactly: every cut of size < L must contain a
 * safeguarded link.  Equivalently, the global minimum cut of the original
 * graph with capacity L on safeguarded links and capacity 1 on the others
 * must be at least L (all-pairs verification collapses to the global min cut).
 *
 * Build: gcc -O2 -o rpls rpls.c
 * Usage: rpls <graph-file> [-L 3] [-T 50] [--seed 1]
 * Graph file: first line "n m", then m lines "u v cost" (1-based ids).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ RNG */
static unsigned long long rng_state = 88172645463325252ULL;

static void rng_seed(unsigned long long s) {
    if (s) rng_state = s;
}
static unsigned long long rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static int rng_below(int k) { return (int)(rng_next() % (unsigned long long)k); }

static void shuffle_ints(int *a, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = rng_below(i + 1);
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

/* ---------------------------------------------------------------- graph */
typedef struct {
    int u, v;
    int cost;
    unsigned char active; /* edge currently present */
    unsigned char gen;    /* generated edge from rule R3b */
    unsigned char trunk;  /* in the trunk pool */
} Edge;

typedef struct {
    int n;
    int m, mcap;
    Edge *e;
    unsigned char *alive; /* node not absorbed by a contraction */
    int *deg;
    int *adj_start;       /* CSR over active edges */
    int *adj_edge;
    int adj_cap;
} Graph;

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}
static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}

static void graph_init(Graph *g, int n) {
    g->n = n;
    g->m = 0;
    g->mcap = 2 * n + 16;
    g->e = xmalloc((size_t)g->mcap * sizeof(Edge));
    g->alive = xmalloc((size_t)n);
    memset(g->alive, 1, (size_t)n);
    g->deg = xcalloc((size_t)n, sizeof(int));
    g->adj_start = xcalloc((size_t)n + 1, sizeof(int));
    g->adj_cap = 2 * n + 16;
    g->adj_edge = xmalloc((size_t)g->adj_cap * sizeof(int));
}

static void graph_free(Graph *g) {
    free(g->e);
    free(g->alive);
    free(g->deg);
    free(g->adj_start);
    free(g->adj_edge);
}

static int add_edge(Graph *g, int u, int v, int cost, int gen) {
    if (g->m == g->mcap) {
        g->mcap *= 2;
        g->e = realloc(g->e, (size_t)g->mcap * sizeof(Edge));
        if (!g->e) { fprintf(stderr, "out of memory\n"); exit(1); }
    }
    Edge *ed = &g->e[g->m];
    ed->u = u;
    ed->v = v;
    ed->cost = cost;
    ed->active = 1;
    ed->gen = (unsigned char)(gen ? 1 : 0);
    ed->trunk = 0;
    return g->m++;
}

/* rebuild the CSR adjacency over active edges with both endpoints alive */
static void build_adj(Graph *g) {
    int n = g->n;
    int *cnt = xcalloc((size_t)n, sizeof(int));
    for (int i = 0; i < g->m; ++i) {
        Edge *ed = &g->e[i];
        if (!ed->active) continue;
        if (!g->alive[ed->u] || !g->alive[ed->v]) { ed->active = 0; continue; }
        cnt[ed->u] += 1;
        cnt[ed->v] += 1;
    }
    g->adj_start[0] = 0;
    for (int v = 0; v < n; ++v) g->adj_start[v + 1] = g->adj_start[v] + cnt[v];
    int need = g->adj_start[n];
    if (need > g->adj_cap) {
        g->adj_cap = need * 2 + 8;
        g->adj_edge = realloc(g->adj_edge, (size_t)g->adj_cap * sizeof(int));
        if (!g->adj_edge) { fprintf(stderr, "out of memory\n"); exit(1); }
    }
    int *cur = xmalloc((size_t)n * sizeof(int));
    memcpy(cur, g->adj_start, (size_t)n * sizeof(int));
    for (int i = 0; i < g->m; ++i) {
        Edge *ed = &g->e[i];
        if (!ed->active) continue;
        g->adj_edge[cur[ed->u]++] = i;
        g->adj_edge[cur[ed->v]++] = i;
    }
    free(cur);
    for (int v = 0; v < n; ++v) g->deg[v] = g->adj_start[v + 1] - g->adj_start[v];
}

static void mark_trunk(Graph *g, int eid) {
    /* generated edges are ineligible for marking (rule R3b) */
    if (!g->e[eid].gen) g->e[eid].trunk = 1;
}

/* ------------------------------------------------------- union-find (DSU) */
static int dsu_find(int *p, int x) {
    while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
    return x;
}
static void dsu_unite(int *p, int *r, int a, int b) {
    a = dsu_find(p, a);
    b = dsu_find(p, b);
    if (a == b) return;
    if (r[a] < r[b]) { int t = a; a = b; b = t; }
    p[b] = a;
    if (r[a] == r[b]) r[a]++;
}

/* ----------------------------------------------- Stoer-Wagner min cut */
/* Exact global min cut of a non-negative weighted undirected graph.
 * w is a dense n x n symmetric matrix. */
static double global_mincut(const double *w0, int n) {
    if (n <= 1) return 0.0;
    double *a = xmalloc((size_t)n * n * sizeof(double));
    memcpy(a, w0, (size_t)n * n * sizeof(double));

    double best = 1e100;
    double *wt = xmalloc((size_t)n * sizeof(double));
    unsigned char *in = xmalloc((size_t)n);

    for (int m = n; m > 1; --m) {
        memset(wt, 0, (size_t)m * sizeof(double));
        memset(in, 0, (size_t)m);
        int last = -1, second = -1;
        for (int k = 0; k < m; ++k) {
            int sel = -1;
            for (int i = 0; i < m; ++i)
                if (!in[i] && (sel < 0 || wt[i] > wt[sel])) sel = i;
            in[sel] = 1;
            second = last;
            last = sel;
            for (int i = 0; i < m; ++i)
                if (!in[i]) wt[i] += a[(size_t)sel * n + i];
        }
        if (wt[last] < best) best = wt[last];
        /* merge last into second (fixed stride n keeps the layout valid) */
        for (int i = 0; i < n; ++i) {
            a[(size_t)second * n + i] += a[(size_t)last * n + i];
            a[(size_t)i * n + second] += a[(size_t)i * n + last];
        }
        /* remove vertex last by moving vertex m-1 into its slot */
        if (last != m - 1) {
            for (int i = 0; i < n; ++i) {
                a[(size_t)last * n + i] = a[(size_t)(m - 1) * n + i];
                a[(size_t)i * n + last] = a[(size_t)i * n + (m - 1)];
            }
            if (second == m - 1) second = last;
        }
    }
    free(a);
    free(wt);
    free(in);
    return best;
}

/* --------------------------------------------------------- preprocessing */
/* Rules R1-R3: contract every maximal component of vertices of current
 * degree at most two.  Marks the trunk pool T0, deletes the contracted
 * structure and inserts generated edges of unit cut weight. */
static void preprocess(Graph *g, int L) {
    int n = g->n;
    int *uf = xmalloc((size_t)n * sizeof(int));
    int *rk = xcalloc((size_t)n, sizeof(int));
    int *head = xmalloc((size_t)n * sizeof(int));
    int *nxt = xmalloc((size_t)n * sizeof(int));
    int *list = xmalloc((size_t)n * sizeof(int));
    int *ce_head = xmalloc((size_t)n * sizeof(int));
    int *ce_next = xmalloc((size_t)(g->m + 2 * n + 4) * sizeof(int));
    int *ce_eid = xmalloc((size_t)(g->m + 2 * n + 4) * sizeof(int));
    unsigned char *ce_kind = xmalloc((size_t)(g->m + 2 * n + 4));
    unsigned char *inW = xmalloc((size_t)n);
    int *seen = xcalloc((size_t)n, sizeof(int));
    int *ext = xmalloc((size_t)n * sizeof(int));
    int stamp = 0;

    for (;;) {
        build_adj(g);
        int alive_cnt = 0, wcnt = 0;
        for (int v = 0; v < n; ++v) {
            if (g->alive[v]) ++alive_cnt;
            inW[v] = (unsigned char)(g->alive[v] && g->deg[v] <= 2);
            if (inW[v]) ++wcnt;
        }
        if (wcnt == 0) break;

        if (wcnt == alive_cnt) {
            /* degenerate: the graph is a path or a cycle, mark everything */
            for (int i = 0; i < g->m; ++i)
                if (g->e[i].active) mark_trunk(g, i);
            break;
        }

        for (int v = 0; v < n; ++v) { uf[v] = v; rk[v] = 0; head[v] = -1; }
        for (int i = 0; i < g->m; ++i) {
            Edge *ed = &g->e[i];
            if (ed->active && inW[ed->u] && inW[ed->v]) dsu_unite(uf, rk, ed->u, ed->v);
        }
        for (int v = 0; v < n; ++v)
            if (inW[v]) { int r = dsu_find(uf, v); nxt[v] = head[r]; head[r] = v; }
        for (int v = 0; v < n; ++v) ce_head[v] = -1;
        for (int i = 0; i < g->m; ++i) {
            Edge *ed = &g->e[i];
            if (!ed->active) continue;
            int bu = inW[ed->u], bv = inW[ed->v];
            int c = -1, kind = -1;
            if (bu && bv) { c = dsu_find(uf, ed->u); kind = 0; }
            else if (bu)  { c = dsu_find(uf, ed->u); kind = 1; }
            else if (bv)  { c = dsu_find(uf, ed->v); kind = 1; }
            if (c < 0) continue;
            ce_eid[i] = i;
            ce_kind[i] = (unsigned char)kind;
            ce_next[i] = ce_head[c];
            ce_head[c] = i;
        }

        int changed = 0;
        for (int r = 0; r < n; ++r) {
            if (head[r] < 0) continue;
            int sz = 0, extc = 0;
            for (int v = head[r]; v >= 0; v = nxt[v]) list[sz++] = v;

            ++stamp;
            for (int ei = ce_head[r]; ei >= 0; ei = ce_next[ei]) {
                if (ce_kind[ei] != 1) continue;
                Edge *ed = &g->e[ce_eid[ei]];
                int out = inW[ed->u] ? ed->v : ed->u;
                if (seen[out] != stamp) { seen[out] = stamp; ext[extc++] = out; }
            }
            /* rule R2: every internal and boundary edge is trunk-marked */
            for (int ei = ce_head[r]; ei >= 0; ei = ce_next[ei])
                mark_trunk(g, ce_eid[ei]);

            if (extc <= 1) {
                /* rule R3a: absorb H into its single external neighbour */
                for (int ei = ce_head[r]; ei >= 0; ei = ce_next[ei])
                    g->e[ce_eid[ei]].active = 0;
                for (int k = 0; k < sz; ++k) g->alive[list[k]] = 0;
                ++changed;
            } else if (extc == 2) {
                /* rule R3b: strip contraction, replace H by one generated edge */
                for (int ei = ce_head[r]; ei >= 0; ei = ce_next[ei])
                    g->e[ce_eid[ei]].active = 0;
                for (int k = 0; k < sz; ++k) g->alive[list[k]] = 0;
                add_edge(g, ext[0], ext[1], 0, 1);
                ++changed;
            }
            /* extc > 2 is excluded by Lemma 3.1 */
        }
        if (!changed) break;
    }

    free(uf); free(rk); free(head); free(nxt); free(list);
    free(ce_head); free(ce_next); free(ce_eid); free(ce_kind);
    free(inW); free(seen); free(ext);
}

/* ---------------------------------------------- Algorithm 2: randTree */
static void rand_tree(Graph *g, int root, int *stat, int *cval, int *acval,
                      int *pval, int *dval, int *par, int *par_edge) {
    int n = g->n;
    for (int v = 0; v < n; ++v) {
        stat[v] = 0; cval[v] = 0; acval[v] = 0; pval[v] = 0;
        dval[v] = 0; par[v] = -1; par_edge[v] = -1;
    }

    struct Frame { int v, pos, neg; };
    struct Frame *stk = xmalloc((size_t)n * sizeof(struct Frame));
    int top = 0;

    stat[root] = 1;
    dval[root] = 1;
    cval[root] = 0;
    stk[top].v = root; stk[top].pos = 0; stk[top].neg = 0; ++top;

    while (top > 0) {
        struct Frame *f = &stk[top - 1];
        int v = f->v;
        int s = g->adj_start[v], en = g->adj_start[v + 1];
        if (s + f->pos < en) {
            int eid = g->adj_edge[s + f->pos];
            ++f->pos;
            Edge *ed = &g->e[eid];
            int u = (ed->u == v) ? ed->v : ed->u;
            if (stat[u] == 1) {           /* back edge to a CREATING node */
                acval[u] += 1; cval[v] += 1; cval[u] -= 1;
            } else if (stat[u] == 0) {    /* tree edge, recurse */
                dval[u] = dval[v] + 1;
                par[u] = v;
                par_edge[u] = eid;
                stat[u] = 1;
                cval[u] = 0;
                stk[top].v = u; stk[top].pos = 0; stk[top].neg = 0; ++top;
            }
        } else {                          /* finish v */
            acval[v] = cval[v] + f->neg;
            stat[v] = 2;
            --top;
            if (top > 0) {
                int p = stk[top - 1].v;
                pval[v] = acval[p] - (cval[v] - acval[p]);
                cval[p] += cval[v];
                if (pval[v] < 0) stk[top - 1].neg += pval[v];
            }
        }
    }
    free(stk);
}

/* ------------------------------------------- Algorithm 3: trunkLink */
static void trunk_link(Graph *g, int root, int L, const int *acval,
                       const int *pval, const int *dval, const int *par,
                       const int *par_edge, int *first_child, int *next_sib) {
    int n = g->n;
    for (int v = 0; v < n; ++v) { first_child[v] = -1; next_sib[v] = -1; }
    for (int v = 0; v < n; ++v) {
        if (!g->alive[v] || par[v] < 0) continue;
        next_sib[v] = first_child[par[v]];
        first_child[par[v]] = v;
    }

    struct Frame { int w, n, fd, sd, phase, it; };
    struct Frame *stk = xmalloc((size_t)n * sizeof(struct Frame));
    int top = 0;
    stk[top].w = -1; stk[top].n = root; stk[top].fd = 0; stk[top].sd = 0;
    stk[top].phase = 0; stk[top].it = 0; ++top;

    while (top > 0) {
        struct Frame *f = &stk[top - 1];
        int v = f->n;
        if (f->phase == 0) {
            if (acval[v] < L) { f->sd = f->fd; f->fd = dval[v]; }
            if (f->w >= 0 && pval[v] < 0 && acval[f->w] >= pval[v] + L &&
                dval[v] == f->fd + 1) {
                f->fd = f->sd; f->sd = 0;
            }
            int p = par[v];
            if (p >= 0 && pval[v] < 0 && acval[p] < L) {
                mark_trunk(g, par_edge[v]);
                f->phase = 1; f->it = 0;
            } else if (g->deg[v] < L) {
                int s = g->adj_start[v], en = g->adj_start[v + 1];
                for (int i = s; i < en; ++i) mark_trunk(g, g->adj_edge[i]);
                f->phase = 1; f->it = 0;
            } else {
                f->phase = 2;
                f->it = first_child[v];
            }
        } else if (f->phase == 2) {
            int c = f->it;
            if (c < 0) {
                f->phase = 1; f->it = 0;
            } else {
                f->it = next_sib[c];
                stk[top].w = v; stk[top].n = c;
                stk[top].fd = f->fd; stk[top].sd = f->sd;
                stk[top].phase = 0; stk[top].it = 0;
                ++top;
            }
        } else {
            int s = g->adj_start[v], en = g->adj_start[v + 1];
            while (s + f->it < en) {
                int eid = g->adj_edge[s + f->it];
                ++f->it;
                Edge *ed = &g->e[eid];
                int u = (ed->u == v) ? ed->v : ed->u;
                if (u == par[v] || (par[u] == v)) continue;
                if (dval[u] < f->fd) mark_trunk(g, eid);
            }
            --top;
        }
    }
    free(stk);
}

/* ---------------------------------------- safeguard stage (Algorithm 4) */
static Edge **sort_base;

static int cmp_edge_cost(const void *pa, const void *pb) {
    const Edge *a = *(const Edge *const *)pa;
    const Edge *b = *(const Edge *const *)pb;
    if (a->cost != b->cost) return (a->cost < b->cost) ? -1 : 1;
    return 0;
}

/* returns number of safeguarded links, fills sel[] with edge ids */
static int safeguard(Graph *g, int m0, int *sel, long long *total_cost) {
    int n = g->n;
    int *uf = xmalloc((size_t)n * sizeof(int));
    int *rk = xcalloc((size_t)n, sizeof(int));
    for (int v = 0; v < n; ++v) { uf[v] = v; rk[v] = 0; }

    for (int i = 0; i < m0; ++i) {
        Edge *ed = &g->e[i];
        if (!ed->active || ed->trunk) continue;
        dsu_unite(uf, rk, ed->u, ed->v);
    }

    Edge **pool = xmalloc((size_t)m0 * sizeof(Edge *));
    int np = 0;
    for (int i = 0; i < m0; ++i)
        if (g->e[i].active && g->e[i].trunk) pool[np++] = &g->e[i];

    sort_base = pool;
    qsort(pool, (size_t)np, sizeof(Edge *), cmp_edge_cost);

    int nsel = 0;
    long long cost = 0;
    for (int k = 0; k < np; ++k) {
        Edge *ed = pool[k];
        if (dsu_find(uf, ed->u) == dsu_find(uf, ed->v)) continue;
        dsu_unite(uf, rk, ed->u, ed->v);
        ed->trunk = 2; /* selected marker */
        sel[nsel++] = (int)(ed - g->e);
        cost += ed->cost;
    }

    free(pool);
    free(uf);
    free(rk);
    *total_cost = cost;
    return nsel;
}

/* ------------------------------------------------------------ main */
static int read_graph(const char *path, Graph *g) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int n, m;
    if (fscanf(f, "%d %d", &n, &m) != 2) { fclose(f); return 0; }
    if (n <= 0 || m < 0) { fclose(f); return 0; }
    graph_init(g, n);
    for (int i = 0; i < m; ++i) {
        int u, v, c;
        if (fscanf(f, "%d %d %d", &u, &v, &c) != 3) { fclose(f); return 0; }
        --u; --v;
        if (u == v) continue;
        add_edge(g, u, v, c, 0);
    }
    fclose(f);
    return 1;
}

static double unit_mincut(Graph *g, int *m0_out) {
    int n = g->n;
    double *w = xcalloc((size_t)n * n, sizeof(double));
    int cnt = 0;
    for (int i = 0; i < g->m; ++i) {
        Edge *ed = &g->e[i];
        if (!ed->active || ed->gen) continue;
        w[(size_t)ed->u * n + ed->v] += 1.0;
        w[(size_t)ed->v * n + ed->u] += 1.0;
        ++cnt;
    }
    if (m0_out) *m0_out = cnt;
    double v = global_mincut(w, n);
    free(w);
    return v;
}

static int verify(Graph *g, int L, const int *sel, int nsel) {
    int n = g->n;
    double *w = xcalloc((size_t)n * n, sizeof(double));
    unsigned char *prot = xcalloc((size_t)g->m, 1);
    for (int k = 0; k < nsel; ++k) prot[sel[k]] = 1;
    int edges = 0;
    for (int i = 0; i < g->m; ++i) {
        Edge *ed = &g->e[i];
        if (!ed->active || ed->gen) continue;
        double cap = prot[i] ? (double)L : 1.0;
        w[(size_t)ed->u * n + ed->v] += cap;
        w[(size_t)ed->v * n + ed->u] += cap;
        ++edges;
    }
    double lambda = global_mincut(w, n);
    free(w);
    free(prot);
    (void)edges;
    return lambda >= (double)L - 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <graph-file> [-L 3] [-T 50] [--seed s]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int L = 3, T = 50;
    unsigned long long seed = 12345;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "-L") && i + 1 < argc) L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-T") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
    }
    rng_seed(seed);

    Graph g;
    if (!read_graph(path, &g)) { fprintf(stderr, "bad graph file\n"); return 1; }

    int m0 = g.m;
    int n = g.n;
    clock_t t0 = clock();
    double l0 = unit_mincut(&g, &m0);
    clock_t t1 = clock();
    printf("instance: %s\n", path);
    printf("nodes=%d  edges=%d  avg_degree=%.3f  L=%d  L0=%.0f  passes=%d\n",
           n, m0, n > 1 ? 2.0 * m0 / n : 0.0, L, l0, T);

    preprocess(&g, L);
    clock_t t2 = clock();
    build_adj(&g);
    int cn = 0, cm = 0;
    for (int v = 0; v < n; ++v) if (g.alive[v]) ++cn;
    for (int i = 0; i < g.m; ++i) if (g.e[i].active) ++cm;
    printf("after contraction: nodes=%d edges=%d\n", cn, cm);
    if (l0 >= L) {
        printf("L0 >= L: no reinforcement needed (cost 0)\n");
        printf("elapsed: preprocess=%.3fs\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
        graph_free(&g);
        return 0;
    }

    int *stat = xcalloc((size_t)n, sizeof(int));
    int *cval = xcalloc((size_t)n, sizeof(int));
    int *acval = xcalloc((size_t)n, sizeof(int));
    int *pval = xcalloc((size_t)n, sizeof(int));
    int *dval = xcalloc((size_t)n, sizeof(int));
    int *par = xcalloc((size_t)n, sizeof(int));
    int *par_edge = xcalloc((size_t)n, sizeof(int));
    int *first_child = xcalloc((size_t)n, sizeof(int));
    int *next_sib = xcalloc((size_t)n, sizeof(int));
    int *alive_nodes = xmalloc((size_t)n * sizeof(int));

    for (int t = 0; t < T; ++t) {
        build_adj(&g);
        int na = 0;
        for (int v = 0; v < n; ++v) if (g.alive[v]) alive_nodes[na++] = v;
        if (na == 0) break;
        for (int k = 0; k < na; ++k) {
            int v = alive_nodes[k];
            shuffle_ints(g.adj_edge + g.adj_start[v], g.adj_start[v + 1] - g.adj_start[v]);
        }
        int root = alive_nodes[rng_below(na)];
        rand_tree(&g, root, stat, cval, acval, pval, dval, par, par_edge);
        build_adj(&g);
        trunk_link(&g, root, L, acval, pval, dval, par, par_edge,
                   first_child, next_sib);
    }
    clock_t t3 = clock();

    /* step 3: expansion -- restore all original edges, drop generated ones */
    for (int i = 0; i < g.m; ++i) g.e[i].active = (i < m0) ? 1 : 0;
    memset(g.alive, 1, (size_t)n);
    build_adj(&g);

    int *sel = xmalloc((size_t)m0 * sizeof(int));
    long long cost = 0;
    int nsel = safeguard(&g, m0, sel, &cost);
    clock_t t4 = clock();

    int ok = verify(&g, L, sel, nsel);

    printf("safeguarded links=%d  total_cost=%lld\n", nsel, cost);
    printf("verification (lambda_S >= L): %s\n", ok ? "PASS" : "FAIL");
    printf("first safeguarded links (u v cost):");
    for (int k = 0; k < nsel && k < 20; ++k) {
        Edge *ed = &g.e[sel[k]];
        printf(" (%d,%d,%d)", ed->u + 1, ed->v + 1, ed->cost);
    }
    printf("\n");
    printf("elapsed: mincut=%.3fs preprocess=%.3fs passes=%.3fs safeguard=%.3fs\n",
           (double)(t1 - t0) / CLOCKS_PER_SEC, (double)(t2 - t1) / CLOCKS_PER_SEC,
           (double)(t3 - t2) / CLOCKS_PER_SEC, (double)(t4 - t3) / CLOCKS_PER_SEC);

    free(stat); free(cval); free(acval); free(pval); free(dval);
    free(par); free(par_edge); free(first_child); free(next_sib);
    free(alive_nodes); free(sel);
    graph_free(&g);
    return ok ? 0 : 2;
}
