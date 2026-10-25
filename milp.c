/* milp.c -- the MILP of Eq. (eq:milp) implemented in C with the HiGHS API.
 *
 * Model (bidirected representation of the undirected instance):
 *   min  sum_e c_e z_e                       ( = 1/2 sum_{a in A} c_a z_a )
 *   s.t. g^{(s,t)}_a - (L - L0) z_{e(a)} <= 1     for all pairs, all arcs (b)
 *        flow conservation of L units for every ordered pair (s,t)        (c)
 *        z_e in {0,1},  g >= 0
 *
 * z_e = z_{e'} is automatic because a single binary is used per undirected
 * link; L0 is the exact global min cut of the instance.
 *
 * Build (Windows, MinGW):
 *   gcc -O2 -I _highs_win/include/highs -o milp.exe milp.c _highs_win/lib/highs.lib
 *   (highs.dll must be next to milp.exe at run time)
 *
 * The sparse matrix is passed column-wise (CSC).  HiGHS is pinned to one thread:
 * the prebuilt highs.dll crashes in the parallel LP solver when driven from a
 * MinGW-built executable.  Set MILP_DUMP=<file>.lp to dump the model.
 *
 * Usage: milp <graph-file> [-L 3] [--tl 300]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "util/HighsInt.h"
#include "interfaces/highs_c_api.h"

#define BIG_INF 1e30

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

typedef struct {
    int u, v, cost;
} Edge;

/* ------------------------------------------------ exact global min cut */
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
        for (int i = 0; i < n; ++i) {
            a[(size_t)second * n + i] += a[(size_t)last * n + i];
            a[(size_t)i * n + second] += a[(size_t)i * n + last];
        }
        if (last != m - 1) {
            for (int i = 0; i < n; ++i) {
                a[(size_t)last * n + i] = a[(size_t)(m - 1) * n + i];
                a[(size_t)i * n + last] = a[(size_t)i * n + (m - 1)];
            }
            if (second == m - 1) second = last;
        }
    }
    free(a); free(wt); free(in);
    return best;
}

static int read_graph(const char *path, int *n_out, Edge **edges_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int n, m;
    if (fscanf(f, "%d %d", &n, &m) != 2) { fclose(f); return 0; }
    if (n <= 0 || m < 0) { fclose(f); return 0; }
    Edge *e = xmalloc((size_t)(m ? m : 1) * sizeof(Edge));
    int k = 0;
    for (int i = 0; i < m; ++i) {
        int u, v, c;
        if (fscanf(f, "%d %d %d", &u, &v, &c) != 3) { fclose(f); free(e); return 0; }
        --u; --v;
        if (u == v) continue;
        e[k].u = u; e[k].v = v; e[k].cost = c; ++k;
    }
    fclose(f);
    *n_out = n;
    *edges_out = e;
    return k;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <graph-file> [-L 3] [--tl 300]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int L = 3;
    double tl = 300.0;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "-L") && i + 1 < argc) L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tl") && i + 1 < argc) tl = atof(argv[++i]);
    }

    int n = 0;
    Edge *edges = NULL;
    int m = read_graph(path, &n, &edges);
    if (m < 0 || n <= 1) { fprintf(stderr, "bad graph file\n"); return 1; }

    /* L0: exact edge connectivity (global min cut, unit capacities) */
    double *w = xcalloc((size_t)n * n, sizeof(double));
    for (int i = 0; i < m; ++i) {
        w[(size_t)edges[i].u * n + edges[i].v] += 1.0;
        w[(size_t)edges[i].v * n + edges[i].u] += 1.0;
    }
    int L0 = (int)(global_mincut(w, n) + 0.5);
    free(w);

    printf("instance: %s\n", path);
    printf("nodes=%d edges=%d avg_degree=%.3f L=%d L0=%d\n",
           n, m, 2.0 * m / n, L, L0);
    if (L0 >= L) {
        printf("L0 >= L: optimal cost 0\n");
        free(edges);
        return 0;
    }

    /* ---------------- model sizes ---------------- */
    long long npairs = (long long)n * (n - 1);
    long long arcs = 2LL * m;
    long long ncol = (long long)m + npairs * arcs;
    long long nrow = npairs * arcs + npairs * n;
    long long nnz = 8LL * npairs * m;
    if (ncol > 4000000LL || nnz > 40000000LL) {
        fprintf(stderr, "model too large (%lld cols, %lld rows, %lld nz)\n",
                ncol, nrow, nnz);
        free(edges);
        return 1;
    }
    printf("model: %lld columns, %lld rows, %lld nonzeros\n", ncol, nrow, nnz);

    double *col_cost = xmalloc((size_t)ncol * sizeof(double));
    double *col_lower = xmalloc((size_t)ncol * sizeof(double));
    double *col_upper = xmalloc((size_t)ncol * sizeof(double));
    HighsInt *integrality = xmalloc((size_t)ncol * sizeof(HighsInt));
    double *row_lower = xmalloc((size_t)nrow * sizeof(double));
    double *row_upper = xmalloc((size_t)nrow * sizeof(double));
    HighsInt *a_start = xmalloc((size_t)(ncol + 1) * sizeof(HighsInt));
    HighsInt *a_index = xmalloc((size_t)nnz * sizeof(HighsInt));
    double *a_value = xmalloc((size_t)nnz * sizeof(double));

    for (int e = 0; e < m; ++e) {
        col_cost[e] = (double)edges[e].cost;
        col_lower[e] = 0.0;
        col_upper[e] = 1.0;
        integrality[e] = kHighsVarTypeInteger;
    }
    for (long long j = m; j < ncol; ++j) {
        col_cost[j] = 0.0;
        col_lower[j] = 0.0;
        col_upper[j] = BIG_INF;
        integrality[j] = kHighsVarTypeContinuous;
    }

    double coef = (double)(L - L0);
    long long *cnt = xcalloc((size_t)ncol, sizeof(long long));
    for (int e = 0; e < m; ++e) cnt[e] = 2 * npairs;
    for (long long p = 0; p < npairs; ++p)
        for (long long a = 0; a < arcs; ++a)
            cnt[(long long)m + p * arcs + a] = 3;

    a_start[0] = 0;
    for (long long j = 0; j < ncol; ++j)
        a_start[j + 1] = (HighsInt)((long long)a_start[j] + cnt[j]);
    long long acc = (long long)a_start[ncol];
    if (acc != nnz) {
        fprintf(stderr, "internal error: nz=%lld expected %lld\n", acc, nnz);
        return 1;
    }
    long long *pos = xmalloc((size_t)ncol * sizeof(long long));
    for (long long j = 0; j < ncol; ++j) pos[j] = (long long)a_start[j];
    for (long long row = 0; row < npairs * arcs; ++row) {
        row_lower[row] = -BIG_INF;
        row_upper[row] = 1.0;
    }
    for (long long p = 0; p < npairs; ++p) {
        long long s = p / (n - 1), t = p % (n - 1);
        if (t >= s) ++t;              /* skip the diagonal s == t */
        for (int i = 0; i < n; ++i) {
            long long row = npairs * arcs + p * n + i;
            double rhs = (i == s) ? (double)L : ((i == t) ? -(double)L : 0.0);
            row_lower[row] = rhs;
            row_upper[row] = rhs;
        }
    }

    /* column-wise (CSC) construction: column e = z_e, column m+p*arcs+a = flow */
    for (int e = 0; e < m; ++e) {
        for (long long p = 0; p < npairs; ++p) {
            long long r0 = p * arcs + 2LL * e;
            a_index[pos[e]] = (HighsInt)r0;
            a_value[pos[e]] = -coef;
            ++pos[e];
            a_index[pos[e]] = (HighsInt)(r0 + 1);
            a_value[pos[e]] = -coef;
            ++pos[e];
        }
    }
    for (long long p = 0; p < npairs; ++p) {
        long long base = (long long)m + p * arcs;
        for (long long a = 0; a < arcs; ++a) {
            long long col = base + a;
            int u = edges[a / 2].u, v = edges[a / 2].v;
            if (a % 2 == 1) { int tmp = u; u = v; v = tmp; }
            a_index[pos[col]] = (HighsInt)(p * arcs + a);
            a_value[pos[col]] = 1.0;
            ++pos[col];
            a_index[pos[col]] = (HighsInt)(npairs * arcs + p * n + u);
            a_value[pos[col]] = 1.0;
            ++pos[col];
            a_index[pos[col]] = (HighsInt)(npairs * arcs + p * n + v);
            a_value[pos[col]] = -1.0;
            ++pos[col];
        }
    }
    free(cnt);
    free(pos);

    /* ---------------- solve with HiGHS ---------------- */
    void *h = Highs_create();
    HighsInt rc = Highs_passMip(h, (HighsInt)ncol, (HighsInt)nrow, (HighsInt)acc,
                                kHighsMatrixFormatColwise,
                                kHighsObjSenseMinimize, 0.0,
                                col_cost, col_lower, col_upper,
                                row_lower, row_upper,
                                a_start, a_index, a_value, integrality);
    if (rc != kHighsStatusOk) {
        fprintf(stderr, "Highs_passMip failed (rc=%d)\n", (int)rc);
        return 1;
    }
    Highs_setBoolOptionValue(h, "output_flag", 1);
    if (getenv("MILP_DUMP")) Highs_writeModel(h, getenv("MILP_DUMP"));
    Highs_setDoubleOptionValue(h, "time_limit", tl);
    Highs_setDoubleOptionValue(h, "mip_rel_gap", 0.0);
    Highs_setIntOptionValue(h, "threads", 1);

    clock_t t0 = clock();
    rc = Highs_run(h);
    clock_t t1 = clock();
    int status = (int)Highs_getModelStatus(h);
    double obj = Highs_getObjectiveValue(h);
    printf("solver status: %d  objective=%.4f  time=%.2fs\n",
           status, obj, (double)(t1 - t0) / CLOCKS_PER_SEC);

    double *x = xmalloc((size_t)ncol * sizeof(double));
    Highs_getSolution(h, x, NULL, NULL, NULL);

    int nsel = 0;
    long long cost = 0;
    for (int e = 0; e < m; ++e) {
        if (x[e] > 0.5) { ++nsel; cost += edges[e].cost; }
    }

    /* exact verification: capacity L on safeguarded links, 1 elsewhere */
    double *wv = xcalloc((size_t)n * n, sizeof(double));
    for (int e = 0; e < m; ++e) {
        double cap = (x[e] > 0.5) ? (double)L : 1.0;
        wv[(size_t)edges[e].u * n + edges[e].v] += cap;
        wv[(size_t)edges[e].v * n + edges[e].u] += cap;
    }
    double lam = global_mincut(wv, n);
    free(wv);

    printf("safeguarded links=%d  total_cost=%lld\n", nsel, cost);
    printf("verification (lambda_S >= L): %s (lambda_S=%.0f)\n",
           lam >= (double)L - 1e-9 ? "PASS" : "FAIL", lam);
    printf("first safeguarded links (u v cost):");
    int shown = 0;
    for (int e = 0; e < m && shown < 20; ++e)
        if (x[e] > 0.5) {
            printf(" (%d,%d,%d)", edges[e].u + 1, edges[e].v + 1, edges[e].cost);
            ++shown;
        }
    printf("\n");

    Highs_destroy(h);
    free(col_cost); free(col_lower); free(col_upper); free(integrality);
    free(row_lower); free(row_upper); free(a_start); free(a_index);
    free(a_value); free(x); free(edges);
    return 0;
}
