#include <bits/stdc++.h>
#include <chrono>
#include <umfpack.h>
using namespace std;
using namespace chrono;
#define mp make_pair

int sed = 114514;
bool is_c432 = false;
long long total_duration_ms = 0;

struct Edge {
  int to, rev;
};

int n;
vector<vector<int>> adj;
map<pair<int, int>, bool> edge_exists;

// Tarjan variables
vector<int> disc, low, parent;
set<int> articulation_points;
int time_counter;

// For marking chain components
vector<bool> visited, is_chain, marked;

void tarjanDFS(int u) {
  disc[u] = low[u] = ++time_counter;
  int children = 0;
  for (int v : adj[u]) {
    if (v == 1)
      continue;
    if (disc[v] == -1) {
      children++;
      parent[v] = u;
      tarjanDFS(v);
      low[u] = min(low[u], low[v]);
      if ((parent[u] == -1 && children > 1) ||
          (parent[u] != -1 && low[v] >= disc[u])) {
        articulation_points.insert(u);
      }
    } else if (v != parent[u]) {
      low[u] = min(low[u], disc[v]);
    }
  }
}

int is_chain_component(int u, int pu = -1) {
  visited[u] = true;
  int deg = 0, sz = -2;
  for (int v : adj[u]) {
    if (v == 1)
      continue;
    if (v == pu)
      continue;
    deg++;
    if (deg > 1)
      return -1;
    if (visited[v])
      return -1;
    sz = is_chain_component(v, u);
    if (sz == -1)
      return -1;
  }
  if (deg == 0)
    return 1;
  return sz + 1;
}

void mark_chain_nodes(int u, int pu = -1) {
  marked[u] = true;
  for (int v : adj[u]) {
    if (v == 1)
      continue;
    if (v == pu)
      continue;
    mark_chain_nodes(v, u);
  }
}

//------------------------------------------------------------------------------
// KLU/Demo/kludemo.c:  demo for KLU (int32_t version)
//------------------------------------------------------------------------------

// KLU, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
// Authors: Timothy A. Davis and Ekanathan Palamadai.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

/* Read in a Matrix Market matrix (using CHOLMOD) and solve a linear system. */

#include "klu.h"
#include "klu_cholmod.h"

/* for handling complex matrices */
#define REAL(X, i) (X[2 * (i)])
#define IMAG(X, i) (X[2 * (i) + 1])
#define CABS(X, i) (sqrt(REAL(X, i) * REAL(X, i) + IMAG(X, i) * IMAG(X, i)))

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/* ========================================================================== */
/* === klu_backslash ======================================================== */
/* ========================================================================== */

static int klu_backslash /* return 1 if successful, 0 otherwise */
    (
        /* --- input ---- */
        int n,      /* A is n-by-n */
        int *Ap,    /* size n+1, column pointers */
        int *Ai,    /* size nz = Ap [n], row indices */
        double *Ax, /* size nz, numerical values */
        int isreal, /* nonzero if A is real, 0 otherwise */
        double *B,  /* size n, right-hand-side */

        /* --- output ---- */
        double *X, /* size n, solution to Ax=b */
        double *R, /* size n, residual r = b-A*x */

        /* --- scalar output --- */
        int *lunz,     /* nnz (L+U+F) */
        double *rnorm, /* norm (b-A*x,1) / norm (A,1) */

        /* --- workspace - */

        klu_common *Common /* default parameters and statistics */
    ) {
  double anorm = 0, asum;
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  int i, j, p;

  if (!Ap || !Ai || !Ax || !B || !X || !B) {
    cout << "failed 1\n";
    return (0);
  }

  /* ---------------------------------------------------------------------- */
  /* symbolic ordering and analysis */
  /* ---------------------------------------------------------------------- */

  Symbolic = klu_analyze(n, Ap, Ai, Common);
  if (!Symbolic) {
    cout << "failed 2\n";
    return (0);
  }

  if (isreal) {

    /* ------------------------------------------------------------------ */
    /* factorization */
    /* ------------------------------------------------------------------ */

    Numeric = klu_factor(Ap, Ai, Ax, Symbolic, Common);
    if (!Numeric) {
      klu_free_symbolic(&Symbolic, Common);
      cout << "failed 3\n";
      return (0);
    }

    /* ------------------------------------------------------------------ */
    /* statistics (not required to solve Ax=b) */
    /* ------------------------------------------------------------------ */

    klu_rgrowth(Ap, Ai, Ax, Symbolic, Numeric, Common);
    klu_condest(Ap, Ax, Symbolic, Numeric, Common);
    klu_rcond(Symbolic, Numeric, Common);
    klu_flops(Symbolic, Numeric, Common);
    *lunz = Numeric->lnz + Numeric->unz - n +
            ((Numeric->Offp) ? (Numeric->Offp[n]) : 0);

    /* ------------------------------------------------------------------ */
    /* solve Ax=b */
    /* ------------------------------------------------------------------ */

    for (i = 0; i < n; i++) {
      X[i] = B[i];
    }
    klu_solve(Symbolic, Numeric, n, 1, X, Common);

    /* ------------------------------------------------------------------ */
    /* compute residual, rnorm = norm(b-Ax,1) / norm(A,1) */
    /* ------------------------------------------------------------------ */

    for (i = 0; i < n; i++) {
      R[i] = B[i];
    }
    for (j = 0; j < n; j++) {
      asum = 0;
      for (p = Ap[j]; p < Ap[j + 1]; p++) {
        /* R (i) -= A (i,j) * X (j) */
        R[Ai[p]] -= Ax[p] * X[j];
        asum += fabs(Ax[p]);
      }
      anorm = MAX(anorm, asum);
    }
    *rnorm = 0;
    for (i = 0; i < n; i++) {
      *rnorm = MAX(*rnorm, fabs(R[i]));
    }

    /* ------------------------------------------------------------------ */
    /* free numeric factorization */
    /* ------------------------------------------------------------------ */

    klu_free_numeric(&Numeric, Common);

  } else {

    /* ------------------------------------------------------------------ */
    /* statistics (not required to solve Ax=b) */
    /* ------------------------------------------------------------------ */

    Numeric = klu_z_factor(Ap, Ai, Ax, Symbolic, Common);
    if (!Numeric) {
      cout << "Common status = " << Common->status << ' ' << KLU_OUT_OF_MEMORY
           << " " << KLU_TOO_LARGE << " " << KLU_SINGULAR << ' ' << KLU_OK
           << '\n';
      klu_free_symbolic(&Symbolic, Common);
      cout << "failed 4\n";
      return (0);
    }

    /* ------------------------------------------------------------------ */
    /* statistics */
    /* ------------------------------------------------------------------ */

    klu_z_rgrowth(Ap, Ai, Ax, Symbolic, Numeric, Common);
    klu_z_condest(Ap, Ax, Symbolic, Numeric, Common);
    klu_z_rcond(Symbolic, Numeric, Common);
    klu_z_flops(Symbolic, Numeric, Common);
    *lunz = Numeric->lnz + Numeric->unz - n +
            ((Numeric->Offp) ? (Numeric->Offp[n]) : 0);

    /* ------------------------------------------------------------------ */
    /* solve Ax=b */
    /* ------------------------------------------------------------------ */

    for (i = 0; i < 2 * n; i++) {
      X[i] = B[i];
    }
    klu_z_solve(Symbolic, Numeric, n, 1, X, Common);

    /* ------------------------------------------------------------------ */
    /* compute residual, rnorm = norm(b-Ax,1) / norm(A,1) */
    /* ------------------------------------------------------------------ */

    for (i = 0; i < 2 * n; i++) {
      R[i] = B[i];
    }
    for (j = 0; j < n; j++) {
      asum = 0;
      for (p = Ap[j]; p < Ap[j + 1]; p++) {
        /* R (i) -= A (i,j) * X (j) */
        i = Ai[p];
        REAL(R, i) -= REAL(Ax, p) * REAL(X, j) - IMAG(Ax, p) * IMAG(X, j);
        IMAG(R, i) -= IMAG(Ax, p) * REAL(X, j) + REAL(Ax, p) * IMAG(X, j);
        asum += CABS(Ax, p);
      }
      anorm = MAX(anorm, asum);
    }
    *rnorm = 0;
    for (i = 0; i < n; i++) {
      *rnorm = MAX(*rnorm, CABS(R, i));
    }

    /* ------------------------------------------------------------------ */
    /* free numeric factorization */
    /* ------------------------------------------------------------------ */

    klu_z_free_numeric(&Numeric, Common);
  }

  /* ---------------------------------------------------------------------- */
  /* free symbolic analysis, and residual */
  /* ---------------------------------------------------------------------- */

  klu_free_symbolic(&Symbolic, Common);
  return (1);
}
//------------------------------------------------------------------------------
// SuiteSparse/KLU/User/klu_cholmod.c: KLU int32_t interface to CHOLMOD
//------------------------------------------------------------------------------

// KLU, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
// Authors: Timothy A. Davis and Ekanathan Palamadai.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

/* klu_cholmod: user-defined ordering function to interface KLU to CHOLMOD.
 *
 * This routine is an example of a user-provided ordering function for KLU.
 * Its return value is klu_cholmod's estimate of max (nnz(L),nnz(U)):
 *      0 if error,
 *      -1 if OK, but estimate of max (nnz(L),nnz(U)) not computed
 *      > 0 if OK and estimate computed.
 *
 * This function can be assigned to KLU's Common->user_order function pointer.
 */

#include "cholmod.h"

#if (CHOLMOD__VERSION < SUITESPARSE__VERCODE(5, 3, 0))
#error                                                                         \
    "KLU:CHOLMOD @KLU_VERSION_MAJOR@.@KLU_VERSION_MINOR@.@KLU_VERSION_SUB@ requires CHOLMOD 5.3.0 or later"
#endif

#define TRUE 1
#define FALSE 0

int32_t klu_cholmod(
    /* inputs */
    int32_t n,    /* A is n-by-n */
    int32_t Ap[], /* column pointers */
    int32_t Ai[], /* row indices */
    /* outputs */
    int32_t Perm[], /* fill-reducing permutation */
    /* user-defined */
    klu_common *Common /* user-defined data is in Common->user_data */
) {
  double one[2] = {1, 0}, zero[2] = {0, 0}, lnz = 0;
  cholmod_sparse Amatrix, *A, *AT, *S;
  cholmod_factor *L;
  cholmod_common cm;
  int32_t *P;
  int32_t k;
  int symmetric;
  klu_common km;
  klu_defaults(&km);

  if (Ap == NULL || Ai == NULL || Perm == NULL || n < 0) {
    /* invalid inputs */
    return (0);
  }

  /* start CHOLMOD */
  cholmod_start(&cm);
  cm.supernodal = CHOLMOD_SIMPLICIAL;
  cm.print = 0;

  /* construct a CHOLMOD version of the input matrix A */
  A = &Amatrix;
  A->nrow = n; /* A is n-by-n */
  A->ncol = n;
  A->nzmax = Ap[n]; /* with nzmax entries */
  A->packed = TRUE; /* there is no A->nz array */
  A->stype = 0;     /* A is unsymmetric */
  A->itype = CHOLMOD_INT;
  A->xtype = CHOLMOD_PATTERN;
  A->dtype = CHOLMOD_DOUBLE;
  A->nz = NULL;
  A->p = Ap;   /* column pointers */
  A->i = Ai;   /* row indices */
  A->x = NULL; /* no numerical values */
  A->z = NULL;
  A->sorted = FALSE; /* columns of A are not sorted */

  /* get the user_data; default is symmetric if user_data is NULL */
  symmetric = true;
  cm.nmethods = 1;
  cm.method[0].ordering = CHOLMOD_AMD;
  int64_t *user_data = (int64_t *)Common->user_data;
  if (user_data != NULL) {
    symmetric = (user_data[0] != 0);
    cm.method[0].ordering = user_data[1];
  }

  /* AT = pattern of A' */
  AT = cholmod_transpose(A, 0, &cm);
  if (symmetric) {
    /* S = the symmetric pattern of A+A' */
    S = cholmod_add(A, AT, one, zero, FALSE, FALSE, &cm);
    cholmod_free_sparse(&AT, &cm);
    if (S != NULL) {
      S->stype = 1;
    }
  } else {
    /* S = A'.  CHOLMOD will order S*S', which is A'*A */
    S = AT;
  }

  /* order and analyze S or S*S' */
  L = cholmod_analyze(S, &cm);

  /* copy the permutation from L to the output */
  if (L != NULL) {
    P = (int32_t *)L->Perm;
    for (k = 0; k < n; k++) {
      Perm[k] = P[k];
    }
    lnz = cm.lnz;
  }

  cholmod_free_sparse(&S, &cm);
  cholmod_free_factor(&L, &cm);
  cholmod_finish(&cm);
  return (lnz);
}

/* ========================================================================== */
/* === klu_demo ============================================================= */
/* ========================================================================== */

/* Given a sparse matrix A, set up a right-hand-side and solve X = A\b */

static void klu_demo(int n, int *Ap, int *Ai, double *Ax, int isreal,
                     int cholmod_ordering) {
  double rnorm;
  klu_common Common;
  double *B, *X, *R;
  int i, lunz;

  /* ---------------------------------------------------------------------- */
  /* set defaults */
  /* ---------------------------------------------------------------------- */

  klu_defaults(&Common);
  int64_t user_data[2];
  if (cholmod_ordering >= 0) {
    Common.ordering = 3;
    Common.user_order = klu_cholmod;
    user_data[0] = 1; // symmetric
    user_data[1] = cholmod_ordering;
    Common.user_data = user_data;
  }

  /* ---------------------------------------------------------------------- */
  /* create a right-hand-side */
  /* ---------------------------------------------------------------------- */

  if (isreal) {
    /* B = 1 + (1:n)/n */
    B = (double *)klu_malloc(n, sizeof(double), &Common);
    X = (double *)klu_malloc(n, sizeof(double), &Common);
    R = (double *)klu_malloc(n, sizeof(double), &Common);
    if (B) {
      for (i = 0; i < n; i++) {
        B[i] = 1 + ((double)i + 1) / ((double)n);
      }
    }
  } else {
    /* real (B) = 1 + (1:n)/n, imag(B) = (n:-1:1)/n */
    B = (double *)klu_malloc(n, 2 * sizeof(double), &Common);
    X = (double *)klu_malloc(n, 2 * sizeof(double), &Common);
    R = (double *)klu_malloc(n, 2 * sizeof(double), &Common);
    if (B) {
      for (i = 0; i < n; i++) {
        REAL(B, i) = 1 + ((double)i + 1) / ((double)n);
        IMAG(B, i) = ((double)n - i) / ((double)n);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  /* X = A\b using KLU and print statistics */
  /* ---------------------------------------------------------------------- */

  auto start = std::chrono::high_resolution_clock::now();

  int st =
      klu_backslash(n, Ap, Ai, Ax, isreal, B, X, R, &lunz, &rnorm, &Common);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  total_duration_ms += duration;

  if (!st) {
    printf("KLU failed\n");
  } else {
    /*
  printf("n %d nnz(A) %d nnz(L+U+F) %d resid %g\n"
         "recip growth %g condest %g rcond %g flops %g\n",
         n, Ap[n], lunz, rnorm, Common.rgrowth, Common.condest, Common.rcond,
         Common.flops);
         */
  }

  /* ---------------------------------------------------------------------- */
  /* free the problem */
  /* ---------------------------------------------------------------------- */

  if (isreal) {
    klu_free(B, n, sizeof(double), &Common);
    klu_free(X, n, sizeof(double), &Common);
    klu_free(R, n, sizeof(double), &Common);
  } else {
    klu_free(B, 2 * n, sizeof(double), &Common);
    klu_free(X, 2 * n, sizeof(double), &Common);
    klu_free(R, 2 * n, sizeof(double), &Common);
  }
  // printf("peak memory usage: %g bytes\n", (double)(Common.mempeak));
}

#include "cholmod.h"

cholmod_sparse *
generate_random_complex_sparse_matrix(cholmod_common *c, size_t nrow,
                                      size_t ncol, size_t avg_nnz_per_col,
                                      const vector<vector<int>> &graph) {
  size_t nzmax = ncol * avg_nnz_per_col;

  cholmod_sparse *A = (cholmod_sparse *)calloc(1, sizeof(cholmod_sparse));
  A->nrow = nrow;
  A->ncol = ncol;
  A->nzmax = nzmax;

  // Allocate memory
  A->p = malloc((ncol + 1) * sizeof(int));
  A->i = malloc(nzmax * sizeof(int));
  A->x = malloc(2 * nzmax * sizeof(double)); // real + imag
  A->z = NULL;
  A->nz = NULL;

  // Matrix format setup
  A->stype = 0;               // unsymmetric
  A->itype = CHOLMOD_INT;     // use int32_t for p, i
  A->xtype = CHOLMOD_COMPLEX; // complex matrix
  A->dtype = CHOLMOD_DOUBLE;  // double precision
  A->sorted = 1;
  A->packed = 1;

  int *Ap = (int *)(A->p);
  int *Ai = (int *)(A->i);
  double *Ax = (double *)(A->x); // [real0, imag0, real1, imag1, ...]

  srand(sed);
  // srand(time(0));
  size_t nnz = 0;

  vector<pair<double, double>> diag;
  vector<int> diag_nnz;
  map<pair<int, int>, pair<int, int>> edge;
  diag.resize(nrow);
  diag_nnz.resize(ncol);

  for (int col = 0; col < ncol; ++col) {
    Ap[col] = nnz;
    int u = col + 1, mid = -1;
    double s_rel = 0, s_img = 0;
    for (int i = 0; i < graph[u].size(); i++) {
      int v = graph[u][i];
      if (v > u)
        break;
      int row = v - 1;
      Ai[nnz] = row;
      double a = (double)(rand() % 4 + 1) * 0.5;
      double b = (double)(rand() % 4 + 1) * 0.5;
      double rel = a / (a * a + b * b);
      double img = -b / (a * a + b * b);

      /*
      int mi = min(col, row), mx = max(col, row);
      if (edge.count(mp(mi, mx)) == 0) {
        edge[mp(mi, mx)] = mp(rel, img);
      }
      rel = edge[mp(mi, mx)].first;
      img = edge[mp(mi, mx)].second;
      */

      diag[row].first += rel;
      diag[row].second += img;

      Ax[2 * nnz] = -rel;
      Ax[2 * nnz + 1] = -img;

      nnz++;
    }
    Ai[mid = nnz] = col;
    diag_nnz[col] = mid;

    nnz++;
    for (int i = 0; i < graph[u].size(); i++) {
      int v = graph[u][i];
      if (v < u)
        continue;
      int row = v - 1;
      Ai[nnz] = row;
      double a = (double)(rand() % 4 + 1) * 0.5;
      double b = (double)(rand() % 4 + 1) * 0.5;
      double rel = a / (a * a + b * b);
      double img = -b / (a * a + b * b);

      /*
      int mi = min(col, row), mx = max(col, row);
      if (edge.count(mp(mi, mx)) == 0) {
        edge[mp(mi, mx)] = mp(rel, img);
      }
      rel = edge[mp(mi, mx)].first;
      img = edge[mp(mi, mx)].second;
      */

      diag[row].first += rel;
      diag[row].second += img;

      Ax[2 * nnz] = -rel;
      Ax[2 * nnz + 1] = -img;

      nnz++;
    }
  }
  for (size_t col = 0; col < ncol; ++col) {
    int mid = diag_nnz[col];
    Ax[2 * mid] = diag[col].first + 1;
    Ax[2 * mid + 1] = diag[col].second;
  }
  Ap[ncol] = nnz;
  A->nzmax = nnz;

  return A;
}
void randomize_and_solve(const vector<vector<int>> &graph) {
  int n = graph.size() - 1, nnz = n;
  for (int i = 1; i <= n; i++) {
    nnz += graph[i].size();
  }
  //--------------------------------------------------------------------------
  // read in a matrix and solve Ax=b
  //--------------------------------------------------------------------------

  cholmod_sparse *A;
  cholmod_common ch;
  cholmod_start(&ch);
  A = generate_random_complex_sparse_matrix(&ch, n, n, (nnz + n - 1) / n,
                                            graph);
  if (A) {
    if (A->nrow != A->ncol || A->stype != 0 ||
        (!(A->xtype == CHOLMOD_REAL || A->xtype == CHOLMOD_COMPLEX))) {
      printf("invalid matrix\n");
    } else {
      klu_demo(A->nrow, (int *)A->p, (int *)A->i, (double *)A->x,
               A->xtype == CHOLMOD_REAL, -1);
    }
    cholmod_free_sparse(&A, &ch);
  }
  cholmod_finish(&ch);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <file_name>" << endl;
    return 1;
  }
  string file_name = argv[1];
  string cir_name = "";
  for (int i = 0; i < file_name.length(); i++) {
    if (file_name[i] == '_')
      break;
    cir_name += file_name[i];
  }
  /*
  if (file_name.substr(0, 4) == "c432") {
    sed = 514;
    is_c432 = 1;
  }
  */
  /*
  cout << argv[1] << " " << sed << '\n';
  */

  ifstream fin(argv[1]);
  string line;
  getline(fin, line); // skip first line
  getline(fin, line);
  stringstream ss(line);
  ss >> n;
  adj.resize(n + 1);

  while (getline(fin, line)) {
    int a, b;
    stringstream ls(line);
    ls >> a >> b;
    if (a == 0 && b == 0)
      break;
    if (a != b && !edge_exists[{min(a, b), max(a, b)}]) {
      adj[a].push_back(b);
      adj[b].push_back(a);
      edge_exists[{min(a, b), max(a, b)}] = true;
    }
  }
  /*
  cout << n << ' ' << edge_exists.size() << '\n';
  */
  for (int i = 1; i <= n; i++) {
    sort(adj[i].begin(), adj[i].end());
  }

  // Find top two highest degrees
  vector<pair<int, int>> degree;
  for (int i = 1; i <= n; ++i) {
    degree.push_back(make_pair(adj[i].size(), i));
  }
  sort(degree.rbegin(), degree.rend());
  /*
  cout << "Top two degrees: " << degree[0].first << ' ' << degree[0].second
       << ", " << degree[1].first << endl;
       */

  // Tarjan
  disc.assign(n + 1, -1);
  low.assign(n + 1, -1);
  parent.assign(n + 1, -1);
  time_counter = 0;
  for (int i = 2; i <= n; ++i) {
    if (disc[i] == -1)
      tarjanDFS(i);
  }

  vector<int> chains;
  marked.assign(n + 1, false);
  // Check chain components
  for (int u : articulation_points) {
    visited.assign(n + 1, false);
    for (int v : adj[u]) {
      int sz = is_chain_component(v, u);
      if (sz > 1) {
        mark_chain_nodes(v, u);
        chains.push_back(sz);
      }
    }
  }
  int mx = 0;
  for (auto p : chains) {
    mx = max(mx, p);
  }

  int marked_count = count(marked.begin(), marked.end(), true);
  cout << "Marked nodes count: " << marked_count << ' ' << chains.size() << ' '
       << mx << ' ' << 1.0 * marked_count / n << endl;

  return 0;

  vector<int> id_map(n + 1);
  int idx = 0;
  for (int i = 1; i <= n; i++) {
    if (marked[i])
      continue;
    id_map[i] = ++idx;
  }

  // Remove marked nodes
  vector<vector<int>> reduced_adj(idx + 1);
  for (int i = 1; i <= n; ++i) {
    if (marked[i])
      continue;
    for (int j : adj[i]) {
      if (!marked[j])
        reduced_adj[id_map[i]].push_back(id_map[j]);
    }
  }

  //--------------------------------------------------------------------------
  // klu version
  //--------------------------------------------------------------------------

  int version[3];
  klu_version(version);
  // printf("KLU v%d.%d.%d\n", version[0], version[1], version[2]);
  if ((version[0] != KLU_MAIN_VERSION) || (version[1] != KLU_SUB_VERSION) ||
      (version[2] != KLU_SUBSUB_VERSION)) {
    // fprintf(stderr, "version in header does not match library\n");
    abort();
  }

  int nrun = 1000;

  // Solve Ax = y for original and reduced graph
  total_duration_ms = 0;
  for (int i = 0; i < nrun; i++) {
    sed = i;
    randomize_and_solve(adj);
  }
  auto t1 = total_duration_ms;
  total_duration_ms = 0;
  for (int i = 0; i < nrun; i++) {
    sed = i;
    randomize_and_solve(reduced_adj);
  }
  auto t2 = total_duration_ms;
  cout << cir_name << ' ' << t1 << ' ' << t2 << '\n';
  return 0;
}

