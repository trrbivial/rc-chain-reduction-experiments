#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define mp make_pair
#define ll long long

using namespace std;

string outName;
ofstream outFile;

bool is_logic_dev(string a) {
  if (a.substr(0, 3) == "and")
    return 1;
  if (a.substr(0, 3) == "not")
    return 1;
  if (a.substr(0, 4) == "nand")
    return 1;
  if (a.substr(0, 2) == "or")
    return 1;
  if (a.substr(0, 3) == "nor")
    return 1;
  if (a.substr(0, 3) == "xor")
    return 1;

  return 0;
}
string logic_dev_name(string a) {
  string ty = "null";
  if (a.substr(0, 3) == "and")
    ty = "and";
  if (a.substr(0, 3) == "not")
    ty = "not";
  if (a.substr(0, 4) == "nand")
    ty = "nand";
  if (a.substr(0, 2) == "or")
    ty = "or";
  if (a.substr(0, 3) == "nor")
    ty = "nor";
  if (a.substr(0, 3) == "xor")
    ty = "xor";
  assert(ty != "null");
  return ty;
}

namespace globalMinCut {
struct Node {
  string name;
  string type;
  int rd, cd, deg, depth;
  Node(string _name, string _type) {
    name = _name;
    type = _type;
    rd = 0;
    cd = 0;
    deg = 0;
    depth = 0;
  }
};
vector<Node> V;

struct Edge {
  string nameA, nameB;
  int idA, idB;
  int mf_edge_id;
  vector<string> label;
  vector<string> type;
  Edge() {
    nameA = "null";
    nameB = "null";
    idA = 0;
    idB = 0;
    mf_edge_id = -1;
  }
};

struct Starry {
  int to, edge_id;
  Starry(int _to = 0, int _edge_id = 0) {
    to = _to;
    edge_id = _edge_id;
  }
};

struct Subckt {
  string name;
  vector<string> ports_name;
  vector<string> inside_nodes_name;
  vector<string> content; // lines inside subckt
  int id_cnt;
};

map<pair<int, int>, Edge> E;

const int MAXN = 1000005;
vector<Starry> G[MAXN];
unordered_map<string, int> node_name2id;
unordered_map<int, string> node_id2name;
int node_id_cnt, edge_id_cnt;

unordered_map<string, Subckt> subckts;

int get_node_id(string a) {
  assert(node_name2id.count(a) > 0);
  return node_name2id[a];
}
string get_node_name(int id) {
  assert(node_id2name.count(id) > 0);
  return node_id2name[id];
}

void addNode(string a, string ty) {
  if (ty == "subckt") {
    if (a.substr(0, 3) == "net")
      ty = a.substr(0, 4);
    if (is_logic_dev(a))
      ty = logic_dev_name(a);
  }
  if (ty == "node") {
    ty = a[0];
  }
  if (node_name2id.count(a) == 0) {
    node_name2id[a] = ++node_id_cnt;
    node_id2name[node_id_cnt] = a;
    V.push_back(Node(a, ty));
  } else {
    int id = node_name2id[a];
    if (V[id].type == "gnd")
      return;
    assert(V[id].type == ty);
  }
}

void addEdge(string a, string b, string ty, string label) {
  int idA = get_node_id(a);
  int idB = get_node_id(b);
  if (idA > idB)
    swap(idA, idB);
  if (E.count(mp(idA, idB)) == 0) {
    auto &p = E[mp(idA, idB)];
    p.nameA = a;
    p.nameB = b;
    p.idA = idA;
    p.idB = idB;
  }
  auto &p = E[mp(idA, idB)];
  p.type.push_back(ty);
  p.label.push_back(label);
  V[idA].deg++;
  V[idB].deg++;
}

void buildSubckt(vector<string> tokens, string subcktName) {
  Subckt &tmpl = subckts[subcktName];
  size_t portCount = tmpl.ports_name.size();
  assert(tokens.size() == portCount + 1);

  tmpl.id_cnt++;
  string ckt = subcktName + "_" + to_string(tmpl.id_cnt);
  // addNode(subcktName + "_" + to_string(tmpl.id_cnt), "subckt");
  // param mapping: tmpl.ports[i] -> tokens[i+1]
  unordered_map<string, string> portMap;

  for (size_t i = 0; i < portCount; ++i) {
    portMap[tmpl.ports_name[i]] = tokens[i];
    addNode(tokens[i], "node");
  }
  for (auto &v : tmpl.inside_nodes_name) {
    addNode(ckt + "_" + v, "inckt");
  }

  for (const string &line : tmpl.content) {
    istringstream Iss(line);
    string w, a[15], d, g, s, b, res = "", tmp;
    Iss >> w;
    transform(w.begin(), w.end(), w.begin(), ::tolower);
    assert(w[0] == 'm' || w[0] == 'c' || w[0] == 'r');
    int iter = 2;
    if (w[0] == 'm')
      iter = 4;
    for (int i = 0; i < iter; i++) {
      Iss >> a[i];
      if (portMap.count(a[i]) == 0) {
        a[i] = ckt + "_" + a[i];
      } else {
        a[i] = portMap[a[i]];
      }
    }
    while (Iss >> tmp) {
      res += tmp + " ";
    }
    if (w[0] == 'c' || w[0] == 'r') {
      addEdge(a[0], a[1], w.substr(0, 1), ckt + "_line=" + line);
    }
    if (w[0] == 'm') {
      d = a[0];
      g = a[1];
      s = a[2];
      b = a[3];
      if (res.substr(0, 4) == "pmos")
        assert(b == "vdd");
      if (res.substr(0, 4) == "nmos")
        assert(b == "gnd");
      string d_prime = ckt + "_" + w + "_" + d + "_" + "prime";
      string s_prime = ckt + "_" + w + "_" + s + "_" + "prime";
      addNode(d_prime, "inckt");
      addNode(s_prime, "inckt");
      addEdge(d, d_prime, w.substr(0, 1) + "_d-dp", ckt + "_line=" + line);
      addEdge(s, s_prime, w.substr(0, 1) + "_s-sp", ckt + "_line=" + line);
      addEdge(g, d_prime, w.substr(0, 1) + "_g-dp", ckt + "_line=" + line);
      addEdge(g, s_prime, w.substr(0, 1) + "_g-sp", ckt + "_line=" + line);
      addEdge(d_prime, s_prime, w.substr(0, 1) + "_dp-sp",
              ckt + "_line=" + line);
    }
  }
}

const ll INF = 1 << 30;
struct MF {
  struct edge {
    int v, nxt, cap, flow;
  } e[MAXN];

  int fir[MAXN], cnt = 0;

  int n, S, T;
  ll maxflow = 0;
  int dep[MAXN], cur[MAXN];

  void init() {
    memset(fir, -1, sizeof fir);
    cnt = 0;
  }

  void addedge(int u, int v, int w) {
    e[cnt] = {v, fir[u], w, 0};
    fir[u] = cnt++;
    e[cnt] = {u, fir[v], 0, 0};
    fir[v] = cnt++;
  }
  void addedge2(int u, int v, int w) {
    e[cnt] = {v, fir[u], w, 0};
    fir[u] = cnt++;
    e[cnt] = {u, fir[v], w, 0};
    fir[v] = cnt++;
  }

  bool bfs() {
    queue<int> q;
    memset(dep, 0, sizeof(int) * (n + 1));

    dep[S] = 1;
    q.push(S);
    while (q.size()) {
      int u = q.front();
      q.pop();
      for (int i = fir[u]; ~i; i = e[i].nxt) {
        int v = e[i].v;
        if ((!dep[v]) && (e[i].cap > e[i].flow)) {
          dep[v] = dep[u] + 1;
          q.push(v);
        }
      }
    }
    return dep[T];
  }

  int dfs(int u, int flow) {
    if ((u == T) || (!flow))
      return flow;

    int ret = 0;
    for (int &i = cur[u]; ~i; i = e[i].nxt) {
      int v = e[i].v, d;
      if ((dep[v] == dep[u] + 1) &&
          (d = dfs(v, min(flow - ret, e[i].cap - e[i].flow)))) {
        ret += d;
        e[i].flow += d;
        e[i ^ 1].flow -= d;
        if (ret == flow)
          return ret;
      }
    }
    return ret;
  }

  int vis_cnt = 0;
  int vis[MAXN];

  void bfs_vis(int s) {
    queue<int> q;
    vis[s] = ++vis_cnt;
    q.push(s);
    while (q.size()) {
      int u = q.front();
      q.pop();
      for (int i = fir[u]; ~i; i = e[i].nxt) {
        int v = e[i].v;
        if (!vis[v] && (e[i].cap > e[i].flow)) {
          vis[v] = vis_cnt;
          q.push(v);
        }
      }
    }
  }

  vector<vector<pair<int, int>>> C;
  void dinic() {
    while (bfs()) {
      memcpy(cur, fir, sizeof(int) * (n + 1));
      maxflow += dfs(S, INF);
    }
    memset(vis, 0, sizeof(int) * (n + 1));
    bfs_vis(S);
    bfs_vis(T);
    for (int i = 1; i <= n; i++) {
      if (!vis[i]) {
        bfs_vis(i);
      }
    }
    C.resize(vis_cnt + 1);
    for (int u = 1; u <= n; u++) {
      for (int i = fir[u]; ~i; i = e[i].nxt) {
        int v = e[i].v;
        if (e[i].cap == e[i].flow && vis[u] != vis[v]) {
          C[vis[u]].push_back(mp(vis[v], i));
        }
      }
    }
  }
} mf;

struct SW {
  map<int, int> edge[MAXN];
  int dist[MAXN];
  bool vis[MAXN], bin[MAXN];
  int n, m;

  int contract(int &s, int &t) { // Find s,t
    memset(dist, 0, sizeof(int) * (n + 1));
    memset(vis, false, sizeof(bool) * (n + 1));
    set<pair<int, int>> D;
    for (int i = 1; i <= n; i++) {
      if (!bin[i]) {
        D.insert(mp(0, -i));
      }
    }
    int i, j, k, mincut = -1, maxc;
    for (i = 1; i <= n; i++) {
      if (D.size() == 0)
        return mincut;
      auto it = D.end();
      it--;
      k = -it->second;
      maxc = it->first;
      s = t;
      t = k;
      mincut = maxc;
      D.erase(it);
      vis[k] = true;
      for (auto it = edge[k].begin(); it != edge[k].end(); it++) {
        j = it->first;
        if (!bin[j] && !vis[j]) {
          D.erase(mp(dist[j], -j));
          dist[j] += it->second;
          D.insert(mp(dist[j], -j));
        }
      }
    }
    return mincut;
  }

  int Stoer_Wagner() {
    int inf = 0x3f3f3f3f;
    int mincut, i, j, s, t, ans;
    for (mincut = inf, i = 1; i < n; i++) {
      ans = contract(s, t);
      bin[t] = true;
      if (mincut > ans) {
        mincut = ans;
        cout << "iter " << i << ": " << ans << '\n';
      }
      if (mincut <= 2)
        return mincut;
      for (auto it = edge[s].begin(); it != edge[s].end(); it++) {
        j = it->first;
        if (!bin[j]) {
          if (edge[j].find(t) != edge[j].end()) {
            it->second = (edge[j][s] += edge[j][t]);
          }
        }
      }
      for (auto it = edge[t].begin(); it != edge[t].end(); it++) {
        j = it->first;
        if (!bin[j]) {
          if (edge[j].find(s) == edge[j].end()) {
            edge[j][s] = edge[s][j] = it->second;
          }
        }
      }
    }
    return mincut;
  }
} sw;

struct HLD {
  vector<int> G[MAXN];              // 原图
  vector<pair<int, int>> backEdges; // 非树边

  int parent[MAXN], depth[MAXN], heavy[MAXN], head[MAXN];
  int dfn[MAXN], dfc;
  int subtree_size[MAXN];
  vector<vector<int>> tree;

  int timer = 0;
  int ring_count[MAXN]; // 每个点所在的简单环数量
  int ring_sum[MAXN];

  int n, m; // 点数、边数

  void addedge(int u, int v) {
    G[u].push_back(v);
    G[v].push_back(u);
  }

  // Step 1: DFS to build tree and identify back edges
  void dfsTree(int u, int p) {
    parent[u] = p;
    dfn[u] = ++dfc;
    depth[u] = (p == -1) ? 0 : depth[p] + 1;
    subtree_size[u] = 1;
    int max_subtree = -1;

    for (int v : G[u]) {
      if (v == p)
        continue;
      if (dfn[v] != -1) {
        // Already visited, so it's a back edge
        if (depth[v] < depth[u]) {
          backEdges.emplace_back(u, v);
        }
        continue;
      }

      tree[u].push_back(v);
      dfsTree(v, u);
      subtree_size[u] += subtree_size[v];
      if (subtree_size[v] > max_subtree) {
        max_subtree = subtree_size[v];
        heavy[u] = v;
      }
    }
  }

  // Step 2: Decompose tree
  void decompose(int u, int h) {
    head[u] = h;

    if (heavy[u] != -1) {
      decompose(heavy[u], h);
    }

    for (int v : tree[u]) {
      if (v != heavy[u]) {
        decompose(v, v);
      }
    }
  }

  // Step 3: LCA with HLD
  int lca(int u, int v) {
    while (head[u] != head[v]) {
      if (depth[head[u]] < depth[head[v]])
        swap(u, v);
      u = parent[head[u]];
    }
    return depth[u] < depth[v] ? u : v;
  }

  // Entry point
  void solve() {
    int k = 0;
    for (int u = 1; u <= n; ++u) {
      if (u >= n / 10 * k) {
        cout << "ring count solve = " << u << " / " << n << '\n';
        k++;
      }

      // reset
      memset(parent, -1, sizeof(int) * (n + 1));
      memset(heavy, -1, sizeof(int) * (n + 1));
      memset(dfn, -1, sizeof(int) * (n + 1));
      dfc = 0;
      backEdges.clear();
      tree.clear();
      tree.resize(n + 1);

      // Step 1: build DFS tree from u
      dfsTree(u, -1);

      // Step 2: build HLD structure
      decompose(u, u);

      // Step 3: process back edges
      for (auto &[x, y] : backEdges) {
        int anc = lca(x, y);
        if (anc == u) {
          ring_count[u]++;
          int l = depth[x] + depth[y] - 2 * depth[u] + 1;
          ring_sum[l]++;
        }
      }
    }
    for (int i = 3; i <= n; i++) {
      // cout << i << ' ' << ring_sum[i] << '\n';
      // assert(ring_sum[i] % i == 0);
      ring_sum[i] /= i;
    }
  }

} hld;

int get_prt(map<int, int> &prt, int x) {
  if (prt[x] == x)
    return x;
  return prt[x] = get_prt(prt, prt[x]);
}

void circleSum(vector<int> sp0_node_id, vector<Edge> sp0_edge) {
  map<int, int> prt;
  map<int, int> deg;
  map<int, vector<int>> deg_map;
  for (auto &u : sp0_node_id) {
    prt[u] = u;
  }
  for (auto &e : sp0_edge) {
    deg[e.idA] += e.type.size();
    deg[e.idB] += e.type.size();
    int pA = get_prt(prt, e.idA);
    int pB = get_prt(prt, e.idB);
    prt[pA] = pB;
  }
  for (auto &u : sp0_node_id) {
    deg_map[deg[u]].push_back(u);
  }

  cout << "=== Center Clique Summary Starts ===\n";
  outFile << ".CenterCliqueSummaryDeg\n";
  for (auto it = deg_map.begin(); it != deg_map.end(); it++) {
    cout << "deg " << it->first << ": size = " << it->second.size() << '\n';
    outFile << it->first << ' ' << it->second.size() << '\n';
    for (auto &u : it->second) {
      cout << V[u].name << ' ';
    }
    cout << '\n';
  }
  outFile << ".END\n";

  int p = get_prt(prt, sp0_node_id[0]);
  for (auto &u : sp0_node_id) {
    assert(get_prt(prt, u) == p);
  }
  map<int, int> id_2_hld;
  map<int, int> hld_2_id;
  for (auto &u : sp0_node_id) {
    if (V[u].name == "vdd" || V[u].name == "gnd")
      continue;
    id_2_hld[u] = ++hld.n;
    hld_2_id[hld.n] = u;
  }
  for (auto &e : sp0_edge) {
    if (e.nameA == "vdd" || e.nameB == "vdd")
      continue;
    if (e.nameA == "gnd" || e.nameB == "gnd")
      continue;
    int idA = id_2_hld[e.idA];
    int idB = id_2_hld[e.idB];
    hld.addedge(idA, idB);
  }
  hld.solve();
  map<int, int> ring_cnt;
  map<int, vector<int>> ring_cnt_map;
  for (int i = 1; i <= hld.n; i++) {
    int u = hld_2_id[i];
    ring_cnt[u] = hld.ring_count[i];
    if (ring_cnt[u] == 0) {
      ring_cnt_map[1].push_back(u);
    } else {
      ring_cnt_map[ring_cnt[u]].push_back(u);
    }
  }

  outFile << ".CenterCliqueSummaryNodeInRing\n";
  for (auto it = ring_cnt_map.begin(); it != ring_cnt_map.end(); it++) {
    cout << "ring_cnt " << it->first << ": size = " << it->second.size()
         << '\n';
    outFile << it->first << ' ' << it->second.size() << '\n';
    for (auto &u : it->second) {
      cout << V[u].name << ' ';
    }
    cout << '\n';
  }
  outFile << ".END\n";

  outFile << ".CenterCliqueSummarySimpleRing\n";
  for (int i = 3; i <= hld.n; i++) {
    if (hld.ring_sum[i]) {
      cout << "ring_size " << i << ": cnt = " << hld.ring_sum[i] << '\n';
      outFile << i << ' ' << hld.ring_sum[i] << '\n';
    }
  }
  outFile << ".END\n";
  cout << "=== Center Clique Summary Ends ===\n";
}

void main(const vector<string> &rawLines, vector<string> split0) {
  cout << "=== globalMinCut main starts ===\n";
  bool isInSubckt = false;
  Subckt currentSubckt;
  map<string, int> portMap;
  vector<string> rawLines2;
  for (const string &line : rawLines) {
    if (line.substr(0, 7) == ".subckt") {
      portMap.clear();
      assert(!isInSubckt);
      isInSubckt = true;
      istringstream iss(line);
      string dot, name;
      iss >> dot >> name;
      currentSubckt = Subckt{name};
      string port;
      while (iss >> port) {
        currentSubckt.ports_name.push_back(port);
        portMap[port] = 1;
      }
    } else if (line.substr(0, 5) == ".ends") {
      isInSubckt = false;
      subckts[currentSubckt.name] = currentSubckt;
    } else if (isInSubckt) {
      currentSubckt.content.push_back(line);

      istringstream subIss(line);
      vector<string> subTokens;
      string w1, d;
      subIss >> w1;
      transform(w1.begin(), w1.end(), w1.begin(), ::tolower);
      assert(w1[0] == 'm' || w1[0] == 'c' || w1[0] == 'r');
      int iter = 0;
      if (w1[0] == 'm')
        iter = 4;
      if (w1[0] == 'c' || w1[0] == 'r')
        iter = 2;
      for (int i = 0; i < iter; i++) {
        subIss >> d;
        if (portMap.count(d) == 0) {
          portMap[d] = 1;
          currentSubckt.inside_nodes_name.push_back(d);
        }
      }
    } else {
      rawLines2.push_back(line);
    }
  }
  node_id_cnt = 0;
  edge_id_cnt = 0;
  V.push_back(Node("null", "null"));
  addNode("gnd", "gnd");
  for (const string &line : rawLines2) {
    if (line[0] == '.')
      continue;
    istringstream subIss(line);
    string w, d;
    subIss >> w;
    transform(w.begin(), w.end(), w.begin(), ::tolower);
    assert(w[0] == 'v' || w[0] == 'r' || w[0] == 'x');
    if (w[0] == 'v' || w[0] == 'r') {
      string a, b;
      subIss >> a >> b;
      addNode(a, "node");
      addNode(b, "node");
      addEdge(a, b, w.substr(0, 1), line);
    } else if (w[0] == 'x') {
      string a, token;

      vector<string> tokens;
      while (subIss >> token) {
        tokens.push_back(token);
      }
      assert(tokens.size() >= 2);

      string subcktName = tokens.back();
      assert(subckts.find(subcktName) != subckts.end());

      buildSubckt(tokens, subcktName);
    }
  }

  edge_id_cnt = E.size();

  outFile << ".GlobalMinCutTotalNodeEdge\n";
  outFile << node_id_cnt << ' ' << edge_id_cnt << '\n';
  outFile << ".END\n";

  cout << "total node count = " << node_id_cnt << '\n';
  cout << "total edge count = " << edge_id_cnt << '\n';
  cout << "== Node Info Starts ==\n";
  for (int i = 1; i <= node_id_cnt; i++) {
    cout << i << ' ' << V[i].name << ' ' << V[i].type << ' '
         << "rd = " << V[i].rd << ", cd = " << V[i].cd << ", deg = " << V[i].deg
         << ", depth = " << V[i].depth << '\n';
  }
  cout << "== Node Info Ends ==\n";

  mf.init();
  mf.n = node_id_cnt;
  mf.S = get_node_id("vdd");
  mf.T = get_node_id("gnd");

  cout << "== Edge Info Starts ==\n";
  map<int, pair<int, int>> rev_map;
  map<int, int> split0_map;
  sw.n = 0;

  vector<int> split0_node_id;
  vector<Edge> split0_edge;
  for (auto &nam : split0) {
    int u = get_node_id(nam);
    split0_map[u] = ++sw.n;
    split0_node_id.push_back(u);
  }
  for (auto &p : E) {
    Edge &e = p.second;
    assert(e.type.size() == e.label.size());
    cout << e.nameA << '(' << e.idA << ')' << " --- " << ' ' << e.nameB << '('
         << e.idB << ')' << ' ' << "weight = " << e.type.size() << " "
         << (e.type.size() > 1 ? "multiedge" : "") << '\n';

    for (int i = 0; i < e.type.size(); i++) {
      cout << "type = " << e.type[i] << ' ' << "label = " << e.label[i] << '\n';
    }

    e.mf_edge_id = mf.cnt;
    rev_map[mf.cnt] = mp(e.idA, e.idB);
    rev_map[mf.cnt + 1] = mp(e.idA, e.idB);
    mf.addedge2(e.idA, e.idB, e.type.size());

    if (split0_map.count(e.idA) > 0 && split0_map.count(e.idB) > 0) {
      int sw_idA = split0_map[e.idA], sw_idB = split0_map[e.idB];
      sw.edge[sw_idA][sw_idB] = e.type.size();
      sw.edge[sw_idB][sw_idA] = e.type.size();
      split0_edge.push_back(e);
    }
  }
  cout << "== Edge Info Ends ==\n";

  mf.dinic();
  vector<vector<pair<int, int>>> C = mf.C;
  outFile << ".GlobalMinCutVddGndMaxflow\n";
  outFile << mf.maxflow << '\n';
  outFile << ".END\n";
  cout << "maxflow = " << mf.maxflow << '\n';

  assert(mf.vis_cnt == 2);
  assert(mf.maxflow == V[get_node_id("vdd")].deg);
  // cout << "total blocks = " << mf.vis_cnt << '\n';
  /*
  for (int u = 1; u < C.size(); u++) {
    for (int j = 0; j < C[u].size(); j++) {
      auto [v, mf_edge_id] = C[u][j];
      auto [idA, idB] = rev_map[mf_edge_id];
      auto &e = E[mp(idA, idB)];

      cout << V[idA].name << ' ' << V[idB].name << ", flow = " << e.type.size()
           << '\n';
      for (int k = 0; k < e.type.size(); k++) {
        cout << " " << e.type[k] << " " << e.label[k] << '\n';
      }
    }
    cout << '\n';
  }
  */

  int tmp = sw.Stoer_Wagner();
  outFile << ".GlobalMinCutOfCenter\n";
  outFile << tmp << '\n';
  outFile << ".END\n";
  cout << "global mincut = " << tmp << '\n';

  circleSum(split0_node_id, split0_edge);

  cout << "=== globalMinCut main ends ===\n";
}
} // namespace globalMinCut

namespace findCutPoint {
struct Node {
  string name;
  string type;
  int rd, cd, deg, depth;
  Node(string _name, string _type) {
    name = _name;
    type = _type;
    rd = 0;
    cd = 0;
    deg = 0;
    depth = 0;
  }
};
vector<Node> V;

struct Edge {
  string nameA, nameB;
  int idA, idB;
  bool is_directed;
  string label;
  string type;
  Edge() {
    nameA = "null";
    nameB = "null";
    idA = 0;
    idB = 0;
    is_directed = 0;
    label = "null";
    type = "null";
  }
};

struct Starry {
  int to, edge_id;
  Starry(int _to = 0, int _edge_id = 0) {
    to = _to;
    edge_id = _edge_id;
  }
};

struct Subckt {
  string name;
  vector<string> ports_name;
  vector<string> inside_nodes_name;
  vector<string> content; // lines inside subckt
  int id_cnt;
};

vector<Edge> E;

const int MAXN = 2000005;
vector<Starry> G[MAXN];
unordered_map<string, int> node_name2id;
unordered_map<int, string> node_id2name;
int node_id_cnt, edge_id_cnt;

unordered_map<string, Subckt> subckts;

int dfn[MAXN], low[MAXN], timestamp = 0;

// 割点相关（用于无向图）
int dfs_clock = 0;
int pre[MAXN], low_cut[MAXN];
bool is_cut[MAXN]; // 是否为割点

/*
bool inStack[MAXN];
stack<int> stk;
vector<vector<int>> sccs; // 存储强连通分量
int scc_id[MAXN];         // 每个点所在的SCC编号
void tarjanSCC(int u) {
  dfn[u] = low[u] = ++timestamp;
  stk.push(u);
  inStack[u] = true;
  for (auto &e : G[u]) {
    int v = e.to;
    if (!dfn[v]) {
      tarjanSCC(v);
      low[u] = min(low[u], low[v]);
    } else if (inStack[v]) {
      low[u] = min(low[u], dfn[v]);
    }
  }

  if (dfn[u] == low[u]) {
    vector<int> scc;
    int v;
    do {
      v = stk.top();
      stk.pop();
      inStack[v] = false;
      scc_id[v] = sccs.size();
      scc.push_back(v);
    } while (v != u);
    sccs.push_back(scc);
  }
}
*/

vector<int> block_sizes[MAXN]; // 每个割点产生的块大小
vector<int> block_v[MAXN];
int prt[MAXN], clr[MAXN], clr_cnt;

int subtree_size[MAXN]; // 子树大小（辅助统计）

// 割点（适用于无向图）
void dfs_cut(int u, int parent) {
  if (V[u].name == "vdd" || V[u].name == "gnd")
    return;
  prt[u] = parent;
  pre[u] = low_cut[u] = ++dfs_clock;
  subtree_size[u] = 1;
  int child = 0;
  for (auto &e : G[u]) {
    int v = e.to;
    if (V[v].name == "vdd" || V[v].name == "gnd")
      continue;
    if (!pre[v]) {
      child++;
      dfs_cut(v, u);
      subtree_size[u] += subtree_size[v];
      low_cut[u] = min(low_cut[u], low_cut[v]);
      if ((parent != -1 && low_cut[v] >= pre[u]) || parent == -1) {
        is_cut[u] = true;
        block_sizes[u].push_back(subtree_size[v]);
        block_v[u].push_back(v);
      }
    } else if (v != parent) {
      low_cut[u] = min(low_cut[u], pre[v]);
    }
  }
  if (parent == -1) {
    is_cut[u] = (child > 1);
    if (child <= 1) {
      block_sizes[u].clear();
      block_v[u].clear();
    }
  }
}

int ddd[MAXN];
map<int, int> ddd_map;
map<int, vector<int>> ddd_map_node;
void dfs_clr(int u) {
  if (clr[u] == -1 || clr[u] == clr_cnt)
    return;
  assert(clr[u] == 0);
  clr[u] = clr_cnt;
  ddd_map[ddd[u]]++;
  ddd_map_node[ddd[u]].push_back(u);
  for (auto &e : G[u]) {
    int v = e.to;
    if (V[v].name == "vdd" || V[v].name == "gnd")
      continue;
    if (clr[v] == -1 || clr[v] == clr_cnt)
      continue;
    ddd[v] = ddd[u] + 1;
    dfs_clr(v);
  }
}

vector<string> Tarjan() {
  int n = node_id_cnt;
  memset(pre, 0, sizeof(pre));
  memset(is_cut, 0, sizeof(is_cut));
  dfs_clock = 0;
  for (int i = 1; i <= n; ++i)
    if (!pre[i])
      dfs_cut(i, -1);

  clr_cnt = 0;

  map<string, int> cutp_type_sum;
  map<int, int> split_blocks_sum;
  memset(ddd, 0, sizeof(ddd));

  int chain_sum = 0;

  cout << "割点:" << endl;
  for (int i = 1; i <= n; ++i) {
    if (is_cut[i]) {
      if (V[i].name.substr(0, 4) == "netg")
        continue;
      cout << V[i].name << " split = " << block_sizes[i].size() + 1
           << ", blocks = ";
      int rest = n - 1; // 总点数减去当前割点
      for (int sz : block_sizes[i]) {
        cout << sz << " ";
        rest -= sz;
      }
      if (rest > 0) {
        cout << rest; // 加上剩下的一块（被多个分支以外的点组成）
        block_sizes[i].push_back(rest);
        block_v[i].push_back(prt[i]);
      }

      cutp_type_sum[V[i].type]++;
      split_blocks_sum[block_sizes[i].size()]++;

      cout << '\n';
      int id = -1, mx = -1;
      for (int j = 0; j < block_sizes[i].size(); j++) {
        if (block_sizes[i][j] > mx) {
          mx = block_sizes[i][j];
          id = j;
        }
      }
      clr[i] = -1;
      ddd[i] = 0;
      for (int j = 0; j < block_v[i].size(); j++) {
        if (j == id)
          continue;
        int v = block_v[i][j];
        if (clr[v]) {
        } else {
          clr_cnt++;
          ddd_map.clear();
          ddd_map_node.clear();
          ddd[v] = ddd[i] + 1;
          dfs_clr(v);
          auto it = ddd_map.end();
          it--;
          int mmm = (*it).first;
          int mx2 = 0;
          for (int k = 1; k <= mmm; k++) {
            cout << ddd_map[k] << ' ';
            if (ddd_map[k] > 1)
              mx2 = k;
            /*
            for (auto uuu : ddd_map_node[k]) {
              cout << V[uuu].name << ' ';
            }
            */
          }
          chain_sum += mmm - mx2;
          cout << '\n';
        }
      }
    }
  }
  cout << endl;

  cout << "clr_cnt = " << clr_cnt << '\n';

  cout << "netg割点:" << endl;
  for (int i = 1; i <= n; ++i) {
    if (is_cut[i] && clr[i] == 0) {
      assert(V[i].name.substr(0, 4) == "netg");
      cout << V[i].name << " split = " << block_sizes[i].size() + 1
           << ", blocks = ";
      int rest = n - 1; // 总点数减去当前割点
      for (int sz : block_sizes[i]) {
        cout << sz << " ";
        rest -= sz;
      }
      if (rest > 0) {
        cout << rest; // 加上剩下的一块（被多个分支以外的点组成）
        block_sizes[i].push_back(rest);
        block_v[i].push_back(prt[i]);
      }
      cutp_type_sum[V[i].type]++;
      split_blocks_sum[block_sizes[i].size()]++;

      cout << '\n';
      int id = -1, mx = -1;
      for (int j = 0; j < block_sizes[i].size(); j++) {
        if (block_sizes[i][j] > mx) {
          mx = block_sizes[i][j];
          id = j;
        }
      }
      clr[i] = -1;
      ddd[i] = 0;
      for (int j = 0; j < block_v[i].size(); j++) {
        if (j == id)
          continue;
        int v = block_v[i][j];
        if (clr[v]) {
        } else {
          ddd_map.clear();
          ddd_map_node.clear();
          ddd[v] = ddd[i] + 1;
          clr_cnt++;
          dfs_clr(v);
          auto it = ddd_map.end();
          it--;
          int mmm = (*it).first;
          int mx2 = 0;
          for (int k = 1; k <= mmm; k++) {
            cout << ddd_map[k] << ' ';
            if (ddd_map[k] > 1)
              mx2 = k;
            /*
            for (auto uuu : ddd_map_node[k]) {
              cout << V[uuu].name << ' ';
            }
            */
          }
          chain_sum += mmm - mx2;
          cout << '\n';
        }
      }
    }
  }
  cout << "chain_sum = " << chain_sum << '\n';
  assert(false);
  vector<vector<int>> split;
  vector<string> split0;
  split.resize(clr_cnt + 1);
  int sum_cut = 0, sum_split = 0;
  for (int i = 1; i <= n; i++) {
    if (clr[i] == -1) {
      sum_cut++;
      continue;
    }
    split[clr[i]].push_back(i);
    if (clr[i] > 0)
      sum_split++;
  }

  outFile << ".CutPoint\n";
  outFile << sum_cut << '\n';
  int tmp = 0;
  for (auto it = cutp_type_sum.begin(); it != cutp_type_sum.end(); it++) {
    cout << it->first << ' ' << it->second << '\n';
    tmp += it->second;
  }
  for (auto it = split_blocks_sum.begin(); it != split_blocks_sum.end(); it++) {
    cout << it->first << ' ' << it->second << '\n';
  }
  assert(tmp == sum_cut);
  outFile << clr_cnt + 1 << ' ' << sum_split << ' ' << n - sum_split << ' '
          << 1.0 * sum_split / n << '\n';

  assert(n - sum_split == split[0].size() + sum_cut);
  for (int i = 1; i <= clr_cnt; i++) {
    outFile << i << ' ' << split[i].size() << '\n';
  }
  outFile << ".END\n";

  for (int i = 0; i <= clr_cnt; i++) {
    cout << "color " << i << ", size = " << split[i].size() << "; ";
    for (auto &u : split[i]) {
      cout << V[u].name << ' ';
    }
    cout << '\n';
  }
  cout << "sum_cut = " << sum_cut << ", sum_split = " << sum_split << '\n';
  for (auto &u : split[0]) {
    split0.push_back(V[u].name);
  }
  for (int i = 1; i <= n; i++) {
    if (clr[i] == -1) {
      split0.push_back(V[i].name);
    }
  }
  return split0;
}

int get_node_id(string a) {
  assert(node_name2id.count(a) > 0);
  return node_name2id[a];
}
string get_node_name(int id) {
  assert(node_id2name.count(id) > 0);
  return node_id2name[id];
}

void addNode(string a, string ty) {
  if (ty == "subckt") {
    if (a.substr(0, 3) == "net")
      ty = a.substr(0, 4);
    if (is_logic_dev(a))
      ty = logic_dev_name(a);
  }
  if (ty == "node") {
    ty = a[0];
  }
  if (node_name2id.count(a) == 0) {
    node_name2id[a] = ++node_id_cnt;
    node_id2name[node_id_cnt] = a;
    V.push_back(Node(a, ty));
  } else {
    int id = node_name2id[a];
    if (V[id].type == "gnd")
      return;
    assert(V[id].type == ty);
  }
}

void addEdge(string a, string b, string is_d, string ty, string label) {
  bool is_directed = false;
  assert(is_d == "directed" || is_d == "undirected");
  if (is_d == "directed")
    is_directed = true;
  else if (is_d == "undirected")
    is_directed = false;
  E.push_back(Edge());
  int id = ++edge_id_cnt;
  E[id].nameA = a;
  E[id].nameB = b;
  E[id].idA = get_node_id(a);
  E[id].idB = get_node_id(b);
  E[id].type = ty;
  E[id].label = label;
  E[id].is_directed = is_directed;

  G[E[id].idA].push_back(Starry(E[id].idB, id));
  if (is_directed) {
    V[E[id].idB].rd++;
    V[E[id].idA].cd++;
  } else {
    V[E[id].idA].deg++;
    V[E[id].idB].deg++;
  }
  if (!is_directed) {
    G[E[id].idB].push_back(Starry(E[id].idA, id));
  }
}

map<string, int> sum_r;
map<string, int> sum_c;
map<string, int> sum_m;

void buildSubckt(vector<string> tokens, string subcktName) {
  Subckt &tmpl = subckts[subcktName];
  size_t portCount = tmpl.ports_name.size();
  assert(tokens.size() == portCount + 1);

  tmpl.id_cnt++;
  string ckt = subcktName + "_" + to_string(tmpl.id_cnt);
  // addNode(subcktName + "_" + to_string(tmpl.id_cnt), "subckt");
  // param mapping: tmpl.ports[i] -> tokens[i+1]
  unordered_map<string, string> portMap;

  for (size_t i = 0; i < portCount; ++i) {
    portMap[tmpl.ports_name[i]] = tokens[i];
    addNode(tokens[i], "node");
  }
  for (auto &v : tmpl.inside_nodes_name) {
    addNode(ckt + "_" + v, "inckt");
  }

  for (const string &line : tmpl.content) {
    istringstream Iss(line);
    string w, a[15], d, g, s, b, res = "", tmp;
    Iss >> w;
    transform(w.begin(), w.end(), w.begin(), ::tolower);
    assert(w[0] == 'm' || w[0] == 'c' || w[0] == 'r');
    int iter = 2;
    if (w[0] == 'm')
      iter = 4;
    for (int i = 0; i < iter; i++) {
      Iss >> a[i];
      if (portMap.count(a[i]) == 0) {
        a[i] = ckt + "_" + a[i];
      } else {
        a[i] = portMap[a[i]];
      }
    }
    while (Iss >> tmp) {
      res += tmp + " ";
    }
    if (w[0] == 'c' || w[0] == 'r') {
      if (w[0] == 'c')
        sum_c[res]++;
      if (w[0] == 'r')
        sum_r[res]++;
      addEdge(a[0], a[1], "undirected", w.substr(0, 1), ckt + "_line=" + line);
    }
    if (w[0] == 'm') {
      sum_m[res]++;
      d = a[0];
      g = a[1];
      s = a[2];
      b = a[3];
      if (res.substr(0, 4) == "pmos")
        assert(b == "vdd");
      if (res.substr(0, 4) == "nmos")
        assert(b == "gnd");
      string d_prime = ckt + "_" + w + "_" + d + "_" + "prime";
      string s_prime = ckt + "_" + w + "_" + s + "_" + "prime";
      addNode(d_prime, "inckt");
      addNode(s_prime, "inckt");
      addEdge(d, d_prime, "undirected", w.substr(0, 1) + "_d-dp",
              ckt + "_line=" + line);
      addEdge(s, s_prime, "undirected", w.substr(0, 1) + "_s-sp",
              ckt + "_line=" + line);
      addEdge(g, d_prime, "undirected", w.substr(0, 1) + "_g-dp",
              ckt + "_line=" + line);
      addEdge(g, s_prime, "undirected", w.substr(0, 1) + "_g-sp",
              ckt + "_line=" + line);
      addEdge(d_prime, s_prime, "undirected", w.substr(0, 1) + "_dp-sp",
              ckt + "_line=" + line);
    }
  }
}

vector<string> main(const vector<string> &rawLines) {
  cout << "=== findCutPoint main starts ===\n";
  bool isInSubckt = false;
  Subckt currentSubckt;
  map<string, int> portMap;
  vector<string> rawLines2;
  for (const string &line : rawLines) {
    if (line.substr(0, 7) == ".subckt") {
      portMap.clear();
      assert(!isInSubckt);
      isInSubckt = true;
      istringstream iss(line);
      string dot, name;
      iss >> dot >> name;
      currentSubckt = Subckt{name};
      string port;
      while (iss >> port) {
        currentSubckt.ports_name.push_back(port);
        portMap[port] = 1;
      }
    } else if (line.substr(0, 5) == ".ends") {
      isInSubckt = false;
      subckts[currentSubckt.name] = currentSubckt;
    } else if (isInSubckt) {
      currentSubckt.content.push_back(line);

      istringstream subIss(line);
      vector<string> subTokens;
      string w1, d;
      subIss >> w1;
      transform(w1.begin(), w1.end(), w1.begin(), ::tolower);
      assert(w1[0] == 'm' || w1[0] == 'c' || w1[0] == 'r');
      int iter = 0;
      if (w1[0] == 'm')
        iter = 4;
      if (w1[0] == 'c' || w1[0] == 'r')
        iter = 2;
      for (int i = 0; i < iter; i++) {
        subIss >> d;
        if (portMap.count(d) == 0) {
          portMap[d] = 1;
          currentSubckt.inside_nodes_name.push_back(d);
        }
      }
    } else {
      rawLines2.push_back(line);
    }
  }
  node_id_cnt = 0;
  edge_id_cnt = 0;
  V.push_back(Node("null", "null"));
  E.push_back(Edge());
  addNode("gnd", "gnd");
  for (const string &line : rawLines2) {
    if (line[0] == '.')
      continue;
    istringstream subIss(line);
    string w, d;
    subIss >> w;
    transform(w.begin(), w.end(), w.begin(), ::tolower);
    assert(w[0] == 'v' || w[0] == 'r' || w[0] == 'x');
    if (w[0] == 'v' || w[0] == 'r') {
      string a, b;
      subIss >> a >> b;
      addNode(a, "node");
      addNode(b, "node");
      addEdge(a, b, "undirected", w.substr(0, 1), line);
    } else if (w[0] == 'x') {
      string a, token;

      vector<string> tokens;
      while (subIss >> token) {
        tokens.push_back(token);
      }
      assert(tokens.size() >= 2);

      string subcktName = tokens.back();
      assert(subckts.find(subcktName) != subckts.end());

      buildSubckt(tokens, subcktName);
    }
  }
  int sum = 0;
  for (auto it = sum_r.begin(); it != sum_r.end(); it++) {
    sum += it->second;
  }
  outFile << ".DevTypeR\n";
  outFile << sum_r.size() << ' ' << sum << '\n';
  cout << "Res type = " << sum_r.size() << ", tot = " << sum << '\n';
  for (auto it = sum_r.begin(); it != sum_r.end(); it++) {
    outFile << it->first << " " << it->second << '\n';
    cout << it->first << " " << it->second << '\n';
  }
  outFile << ".END\n";

  sum = 0;
  for (auto it = sum_c.begin(); it != sum_c.end(); it++) {
    sum += it->second;
  }
  outFile << ".DevTypeC\n";
  outFile << sum_c.size() << ' ' << sum << '\n';
  cout << "Cap type = " << sum_c.size() << ", tot = " << sum << '\n';
  for (auto it = sum_c.begin(); it != sum_c.end(); it++) {
    outFile << it->first << " " << it->second << '\n';
    cout << it->first << " " << it->second << '\n';
  }
  outFile << ".END\n";

  sum = 0;
  for (auto it = sum_m.begin(); it != sum_m.end(); it++) {
    sum += it->second;
  }
  outFile << ".DevTypeM\n";
  outFile << sum_m.size() << ' ' << sum << '\n';
  cout << "Mos type = " << sum_m.size() << ", tot = " << sum << '\n';
  for (auto it = sum_m.begin(); it != sum_m.end(); it++) {
    outFile << it->first << " " << it->second << '\n';
    cout << it->first << " " << it->second << '\n';
  }
  outFile << ".END\n";

  outFile << ".FindCutPointNodeEdge\n";
  outFile << node_id_cnt << ' ' << edge_id_cnt << '\n';
  outFile << ".END\n";

  cout << "total node count = " << node_id_cnt << '\n';
  cout << "total edge count = " << edge_id_cnt << '\n';
  cout << "== Node Info Starts ==\n";
  for (int i = 1; i <= node_id_cnt; i++) {
    cout << i << ' ' << V[i].name << ' ' << V[i].type << ' '
         << "rd = " << V[i].rd << ", cd = " << V[i].cd << ", deg = " << V[i].deg
         << ", depth = " << V[i].depth << '\n';
  }
  cout << "== Node Info Ends ==\n";

  cout << "== Edge Info Starts ==\n";
  for (int i = 1; i <= edge_id_cnt; i++) {
    cout << E[i].nameA << '(' << E[i].idA << ')' << ' '
         << (E[i].is_directed ? "-->" : "---") << ' ' << E[i].nameB << '('
         << E[i].idB << ')' << ' ' << "type = " << E[i].type << ' '
         << "label = " << E[i].label << '\n';
  }
  cout << "== Edge Info Ends ==\n";

  vector<string> split0 = Tarjan();
  cout << "=== findCutPoint main ends ===\n";
  return split0;
}
} // namespace findCutPoint

namespace generalGraph {
struct Node {
  string name;
  string type;
  int rd, cd, deg, depth;
  Node(string _name, string _type) {
    name = _name;
    type = _type;
    rd = 0;
    cd = 0;
    deg = 0;
    depth = 0;
  }
};
vector<Node> V;

struct Edge {
  string nameA, nameB;
  int idA, idB;
  bool is_directed;
  string label;
  string type;
  Edge() {
    nameA = "null";
    nameB = "null";
    idA = 0;
    idB = 0;
    is_directed = 0;
    label = "null";
    type = "null";
  }
};

struct Starry {
  int to, edge_id;
  Starry(int _to = 0, int _edge_id = 0) {
    to = _to;
    edge_id = _edge_id;
  }
};

struct Subckt {
  string name;
  vector<string> ports_name;
  vector<string> inside_nodes_name;
  vector<string> content; // lines inside subckt
  int id_cnt;
};

vector<Edge> E;

const int MAXN = 2000005;
vector<Starry> G[MAXN];
unordered_map<string, int> node_name2id;
unordered_map<int, string> node_id2name;
int node_id_cnt, edge_id_cnt;

unordered_map<string, Subckt> subckts;

int get_node_id(string a) {
  assert(node_name2id.count(a) > 0);
  return node_name2id[a];
}
string get_node_name(int id) {
  assert(node_id2name.count(id) > 0);
  return node_id2name[id];
}

void addNode(string a, string ty) {
  if (ty == "subckt") {
    if (a.substr(0, 3) == "net")
      ty = a.substr(0, 4);
    if (is_logic_dev(a))
      ty = logic_dev_name(a);
  }
  if (ty == "node") {
    ty = a[0];
  }
  if (node_name2id.count(a) == 0) {
    node_name2id[a] = ++node_id_cnt;
    node_id2name[node_id_cnt] = a;
    V.push_back(Node(a, ty));
  } else {
    int id = node_name2id[a];
    if (V[id].type == "gnd")
      return;
    assert(V[id].type == ty);
  }
}

void topSort() {
  vector<int> cd, rd, deg, depth;
  vector<bool> vis;
  int n = node_id_cnt;
  cd.resize(n + 1);
  rd.resize(n + 1);
  deg.resize(n + 1);
  depth.resize(n + 1);
  vis.resize(n + 1);
  for (int i = 1; i <= n; i++) {
    cd[i] = V[i].cd;
    rd[i] = V[i].rd;
    deg[i] = V[i].deg;
  }
  queue<int> q;
  for (int i = 1; i <= n; i++) {
    for (auto &e : G[i]) {
      int j = e.to;
      if (V[i].name == "vdd") {
        rd[j]--;
      }
      if (V[j].name == "gnd") {
        cd[i]--;
      }
    }
  }

  for (int i = 1; i <= n; ++i) {
    if (V[i].name == "vdd" || V[i].name == "gnd") {
      vis[i] = 1;
      continue;
    }
    if (rd[i] == 0 && ((cd[i] == 0 && deg[i] == 1) || deg[i] == 0)) {
      q.push(i);
      depth[i] = 1;
      vis[i] = true;
    }
  }

  while (true) {
    while (!q.empty()) {
      int u = q.front();
      q.pop();

      for (auto &e : G[u]) {
        int v = e.to, eid = e.edge_id;
        if (vis[v])
          continue;

        if (E[eid].is_directed) {
          rd[v]--;
          if (V[u].name == "vdd")
            continue;
          // cout << "--> " << v << ' ' << rd[v] << ' ' << deg[v] << '\n';
          depth[v] = max(depth[v], depth[u] + 1);
          if (rd[v] == 0 && deg[v] <= 1) {
            q.push(v);
            vis[v] = true;
          }
        } else {
          deg[v]--;
          depth[v] = max(depth[v], depth[u] + 1);
          // cout << "--- " << v << ' ' << rd[v] << ' ' << deg[v] << '\n';
          if (rd[v] == 0 && deg[v] <= 1) {
            q.push(v);
            vis[v] = true;
          }
        }
      }
    }
    for (int i = 2; i <= n; i++) {
      if (!vis[i] && depth[i] > 0 && !is_logic_dev(V[i].name)) {
        q.push(i);
        vis[i] = 1;
      }
    }
    if (q.empty())
      break;
  }
  for (int i = 1; i <= n; i++) {
    for (auto &e : G[i]) {
      int j = e.to;
      if (V[j].name == "gnd") {
        depth[j] = max(depth[j], depth[i] + 1);
      }
    }
  }
  for (int i = 1; i <= n; i++) {
    if (!vis[i]) {
      cout << V[i].name << " is not visited\n";
    }
  }
  for (int i = 1; i <= n; i++) {
    V[i].depth = depth[i];
  }
}

void addEdge(string a, string b, string is_d, string ty, string label) {
  if (a == "gnd") {
    swap(a, b);
  }
  if (b == "vdd") {
    swap(a, b);
  }
  if (a == "vdd" || b == "gnd") {
    is_d = "directed";
  }
  bool is_directed = false;
  assert(is_d == "directed" || is_d == "undirected");
  if (is_d == "directed")
    is_directed = true;
  else if (is_d == "undirected")
    is_directed = false;
  E.push_back(Edge());
  int id = ++edge_id_cnt;
  E[id].nameA = a;
  E[id].nameB = b;
  E[id].idA = get_node_id(a);
  E[id].idB = get_node_id(b);
  E[id].type = ty;
  E[id].label = label;
  E[id].is_directed = is_directed;

  G[E[id].idA].push_back(Starry(E[id].idB, id));
  if (is_directed) {
    V[E[id].idB].rd++;
    V[E[id].idA].cd++;
  } else {
    V[E[id].idA].deg++;
    V[E[id].idB].deg++;
  }
  if (!is_directed) {
    G[E[id].idB].push_back(Starry(E[id].idA, id));
  }
}

void main(const vector<string> &rawLines) {
  cout << "=== generalGraph main starts ===\n";
  bool isInSubckt = false;
  Subckt currentSubckt;
  map<string, int> portMap;
  vector<string> rawLines2;
  for (const string &line : rawLines) {
    if (line.substr(0, 7) == ".subckt") {
      portMap.clear();
      assert(!isInSubckt);
      isInSubckt = true;
      istringstream iss(line);
      string dot, name;
      iss >> dot >> name;
      currentSubckt = Subckt{name};
      string port;
      while (iss >> port) {
        currentSubckt.ports_name.push_back(port);
        portMap[port] = 1;
      }
    } else if (line.substr(0, 5) == ".ends") {
      isInSubckt = false;
      subckts[currentSubckt.name] = currentSubckt;
    } else if (isInSubckt) {
      currentSubckt.content.push_back(line);

      istringstream subIss(line);
      vector<string> subTokens;
      string w1, d;
      subIss >> w1;
      transform(w1.begin(), w1.end(), w1.begin(), ::tolower);
      assert(w1[0] == 'm' || w1[0] == 'c' || w1[0] == 'r');
      int iter = 0;
      if (w1[0] == 'm')
        iter = 4;
      if (w1[0] == 'c' || w1[0] == 'r')
        iter = 2;
      for (int i = 0; i < iter; i++) {
        subIss >> d;
        if (portMap.count(d) == 0) {
          portMap[d] = 1;
          currentSubckt.inside_nodes_name.push_back(currentSubckt.name + '_' +
                                                    d);
        }
      }
    } else {
      rawLines2.push_back(line);
    }
  }
  node_id_cnt = 0;
  edge_id_cnt = 0;
  V.push_back(Node("null", "null"));
  E.push_back(Edge());
  addNode("gnd", "gnd");
  for (const string &line : rawLines2) {
    if (line[0] == '.')
      continue;
    istringstream subIss(line);
    string w, d;
    subIss >> w;
    transform(w.begin(), w.end(), w.begin(), ::tolower);
    assert(w[0] == 'v' || w[0] == 'r' || w[0] == 'x');
    if (w[0] == 'v' || w[0] == 'r') {
      string a, b;
      subIss >> a >> b;
      addNode(a, "node");
      addNode(b, "node");
      addEdge(a, b, "directed", w.substr(0, 1), line);
    } else if (w[0] == 'x') {
      string a, token;

      vector<string> tokens;
      while (subIss >> token) {
        tokens.push_back(token);
      }
      assert(tokens.size() >= 2);

      string subcktName = tokens.back();
      assert(subckts.find(subcktName) != subckts.end());

      Subckt &tmpl = subckts[subcktName];
      size_t portCount = tmpl.ports_name.size();
      assert(tokens.size() == portCount + 1);

      tmpl.id_cnt++;
      string ckt = subcktName + "_" + to_string(tmpl.id_cnt);
      addNode(subcktName + "_" + to_string(tmpl.id_cnt), "subckt");
      // param mapping: tmpl.ports[i] -> tokens[i+1]
      unordered_map<string, string> portMap;

      bool is_log_dev = is_logic_dev(ckt);
      for (size_t i = 0; i < portCount; ++i) {
        portMap[tmpl.ports_name[i]] = tokens[i];
        addNode(tokens[i], "node");
        if (is_log_dev) {
          if (tmpl.ports_name[i].length() == 1) {
            if (tmpl.ports_name[i] != "z") {
              addEdge(tokens[i], ckt, "directed", "no_r",
                      "ports: " + tmpl.ports_name[i]);
            } else {
              addEdge(ckt, tokens[i], "directed", "no_r",
                      "ports: " + tmpl.ports_name[i]);
            }
          } else {
            addEdge(ckt, tokens[i], "undirected", "no_r",
                    "ports: " + tmpl.ports_name[i]);
          }
        } else {
          addEdge(ckt, tokens[i], "undirected", "no_r",
                  "ports: " + tmpl.ports_name[i]);
        }
      }
    }
  }
  topSort();
  outFile << ".GeneralGraphNodeEdge\n";
  outFile << node_id_cnt << ' ' << edge_id_cnt << '\n';
  int directed_cnt = 0;
  for (int i = 1; i <= edge_id_cnt; i++) {
    directed_cnt += E[i].is_directed;
  }
  outFile << directed_cnt << ' ' << edge_id_cnt - directed_cnt << '\n';
  outFile << ".END\n";

  cout << "total node count = " << node_id_cnt << '\n';
  cout << "total edge count = " << edge_id_cnt << '\n';
  cout << "== Node Info Starts ==\n";
  for (int i = 1; i <= node_id_cnt; i++) {
    cout << i << ' ' << V[i].name << ' ' << V[i].type << ' '
         << "rd = " << V[i].rd << ", cd = " << V[i].cd << ", deg = " << V[i].deg
         << ", depth = " << V[i].depth << '\n';
  }
  cout << "== Node Info Ends ==\n";

  cout << "== Edge Info Starts ==\n";
  for (int i = 1; i <= edge_id_cnt; i++) {
    cout << E[i].nameA << '(' << E[i].idA << ", dep = " << V[E[i].idA].depth
         << ')' << ' ' << (E[i].is_directed ? "-->" : "---") << ' '
         << E[i].nameB << '(' << E[i].idB << ", dep = " << V[E[i].idB].depth
         << ')' << ' '
         << ((abs(V[E[i].idA].depth - V[E[i].idB].depth) != 1 && E[i].idB != 1)
                 ? (V[E[i].idA].depth < V[E[i].idB].depth ? "depth!=1<"
                                                          : "depth!=1>")
                 : "")
         << ' ' << "type = " << E[i].type << ' ' << "label = " << E[i].label
         << '\n';
  }
  cout << "== Edge Info Ends ==\n";

  int maxx = 0;
  for (int i = 1; i <= node_id_cnt; i++) {
    maxx = max(maxx, V[i].depth);
  }

  outFile << ".TopSortStruct\n";
  outFile << maxx << '\n';

  cout << "Layer = " << maxx << '\n';
  vector<vector<int>> layer;
  layer.resize(maxx + 1);
  for (int i = 1; i <= node_id_cnt; i++) {
    // cout << V[i].depth << '\n';
    layer[V[i].depth].push_back(i);
  }

  map<pair<int, int>, int> fb;
  for (int i = 1; i <= edge_id_cnt; i++) {
    int idA = E[i].idA, idB = E[i].idB;
    if (V[idA].name == "vdd" || V[idB].name == "gnd")
      continue;
    if (abs(V[idA].depth - V[idB].depth) != 1) {
      auto p = mp(V[idA].depth, V[idB].depth);
      if (fb.count(p) == 0)
        fb[p] = 0;
      fb[p]++;
    }
  }
  cout << "== Layer Info Starts ==\n";
  for (int i = 1; i < layer.size(); i++) {
    map<string, int> tot;
    for (int j = 0; j < layer[i].size(); j++) {
      int u = layer[i][j];
      tot[V[u].type]++;
    }
    outFile << i << ' ' << tot.size() << ' ';
    cout << "layer " << i << ' ';
    for (auto it = tot.begin(); it != tot.end(); it++) {
      outFile << it->first << ' ' << it->second << ' ';
      cout << it->first << " = " << it->second << ", ";
    }
    outFile << '\n';
    cout << '\n';
  }
  outFile << ".END\n";
  outFile << ".TopSortForwardBackwardEdge\n";
  cout << "== Layer Info Ends ==\n";
  for (auto it = fb.begin(); it != fb.end(); it++) {
    auto [dA, dB] = it->first;
    if (dA + 1 < dB) {
      outFile << "f ";
      cout << "forward ";
    }
    if (dB + 1 < dA) {
      outFile << "b ";
      cout << "backward ";
    }
    outFile << dA << ' ' << dB << ' ' << it->second << '\n';
    cout << "layer " << dA << " --> layer " << dB << " : " << it->second
         << '\n';
  }
  outFile << ".END\n";
  cout << "=== generalGraph main ends ===\n";
}
} // namespace generalGraph

struct Edge {
  string node1, node2;
  string label;
};

struct Subckt {
  string name;
  vector<string> ports;
  vector<string> content; // lines inside subckt
  int id_cnt;
};

unordered_map<string, Subckt> subckts;
vector<Edge> edges;
unordered_map<string, unordered_set<string>> adjacencyList;
vector<string> rawLines;

void addEdge(const string &a, const string &b, const string &label) {
  if (a == b)
    return;
  edges.push_back({a, b, label});
  adjacencyList[a].insert(b);
  adjacencyList[b].insert(a);
}

int main(int argc, char *argv[]) {
  bool isInSubckt = false;
  bool isInCtrl = false;
  Subckt currentSubckt;
  if (argc < 3) {
    cerr << "Usage: ./parser input output" << endl;
    return 1;
  }

  ifstream infile(argv[1]);
  if (!infile) {
    cerr << "Cannot open file: " << argv[1] << endl;
    return 1;
  }

  outFile.open("/dev/null");
  /*
  outName = argv[2];
  outFile.open(outName);
  */
  if (!outFile) {
    cerr << "Cannot open file: " << argv[2] << endl;
    return 1;
  }

  string line;
  // First pass: collect all lines
  while (getline(infile, line)) {
    // Remove comments
    size_t comment = line.find('*');
    if (comment != string::npos)
      line = line.substr(0, comment);
    line = regex_replace(line, regex("^\\s+|\\s+$"), "");
    if (line.substr(0, 8) == ".control") {
      isInCtrl = true;
    } else if (line.substr(0, 5) == ".endc") {
      isInCtrl = false;
    } else if (!isInCtrl && !line.empty()) {
      rawLines.push_back(line);
    }
  }
  infile.close();

  // generalGraph::main(rawLines);
  vector<string> split0 = findCutPoint::main(rawLines);
  // globalMinCut::main(rawLines, split0);
  return 0;
}

