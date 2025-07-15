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

vector<string> dotLines, ctrlLines;
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
  int is_device;
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

vector<int> block_sizes[MAXN]; // 每个割点产生的块大小
vector<int> block_v[MAXN];
int prt[MAXN], clr[MAXN], clr_cnt;

int subtree_size[MAXN]; // 子树大小（辅助统计）

int ddd[MAXN];
map<int, int> ddd_map;
map<int, vector<int>> ddd_map_node;

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

  if (tmpl.is_device) {
    for (int i = 0; i < portCount; i++) {
      addEdge(tokens[i], tokens[(i + 1) % portCount], "undirected", "null",
              "null");
    }
    return;
  }
  for (auto &v : tmpl.inside_nodes_name) {
    addNode(ckt + "_" + v, "inckt");
  }

  for (const string &line : tmpl.content) {
    istringstream Iss(line);
    string w, a[15], d, g, s, b, res = "", tmp;
    Iss >> w;
    transform(w.begin(), w.end(), w.begin(), ::tolower);
    assert(w[0] == 'c' || w[0] == 'r');
    int iter = 2;
    for (int i = 0; i < iter; i++) {
      Iss >> a[i];
      if (portMap.count(a[i]) == 0) {
        a[i] = ckt + "_" + a[i];
      } else {
        a[i] = portMap[a[i]];
      }
    }
    Iss >> res;
    if (w[0] == 'c' || w[0] == 'r') {
      if (w[0] == 'c')
        sum_c[res]++;
      if (w[0] == 'r')
        sum_r[res]++;
      addEdge(a[0], a[1], "undirected", w.substr(0, 1), res);
    }
  }
}

vector<bool> visited, marked;
string capa, resi;

int is_chain_component(int u, int pu = -1) {
  visited[u] = true;
  int deg = 0, sz = -2;
  for (auto &p : G[u]) {
    int v = p.to, eid = p.edge_id;
    if (v == 1) {
      if (E[eid].type == "c" && E[eid].label == capa) {
        continue;
      } else {
        if (E[eid].type == "c") {
          cout << "assert false = " << E[eid].type << ' ' << E[eid].label << ' '
               << capa << '\n';
          assert(false);
        }
        return -1;
      }
    }
    if (v == pu)
      continue;
    deg++;
    if (deg > 1)
      return -1;
    if (visited[v])
      return -1;
    if (E[eid].type != "r")
      return -1;
    if (E[eid].label != resi) {
      cout << "assert false = " << E[eid].type << ' ' << E[eid].label << ' '
           << resi << '\n';
      assert(false);
      return -1;
    }
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
  for (auto &p : G[u]) {
    int v = p.to;
    if (v == 1)
      continue;
    if (v == pu)
      continue;
    mark_chain_nodes(v, u);
  }
}

int now_head;
vector<int> prt_head;
int dfs_print(int u, int pu = -1) {
  prt_head[u] = now_head;
  if (pu == -1) {
    // cout << "\n" << V[u].name;
  }
  assert(marked[u] && V[u].deg <= 3);
  for (auto &p : G[u]) {
    int v = p.to, eid = p.edge_id;
    if (v == pu)
      continue;
    if (marked[v]) {
      // cout << "-" << E[eid].label << "-" << V[v].name;
      return dfs_print(v, u) + 1;
      break;
    }
  }
  return 1;
}

void main(const vector<string> &rawLines) {
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
      if (w1[0] == 'm') {
        iter = 4;
        currentSubckt.is_device = 1;
      }
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
      string res = "", tmp;
      while (subIss >> tmp) {
        res += tmp + " ";
      }
      addEdge(a, b, "undirected", w.substr(0, 1), res);
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
  assert(sum_c.size() <= 1);
  capa = sum_c.begin()->first;
  resi = sum_r.begin()->first;

  /*
  cout << "total node count = " << node_id_cnt << '\n';
  cout << "total edge count = " << edge_id_cnt << '\n';
  cout << "== Edge Info Starts ==\n";
  for (int i = 1; i <= edge_id_cnt; i++) {
    cout << E[i].nameA << '(' << E[i].idA << ')' << ' '
         << (E[i].is_directed ? "-->" : "---") << ' ' << E[i].nameB << '('
         << E[i].idB << ')' << ' ' << "type = " << E[i].type << ' '
         << "label = " << E[i].label << '\n';
  }
  cout << "== Edge Info Ends ==\n";
  */

  marked.assign(node_id_cnt + 1, false);
  for (int u = 1; u <= node_id_cnt; u++) {
    visited.assign(node_id_cnt + 1, false);
    for (auto &p : G[u]) {
      int v = p.to;
      int sz = is_chain_component(v, u);
      if (sz > 1) {
        mark_chain_nodes(v, u);
      }
    }
  }
  vector<int> chain_head;
  int marked_count = count(marked.begin(), marked.end(), true);
  for (int u = 1; u <= node_id_cnt; u++) {
    if (marked[u]) {
      int sm = 0;
      for (auto &p : G[u]) {
        int v = p.to;
        sm += marked[v];
      }
      if (sm == 1 && V[u].deg >= 3) {
        chain_head.push_back(u);
      }
    }
  }
  vector<int> chain_size;
  prt_head.assign(node_id_cnt + 1, 0);
  for (auto u : chain_head) {
    now_head = u;
    int sz = dfs_print(u, -1);
    chain_size.push_back(sz);
  }
  cout << '\n';
  cout << outName << '\n';
  cout << "total chain nodes = " << marked_count << '\n';
  cout << "total chains = " << chain_head.size() << '\n';
  for (auto u : chain_size) {
    cout << u << ' ';
  }
  cout << '\n';

  for (auto &line : ctrlLines) {
    if (line.substr(0, 6) == "wrdata" || line.substr(0, 6) == ".print") {
      string ans = "";
      istringstream subIss(line);
      string w, d;
      subIss >> w >> d;
      ans += w + " " + d;
      string tmp, nw;
      while (subIss >> tmp) {
        string name = "";
        for (int i = 2; i < tmp.length(); i++) {
          if (tmp[i] == ')')
            break;
          name += tmp[i];
        }
        int id = get_node_id(name);
        if (marked[id]) {
          nw = "V(" + get_node_name(prt_head[id]) + ")";
          ans += " " + nw;
        } else {
          ans += " " + tmp;
        }
      }
      line = ans;
    }
  }

  for (int i = 0; i < chain_head.size(); i++) {
    int u = chain_head[i];
    int sz = chain_size[i];
    marked[u] = 0;
    for (auto &p : G[u]) {
      int v = p.to, eid = p.edge_id;
      if (v == 1) {
        assert(capa == E[eid].label);
        assert(capa[capa.length() - 1] == 'f');
        double C = std::stod(capa.substr(0, capa.length() - 1)) * sz;
        E[eid].label = to_string(C) + 'f';
      }
    }
  }

  outFile
      << "* ISCAS85 benchmark circuit SPICE netlist \n* generated by "
         "spicegen.pl 1.0 \n* by Jingye Xu @VLSI group, Dept of ECE, UIC\n\n";
  for (auto l : dotLines) {
    outFile << l << '\n';
  }

  outFile << '\n';
  map<string, int> dev_id;
  for (auto &e : E) {
    int u = e.idA, v = e.idB;
    if (marked[u] || marked[v])
      continue;
    if (e.label == "null") {
      continue;
    }
    string t = e.type;
    assert(t == "v" || t == "r" || t == "c");
    transform(t.begin(), t.end(), t.begin(), ::toupper);
    int id = dev_id[t]++;
    outFile << t << id << ' ' << e.nameA << ' ' << e.nameB << ' ' << e.label
            << '\n';
  }
  outFile << '\n';

  for (const string &line : rawLines2) {
    if (line[0] == '.')
      continue;
    istringstream subIss(line);
    string w, d;
    subIss >> w;
    transform(w.begin(), w.end(), w.begin(), ::tolower);
    assert(w[0] == 'v' || w[0] == 'r' || w[0] == 'x');
    if (w[0] == 'x') {
      string a, token;

      vector<string> tokens;
      while (subIss >> token) {
        tokens.push_back(token);
      }
      assert(tokens.size() >= 2);

      string subcktName = tokens.back();
      assert(subckts.find(subcktName) != subckts.end());

      if (subckts[subcktName].is_device) {
        outFile << line << '\n';
        continue;
      }
    }
  }

  outFile << '\n';

  for (auto &p : subckts) {
    auto tmp = p.second;
    if (tmp.is_device) {
      outFile << "\n.subckt " << p.first << " ";
      for (auto &port : tmp.ports_name) {
        outFile << port << " ";
      }
      outFile << '\n';
      for (auto &line : tmp.content) {
        outFile << line << '\n';
      }
      outFile << ".ends\n";
    }
  }

  outFile << '\n';

  for (auto l : ctrlLines) {
    outFile << l << '\n';
  }
  outFile << "\n.end\n";
}
} // namespace findCutPoint

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

  // outFile.open("/dev/null");
  outName = argv[2];
  outFile.open(outName);
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
    if (line.substr(0, 8) == ".INCLUDE" || line.substr(0, 8) == ".OPTIONS") {
      dotLines.push_back(line);
      continue;
    }
    if (line.substr(0, 5) == ".tran" || line.substr(0, 6) == ".print") {
      ctrlLines.push_back(line);
      continue;
    }
    if (isInCtrl) {
      ctrlLines.push_back(line);
    }
    line = regex_replace(line, regex("^\\s+|\\s+$"), "");
    if (line.substr(0, 8) == ".control") {
      ctrlLines.push_back(line);
      isInCtrl = true;
    } else if (line.substr(0, 5) == ".endc") {
      isInCtrl = false;
    } else if (!isInCtrl && !line.empty()) {
      rawLines.push_back(line);
    }
  }
  infile.close();

  // generalGraph::main(rawLines);
  findCutPoint::main(rawLines);
  // globalMinCut::main(rawLines, split0);
  return 0;
}

