import os
import re
from collections import defaultdict
import hashlib
import networkx as nx
import matplotlib.pyplot as plt


def draw_subckt_graph(name, subckt):
    G = nx.Graph()
    G.add_nodes_from(subckt['ports'])
    G.add_nodes_from(subckt['inners'])

    for a, neighbors in subckt['graph'].items():
        for b in neighbors:
            G.add_edge(a, b)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    # 画节点
    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=subckt['ports'],
                           node_color='skyblue',
                           node_size=600,
                           label='Ports')
    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=subckt['inners'],
                           node_color='lightgreen',
                           node_size=600,
                           label='Inners')

    # 画边和标签
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title(f"Subckt Graph: {name}")
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_netlist_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    subckts = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('.subckt'):
            tokens = line.split()
            name = tokens[1]
            ports = tokens[2:]

            body_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('.ends'):
                body_lines.append(lines[i].strip())
                i += 1

            body_str = '\n'.join(body_lines).strip()
            key = f'{name}_{hashlib.md5(body_str.encode()).hexdigest()}'

            # 检查是否存在相同 name 但内容不同的 subckt
            for existing_key in subckts:
                if existing_key.startswith(
                        name +
                        '_') and subckts[existing_key]['body'] != body_str:
                    raise ValueError(
                        f"Duplicate subckt name '{name}' with different body.")

            graph = defaultdict(set)
            all_nodes = set()

            mos_cnt = 0

            for body_line in body_lines:
                if body_line.startswith('m'):
                    mos_cnt += 1
                    tokens = body_line.split()
                    if len(tokens) >= 5:
                        a = tokens[1]
                        b = tokens[3]
                        graph[a].add(b)
                        all_nodes.update([a, b])

            inners = sorted(list(all_nodes - set(ports)))

            subckts[key] = {
                'name': name,
                'ports': ports,
                'inners': inners,
                'graph': dict(graph),
                'body': body_str,
                'mos_cnt': mos_cnt,
                'inner_cnt': len(inners)
            }
        i += 1

    return subckts


def main():
    filename = 'all_logic_dev.net'
    if not os.path.exists(filename):
        print(f"File '{filename}' not found in current directory.")
        return

    try:
        subckts = parse_netlist_file(filename)
        for k, info in subckts.items():
            print(f"\nSubckt: {info['name']}")
            print(f"  Ports: {info['ports']}")
            print(f"  Inners: {info['inners']}")
            print(f"  Graph edges:")
            inner = 0
            cnt = 0
            for node, neighbors in info['graph'].items():
                for neighbor in neighbors:
                    cnt += 1
                    print(f"    ({node}, {neighbor})")
        for k, info in subckts.items():
            print(
                f"{info['name']} & {info['mos_cnt']} & {info['inner_cnt']} \\\\"
            )
        for k, info in subckts.items():
            if info['name'] == 'and4':
                draw_subckt_graph('or4', info)
                break
        else:
            print("Subckt 'or4' not found.")
    except ValueError as e:
        print("Error:", e)


if __name__ == '__main__':
    main()
