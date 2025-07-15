import re
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

TARGET_FUNCTIONS = {
    "CreateFillin", "SearchForPivot", "QuicklySearchDiagonal",
    "ExchangeRowsAndCols", "UpdateMarkowitzNumbers", "SearchForSingleton",
    "FindBiggestInColExclude", "ExchangeRowElements",
    "ComplexRowColElimination", "RealRowColElimination", "spOrderAndFactor",
    "MarkowitzProducts", "SearchDiagonal", "SearchEntireMatrix",
    "ExchangeColElements", "FactorComplexMatrix", "CountMarkowitz",
    "FindLargestInCol", "spFactor", "spcColExchange", "spcRowExchange"
}


def extract_functions_and_calls(code):

    functions = {}
    for func in TARGET_FUNCTIONS:
        functions[func] = set()
    current_function = None
    brace_count = 0

    for line in code.splitlines():
        function_match = None
        for function in TARGET_FUNCTIONS:
            if function in line:
                function_match = function
        if current_function:
            brace_count += line.count("{") - line.count("}") + line.count(
                "(") - line.count(")")
            if brace_count <= 0:
                current_function = None
                continue
            for call in TARGET_FUNCTIONS:
                if call in line:
                    if current_function != call:
                        functions[current_function].add(call)
                        print("find calls: ", line)
        elif function_match:
            current_function = function_match
            brace_count = line.count("{") + line.count("(") - line.count(")")
            if brace_count <= 0:
                current_function = None
            else:
                print("find function: ", line)

    print(functions)
    functions["QuicklySearchDiagonal"].discard("SearchDiagonal")
    return functions


def build_call_graph(functions):
    graph = nx.DiGraph()
    for func, calls in functions.items():
        graph.add_node(func)
        for call in calls:
            graph.add_edge(func, call)
    return graph


def plot_call_graph(graph):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph)
    pos["spFactor"] = (0, 3)
    pos["spOrderAndFactor"] = (1, 2)
    pos["ZeroPivot"] = (-2, 2)
    pos["FactorComplexMatrix"] = (-1, 2)
    pos["MarkowitzProducts"] = (-2, 1)
    pos["CountMarkowitz"] = (-1, 1)
    pos["UpdateMarkowitzNumbers"] = (0, 1)
    pos["SearchForPivot"] = (2, 1)
    pos["ExchangeRowsAndCols"] = (1, 1)
    pos["FindLargestInCol"] = (2, -1)
    pos["ComplexRowColElimination"] = (-3, 1)
    pos["RealRowColElimination"] = (-4, 1)
    pos["CreateFillin"] = (-5, 0)
    pos["SearchEntireMatrix"] = (2, 0)
    pos["spcColExchange"] = (-4, 0)
    pos["spcRowExchange"] = (-3, 0)
    pos["ExchangeColElements"] = (-3, -1)
    pos["ExchangeRowElements"] = (-4, -1)
    pos["QuicklySearchDiagonal"] = (-2, 0)
    pos["SearchDiagonal"] = (-1, 0)
    pos["SearchForSingleton"] = (0, 0)
    pos["FindBiggestInColExclude"] = (-1, -1)

    nx.draw(graph,
            pos,
            with_labels=True,
            node_color='lightblue',
            edge_color='gray',
            node_size=4000,
            font_size=14,
            font_weight='bold',
            width=2)
    plt.show()


def main(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    functions = extract_functions_and_calls(code)
    graph = build_call_graph(functions)
    plot_call_graph(graph)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 c_function_call_tree.py <source.c>")
        sys.exit(1)
    main(sys.argv[1])
