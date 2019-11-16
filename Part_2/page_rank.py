#!/usr/bin/env python3

from random import choice
import networkx as nx

graph_v1 = nx.DiGraph()
graph_v2 = nx.DiGraph()

with open('as-caida20071105.txt') as f:
    for line in f.readlines():
        if line.startswith('#'):
            print(line.strip())
            continue

        source, target, relation = [int(x) for x in line.strip().split('\t')]
        graph_v1.add_edge(source, target)

        if relation == 1:
            graph_v2.add_edge(source, target)
        elif relation == -1:
            graph_v2.add_edge(target, source)
        else:
            graph_v2.add_edge(source, target) if choice([True, False]) else graph_v2.add_edge(target, source)

print()
for i, graph in enumerate([graph_v1, graph_v2]):
    print('Graph v{}:'.format(i+1))
    print('Number of edges:', graph.number_of_edges())
    print('Number of nodes:', graph.number_of_nodes())
    pr = nx.pagerank(graph, alpha=0.9)
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)

    print('Top 10 ranked nodes:')
    for node, rank in sorted_pr[:10]:
        print('\tNode: {}\t\tRank: {:.5f}'.format(node, rank))
    print()
    print('Bottom 10 ranked nodes:')
    for node, rank in sorted_pr[-10:]:
        print('\tNode: {}\t\tRank: {:.5e}'.format(node, rank))
    print('-' * 50)
