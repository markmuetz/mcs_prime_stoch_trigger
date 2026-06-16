"""Dump the remake2 rule-level DAG for a remakefile.

Run from within ctrl/remakefiles/ (load_remake uses cwd):
    python dump_dag.py <remakefile.py>
"""
import sys

import networkx as nx

from remake.loader import load_remake


def main(remakefile):
    rmk = load_remake(remakefile, finalize=False)
    g = rmk.rule_dg
    rules = sorted(n.__name__ for n in g.nodes)
    print(f'# {remakefile}: {len(rules)} rules, {g.number_of_edges()} rule-edges')
    print('## nodes')
    for r in rules:
        print(f'  {r}')
    print('## edges (upstream -> downstream)')
    for u, v in sorted((a.__name__, b.__name__) for a, b in g.edges):
        print(f'  {u} -> {v}')
    print('## topological order')
    try:
        topo = [n.__name__ for n in nx.topological_sort(g)]
        print('  ' + ' -> '.join(topo))
    except nx.NetworkXUnfeasible:
        print('  (rule_dg has a cycle - not a DAG)')


if __name__ == '__main__':
    main(sys.argv[1])
