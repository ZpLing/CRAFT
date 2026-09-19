"""CRAFT — the three modules of §3.2, one package per module of the paper.

    Module I   Multi-Trace Generation & Steps Filtering   module1_generation_filtering/
      generate_traces.py    roll out K candidate traces at temperature T
      extract_terms.py      TF-IRF consensus terms T_Con (alpha, beta)
      anomaly_filter.py     z-score steps filtering (gamma); --method rkg re-runs it
                            against G* once Module II has built the graph

    Module II  Consensus RKG Construction                 module2_rkg_construction/
      build_rkg.py          per-trace RKG, edge weight W(e) (lambda), edge and node
                            filtering (theta), aggregation into the consensus RKG G*

    Module III Topology-guided Trace Synthesis            module3_synthesis/
      synthesize_trace.py   topological walk over G*, one step generated per node

Run order is Module I generation -> Module I filtering -> Module II -> Module I's RKG
pass -> Module III; the exact commands and the paper's hyperparameter values are in the
repo README's Quick start.
"""
