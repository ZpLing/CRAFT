"""Domain optimization — the passes that run after Module III, per dataset.

These are not part of the three-module framework the paper describes. Module III
ends with a trace written over the consensus RKG; what is here reopens that trace
against the original problem when the dataset gives a specific reason to, and
each pass applies to the datasets whose structure it exploits:

    cwa_recheck.py        ProofWriter. The slice is closed-world at a known
                          depth, so __DISPROVED__ means no derivation was found,
                          which is only sound if the search was exhaustive. The
                          pass searches again in a directed way and accepts a
                          flip only against a chain the reply writes out.
    adjudicate_math.py    Omni-MATH, OlympiadBench. Two candidate answers that
                          a symbolic comparison cannot separate are put to the
                          model as a question about which one the problem asks
                          for.
    apply_adjudication.py Writes that decision back into the trace file.

They are kept apart from module3 because they do not read the graph: they go
back to the problem statement. Folding them into Module III would let the
topology-guided synthesis take credit for what they do, and on ProofWriter that
credit is most of the gain -- synthesis alone scores 58.4 there against the
filtered vote's 59.2, and these passes take it to 87.4.
"""
