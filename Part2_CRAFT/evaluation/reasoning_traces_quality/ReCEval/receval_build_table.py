"""
Build the raw-CoT vs CRAFT-post-processed comparison table from
receval_evaluate_traces.py outputs.

Input: multiple score JSONs produced by receval_evaluate_traces.py.
       In each JSON, 'craft' = CRAFT post-processed, 'raw' = raw CoT
       (convention set by receval_adapter_craft.py).

Output: a markdown table (stdout) + LaTeX table (--latex_out).

Example:
    python receval_build_table.py \\
        --scores \\
            "FLD / GPT-5.4-nano:scores/fld_nano.json" \\
            "FLD / Gemini-3.1-flash-lite:scores/fld_gemini.json" \\
            "ProofWriter / GPT-5.4-nano:scores/proofwriter_nano.json" \\
            "ProofWriter / Gemini-3.1-flash-lite:scores/proofwriter_gemini.json" \\
        --metrics entail contradict \\
        --latex_out receval_table.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from config import resolve_input, resolve_output
except ImportError:
    resolve_input = resolve_output = Path


METRIC_PRETTY = {
    "entail":     "Entail$\\uparrow$",
    "contradict": "Coherence$\\uparrow$",   # 1 - max contradiction
    "pvi":        "PVI$\\uparrow$",
    "ll-info":    "Info-Gain$\\uparrow$",
}


def fmt(v, digits=3):
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def fmt_delta(v, digits=3):
    if v is None:
        return "—"
    sign = "$+$" if v >= 0 else "$-$"
    return f"{sign}{abs(v):.{digits}f}"


def load_score_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", nargs="+", required=True,
                    help='Row specs: "LABEL:PATH" — label shown in the table, path to ReCEval score JSON.')
    ap.add_argument("--metrics", nargs="+", default=["entail", "contradict"],
                    help="Metric keys to include (default: entail contradict)")
    ap.add_argument("--latex_out", default=None, help="Optional LaTeX file output")
    ap.add_argument("--digits", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for spec in args.scores:
        if ":" not in spec:
            raise ValueError(f"bad --scores entry (need LABEL:PATH): {spec}")
        label, path = spec.split(":", 1)
        data = load_score_file(path)
        wa = data["craft"]["aggregate"]
        bl = data["raw"]["aggregate"]
        n = data.get("config", {}).get("n_samples", len(data["craft"]["per_sample"]))
        row = {"label": label.strip(), "n": n}
        for m in args.metrics:
            row[f"raw__{m}"]   = bl.get(m)
            row[f"craft__{m}"] = wa.get(m)
            if bl.get(m) is not None and wa.get(m) is not None:
                row[f"delta__{m}"] = wa[m] - bl[m]
            else:
                row[f"delta__{m}"] = None
        rows.append(row)

    # ---- Markdown ----
    md_header = ["Dataset / Model", "N"]
    for m in args.metrics:
        pm = m.replace("-", "\\-")
        md_header += [f"Raw {pm}", f"CRAFT {pm}", f"Δ {pm}"]
    md_sep = ["---"] * len(md_header)
    md_lines = ["| " + " | ".join(md_header) + " |", "| " + " | ".join(md_sep) + " |"]
    for r in rows:
        cells = [r["label"], str(r["n"])]
        for m in args.metrics:
            cells += [
                fmt(r[f"raw__{m}"], args.digits),
                fmt(r[f"craft__{m}"], args.digits),
                fmt_delta(r[f"delta__{m}"], args.digits).replace("$+$", "+").replace("$-$", "-"),
            ]
        md_lines.append("| " + " | ".join(cells) + " |")

    print("\n".join(md_lines))

    # ---- LaTeX ----
    if args.latex_out:
        col_spec = "ll" + "ccc" * len(args.metrics)
        tex = []
        tex.append("\\begin{table}[t]")
        tex.append("  \\centering")
        tex.append("  \\resizebox{1.0\\linewidth}{!}{%")
        tex.append("  \\setlength{\\tabcolsep}{3pt}")
        tex.append("  \\renewcommand{\\arraystretch}{1.15}")
        tex.append("  {\\small")
        tex.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        tex.append("  \\toprule")
        # Header row 1
        h1 = ["\\textbf{Dataset / Model}", "\\textbf{N}"]
        for m in args.metrics:
            h1.append(f"\\multicolumn{{3}}{{c}}{{{METRIC_PRETTY.get(m, m)}}}")
        tex.append("  " + " & ".join(h1) + " \\\\")
        tex.append("  \\cmidrule(lr){3-" + str(2 + 3 * len(args.metrics)) + "}")
        # Header row 2
        h2 = ["", ""]
        for _ in args.metrics:
            h2 += ["Raw", "CRAFT", "$\\Delta$"]
        tex.append("  " + " & ".join(h2) + " \\\\")
        tex.append("  \\midrule")
        # Body
        for r in rows:
            cells = [r["label"].replace("/", "$\\slash$"), str(r["n"])]
            for m in args.metrics:
                cells += [
                    fmt(r[f"raw__{m}"], args.digits),
                    f"\\textbf{{{fmt(r[f'craft__{m}'], args.digits)}}}",
                    fmt_delta(r[f"delta__{m}"], args.digits),
                ]
            tex.append("  " + " & ".join(cells) + " \\\\")
        tex.append("  \\bottomrule")
        tex.append("  \\end{tabular}}")
        tex.append("  }")
        tex.append("  \\caption{ReCEval evaluation on paired samples: raw CoT (first generated trace) vs CRAFT post-processed. Δ is CRAFT − Raw; higher is better for both metrics.}")
        tex.append("  \\label{tab:receval_craft}")
        tex.append("\\end{table}")
        out = Path(args.latex_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(tex))
        print(f"\n[latex written] → {out}")


if __name__ == "__main__":
    main()
