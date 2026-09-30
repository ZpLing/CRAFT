#!/usr/bin/env python3
"""
run_cell.py — run one dataset x backbone cell of CRAFT end to end.

    python run_cell.py --dataset FLD --model gpt-5.4-nano --run_dir craft_runs/fld_nano

Every setting that differs between cells comes from cells.py, and this script
takes no flag that could change one: a cell run by hand used to need its
Module II weighting, its Module III prior mode and its post passes typed out
again, and a rerun that omitted them still produced a number. Here the stages
are fixed by the cell:

    generate      Module I   K traces at T (config.K, config.T)
    cleaned_z     Module I   z-score steps filtering against T_Con
    rkg           Module II  consensus RKG, with the cell's build arguments
    cleaned       Pass 2     the same filter, now pruning against G*
    synthesized   Module III with the cell's --prior_mode
    <post steps>  the cell's own passes, in order (cells.py)
    score         evaluate_accuracy.py score on the reported file

Each stage is the existing script run as a subprocess, its output written to
<run_dir>/<stage>.json (generate writes k_traces_<n>_samples.json, the name
the evaluation scripts look for) and its console to <run_dir>/log_<stage>.txt.
A stage whose output exists is skipped, so an interrupted run picks up where it
stopped; once one stage runs, every later one runs too, since its input changed.
<run_dir>/manifest.json records the cell, the hyperparameters, the commit and
every stage, and a run directory made for one cell refuses to resume as another.

    --k_traces FILE   reuse Module I's rollouts instead of generating them
    --from STAGE      rerun from STAGE onward, ignoring existing outputs
    --no_resume       rerun every stage
    --dry_run         print the commands and run nothing
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PART_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PART_ROOT))
import config as _cfg  # noqa: E402
from cells import CELLS, SYNTH_STAGE, Cell, get_cell  # noqa: E402

M1 = "framework/module1_generation_filtering"
M2 = "framework/module2_consensus_rkg_construction"
M3 = "framework/module3_topology_guided_synthesis"
SCORE = "evaluation/label_prediction/evaluate_accuracy.py"


def hyperparameters() -> Dict[str, object]:
    """The Algorithm 1 values every stage defaults to, as this run resolves them."""
    return {name: getattr(_cfg, name) for name in
            ("K", "T", "ALPHA", "BETA", "GAMMA", "LAMBDA", "THETA", "ATOMIC_STEPS")}


def dataset_path(cell: Cell) -> Path:
    return _cfg.DATASET_ROOT / f"{cell.dataset}.json"


def k_traces_name(cell: Cell) -> str:
    """k_traces_<n>_samples.json, the name every downstream glob looks for."""
    try:
        n = len(json.loads(dataset_path(cell).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        n = 500
    return f"k_traces_{n}_samples.json"


def plan(cell: Cell, run_dir: Path, k_traces: Path,
         fresh: bool = False) -> List[Tuple[str, List[str], Path]]:
    """Every stage of the cell as (name, command, output), in order.

    fresh: Module III also ignores its own <output>.partial.jsonl checkpoint.
    """
    py = sys.executable
    out = {"k_traces": k_traces}

    def o(name: str) -> str:
        out.setdefault(name, run_dir / f"{name}.json")
        return str(out[name])

    d, m = cell.domain, cell.model
    stages = [
        ("generate", [py, f"{M1}/generate_traces.py", "--datasets", str(dataset_path(cell)),
                      "--K", str(_cfg.K), "--T", str(_cfg.T), "--domain", d, "--model", m,
                      "--output", str(k_traces)]),
        ("cleaned_z", [py, f"{M1}/steps_filter.py", "--input", str(k_traces),
                       "--method", "unsupervised", "--domain", d, "--output", o("cleaned_z")]),
        ("rkg", [py, f"{M2}/build_rkg.py", "--input", o("cleaned_z"), "--domain", d,
                 "--model", m, *cell.build_args, "--output", o("rkg")]),
        ("cleaned", [py, f"{M1}/steps_filter.py", "--input", o("cleaned_z"),
                     "--method", "rkg", "--rkg_file", o("rkg"), "--domain", d,
                     "--output", o("cleaned")]),
        (SYNTH_STAGE, [py, f"{M3}/synthesize_trace.py", "--input", o("cleaned"),
                       "--rkg_file", o("rkg"), "--original_file", str(k_traces),
                       "--domain", d, "--model", m, "--prior_mode", cell.prior_mode,
                       "--output", o(SYNTH_STAGE), *(["--no_resume"] if fresh else [])]),
    ]
    fill = {"model": m, "dataset": cell.dataset}
    for step in cell.post:
        # Only stages that ran before this one can be referenced, so an
        # out-of-order placeholder is a KeyError here rather than a missing file later.
        args = [a.format(**fill, **{k: str(v) for k, v in out.items()}) for a in step.args]
        stages.append((step.name, [py, step.script, *args, "--output", o(step.name)]))
    stages.append(("score", [py, SCORE, "score", "--input", o(cell.reported_stem),
                             "--source", "synthesized", "--output", o("score")]))
    return [(name, cmd, out["k_traces"] if name == "generate" else out[name])
            for name, cmd in stages]


def git_commit() -> Dict[str, object]:
    def git(*a):
        r = subprocess.run(["git", "-C", str(PART_ROOT), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def check_k_traces(path: Path, cell: Cell) -> None:
    """Refuse rollouts that belong to another backbone or dataset."""
    meta = json.loads(path.read_text(encoding="utf-8"))
    rows = meta.get("results", []) if isinstance(meta, dict) else meta
    model = (meta.get("metadata") or {}).get("model") if isinstance(meta, dict) else None
    if model and model != cell.model:
        raise SystemExit(f"{path} was generated by {model}, not {cell.model}")
    names = {str(r.get("source_dataset", "")) for r in rows[:50]}
    if names and not any(n.startswith(cell.dataset + ".") or n == cell.dataset for n in names):
        raise SystemExit(f"{path} holds {sorted(names)}, not {cell.dataset}")


def run_stage(cmd: List[str], log: Path) -> int:
    """Run one stage, its output shown and written to its log as it arrives."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log.open("w", encoding="utf-8") as fh:
        fh.write("$ " + shlex.join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=PART_ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted({d for d, _ in CELLS}))
    ap.add_argument("--model", required=True, choices=sorted({m for _, m in CELLS}))
    ap.add_argument("--run_dir", required=True,
                    help="Directory for every stage's output; relative paths resolve under "
                         "the results root, as the stages' own --output does")
    ap.add_argument("--k_traces", default=None,
                    help="An existing Module I rollout file to reuse; generate is not run")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help="Rerun from this stage onward (see the stage list above)")
    ap.add_argument("--no_resume", action="store_true",
                    help="Rerun every stage even where its output exists")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the commands without running them")
    args = ap.parse_args()

    cell = get_cell(args.dataset, args.model)
    # Absolute, because each stage's resolve_output puts a relative --output under
    # the results root and resolve_input looks in the CWD first: the two would not
    # agree on where a relative path is.
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _cfg.RESULTS_ROOT / run_dir
    run_dir = run_dir.resolve()

    reused = args.k_traces is not None
    if reused:
        k_traces = Path(_cfg.resolve_input(args.k_traces)).resolve()
        if not k_traces.exists():
            raise SystemExit(f"--k_traces {k_traces} does not exist")
        check_k_traces(k_traces, cell)
    else:
        k_traces = run_dir / k_traces_name(cell)

    stages = plan(cell, run_dir, k_traces, fresh=args.no_resume)
    names = [n for n, _, _ in stages]
    if args.from_stage is not None and args.from_stage not in names:
        raise SystemExit(f"--from {args.from_stage!r}: this cell's stages are {', '.join(names)}")
    if reused and args.from_stage == "generate":
        raise SystemExit("--from generate would regenerate the rollouts --k_traces reuses")
    first_forced = names.index(args.from_stage) if args.from_stage else len(names)
    if args.no_resume:
        first_forced = 0

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "cell": {**dataclasses.asdict(cell), "setting": cell.describe(),
                 "reported_stem": cell.reported_stem},
        "hyperparameters": hyperparameters(),
        "k_traces": str(k_traces),
        "k_traces_reused": reused,
        "stages": {},
    }
    if manifest_path.exists() and not args.no_resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Outputs made under other settings are not this cell's, and resuming
        # on top of them is the silent mix this runner exists to prevent.
        for key in ("cell", "hyperparameters"):
            if old.get(key) != json.loads(json.dumps(manifest[key])):
                raise SystemExit(f"{run_dir} was run with a different {key}:\n"
                                 f"  was {old.get(key)}\n  now {manifest[key]}\n"
                                 "Use a fresh --run_dir, or --no_resume to redo every stage.")
        manifest["stages"] = old.get("stages", {})

    print(f"Cell: {cell.dataset} x {cell.model}  [{cell.describe()}]")
    print(f"Run directory: {run_dir}")
    if args.dry_run:
        # Every stage runs from Part2_CRAFT, as the scripts' relative imports expect.
        print(f"cd {shlex.quote(str(PART_ROOT))}")
        for i, (name, cmd, output) in enumerate(stages):
            if name == "generate" and reused:
                print(f"# {name}: reusing {k_traces}")
                continue
            skip = i < first_forced and output.exists()
            print(f"# {name}" + ("  (output exists, would be skipped)" if skip else ""))
            print(shlex.join(cmd))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.update(git_commit())
    if reused:
        # The evaluation scripts find a run's rollouts by globbing its directory.
        link = run_dir / k_traces_name(cell)
        if not link.exists():
            link.symlink_to(k_traces)
        elif link.resolve() != k_traces:
            raise SystemExit(f"{link} already exists and is not {k_traces}")

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ran_one = False
    for i, (name, cmd, output) in enumerate(stages):
        entry = {"command": cmd, "output": str(output), "log": str(run_dir / f"log_{name}.txt")}
        if name == "generate" and reused:
            manifest["stages"][name] = {**entry, "status": "reused", "log": None}
            continue
        if not ran_one and i < first_forced and output.exists():
            print(f"[{name}] skipped: {output.name} exists")
            manifest["stages"].setdefault(name, {**entry, "status": "skipped (output exists)"})
            continue
        print(f"[{name}] {shlex.join(cmd)}")
        t0 = time.time()
        code = run_stage(cmd, Path(entry["log"]))
        entry["seconds"] = round(time.time() - t0, 1)
        if code != 0:
            # A stage that failed after writing its output would otherwise be
            # taken as finished by the next resume.
            if output.exists():
                output.rename(output.with_name(output.name + ".failed"))
            manifest["stages"][name] = {**entry, "status": f"failed (exit {code})"}
            save()
            raise SystemExit(f"[{name}] failed with exit code {code}; see {entry['log']}")
        manifest["stages"][name] = {**entry, "status": "ran"}
        ran_one = True
        save()

    final = run_dir / f"{cell.reported_stem}.json"
    manifest["final_file"] = str(final)
    manifest["score_file"] = str(run_dir / "score.json")
    try:
        score = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))
        manifest["accuracy"] = score["overall"]["accuracy"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    save()
    print(f"Done. Reported file: {final}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
