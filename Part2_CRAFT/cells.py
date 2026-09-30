"""
cells.py — the settings of every dataset x backbone cell CRAFT reports.

A cell's number depends on more than the shared hyperparameters in config.py:
Module II weights ProofWriter's traces by proof depth, Module III follows or
verifies the consensus vote depending on the backbone, and ProofWriter and
gemini's mathematics get passes of their own after Module III. Those settings
used to be written down only as a description string beside the export, which
no runner read, so a rerun that did not know them silently left them out.

This file is the one place they are defined. run_cell.py builds every command
of a cell from it and takes no per-cell flag of its own, and the export's
setting string is Cell.describe(), so the table, the runner and the exported
traces cannot drift apart. K, T and the thresholds of Algorithm 1 are the same
for every cell and stay in config.py.

A post step's arguments are tokens with placeholders filled in by run_cell.py:
{model}, {dataset}, {k_traces}, and {<stage>} for the output file of an
earlier stage (e.g. {synthesized}, {cleaned}). Every stage writes <name>.json
in the run directory, so the last post step's name is the reported file's stem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

PART_ROOT = Path(__file__).resolve().parent
DOMAIN_OPT = "framework/domain_optimization"

DATASETS = ("FLD", "ProofWriter", "OmniMATH", "OlympiadBench")
MODELS = ("gpt-5.4-nano", "gemini-3.1-flash-lite")
DOMAINS = {"FLD": "logical", "ProofWriter": "logical",
           "OmniMATH": "math", "OlympiadBench": "math"}

# The file Module III writes. The evaluation scripts find a run by this name
# (synthesized*.json), so it is the reported file of a cell with no post pass.
SYNTH_STAGE = "synthesized"


@dataclass(frozen=True)
class PostStep:
    """One pass after Module III: a script under Part2_CRAFT and its arguments."""
    name: str                   # stage name; its output is <name>.json
    script: str                 # path relative to Part2_CRAFT
    args: Tuple[str, ...]       # tokens, with the placeholders described above
    label: Optional[str] = None  # how describe() names it; None leaves it out

    @property
    def script_path(self) -> Path:
        return PART_ROOT / self.script


@dataclass(frozen=True)
class Cell:
    dataset: str
    model: str
    domain: str
    prior_mode: str                              # Module III --prior_mode
    build_args: Tuple[str, ...] = ()             # extra Module II arguments
    post: Tuple[PostStep, ...] = field(default=())

    def __post_init__(self):
        # A typo here would otherwise surface only as a wrong number after the run.
        if DOMAINS.get(self.dataset) != self.domain:
            raise ValueError(f"{self.dataset} is {DOMAINS.get(self.dataset)!r}, not {self.domain!r}")
        if self.prior_mode not in ("verify", "follow"):
            raise ValueError(f"prior_mode must be verify or follow, got {self.prior_mode!r}")
        if len(self.build_args) % 2:
            raise ValueError(f"build_args must be --flag value pairs: {self.build_args}")

    @property
    def reported_stem(self) -> str:
        """Stem of the file this cell reports: the last stage's output."""
        return self.post[-1].name if self.post else SYNTH_STAGE

    def describe(self) -> str:
        """The setting string the export writes beside each reported trace."""
        parts = [f"prior_mode={self.prior_mode}"]
        toks = list(self.build_args)
        # Module II's arguments come in --flag value pairs.
        for flag, value in zip(toks[::2], toks[1::2]):
            parts.append(f"{flag.lstrip('-')}={value}")
        parts += [s.label for s in self.post if s.label]
        return ", ".join(parts)


# ── Post passes ─────────────────────────────────────────────────────────────
# ProofWriter's slice is closed-world at depth 5 (see cwa_recheck.py): first a
# directed proof search on the samples answered __DISPROVED__, its chain written
# back as steps, then the samples whose last step reasons from the absence of a
# derivation are searched in both directions.
_PW_DEPTH = "5"
_CWA = (
    PostStep("cwa", f"{DOMAIN_OPT}/cwa_recheck.py",
             ("--synth", "{synthesized}", "--k_traces", "{k_traces}",
              "--direction", "prove", "--expected_depth", _PW_DEPTH,
              "--model", "{model}"),
             label="cwa_recheck (integrated)"),
    PostStep("cwa_resolve", f"{DOMAIN_OPT}/cwa_recheck.py",
             ("--synth", "{cwa}", "--k_traces", "{k_traces}",
              "--direction", "resolve", "--expected_depth", _PW_DEPTH,
              "--model", "{model}"),
             label="cwa_resolve"),
)
# gpt-5.4-nano's ProofWriter trace is restated with its answer pinned.
_POLISH = (
    PostStep("polish", f"{DOMAIN_OPT}/polish_trace.py",
             ("--synth", "{cwa_resolve}", "--k_traces", "{k_traces}",
              "--dataset", "{dataset}", "--style", "two3", "--model", "{model}"),
             label="polish two3"),
)
# gemini's mathematics: the split votes are worked again, the derivation that
# settles one is written back as steps, and the trace opens with the goal.
_ADJUDICATE = (
    PostStep("adjudicated", f"{DOMAIN_OPT}/adjudicate_math.py",
             ("--k_traces", "{k_traces}", "--dataset", "{dataset}", "--model", "{model}")),
    PostStep("adj_applied", f"{DOMAIN_OPT}/apply_adjudication.py",
             ("--synth", "{synthesized}", "--adjudicated", "{adjudicated}",
              "--dataset", "{dataset}", "--model", "{model}"),
             label="adjudication (integrated)"),
    PostStep("adj_goal", f"{DOMAIN_OPT}/state_goal.py",
             ("--synth", "{adj_applied}", "--problems", "{cleaned}"),
             label="goal stated"),
)

# Module II on ProofWriter: every problem needs a five-step deduction, so the
# consensus favours traces whose length is near that depth.
_PW_BUILD = ("--weight_by", "gold_depth", "--expected_depth", _PW_DEPTH)

_GEM, _NANO = "gemini-3.1-flash-lite", "gpt-5.4-nano"

# prior_mode was chosen per configuration on a validation split: a backbone
# whose single re-derivation is weaker than its own vote does better following
# the vote ('follow') than checking it ('verify').
_ALL = (
    Cell("FLD", _GEM, "logical", "verify"),
    Cell("FLD", _NANO, "logical", "follow"),
    Cell("ProofWriter", _GEM, "logical", "verify", _PW_BUILD, _CWA),
    Cell("ProofWriter", _NANO, "logical", "verify", _PW_BUILD, _CWA + _POLISH),
    Cell("OmniMATH", _GEM, "math", "verify", (), _ADJUDICATE),
    Cell("OmniMATH", _NANO, "math", "follow"),
    Cell("OlympiadBench", _GEM, "math", "verify", (), _ADJUDICATE),
    Cell("OlympiadBench", _NANO, "math", "follow"),
)

CELLS: Dict[Tuple[str, str], Cell] = {(c.dataset, c.model): c for c in _ALL}


class UnknownCell(KeyError):
    """No cell is defined for a dataset x backbone pair."""
    def __str__(self) -> str:  # KeyError would print the message as a repr
        return str(self.args[0])


def get_cell(dataset: str, model: str) -> Cell:
    """The cell for one dataset and backbone; an unknown pair is an error, not a default."""
    try:
        return CELLS[(dataset, model)]
    except KeyError:
        known = "\n  ".join(f"{d} x {m}" for d, m in sorted(CELLS))
        raise UnknownCell(f"No cell for dataset={dataset!r}, model={model!r}. "
                       f"The cells are:\n  {known}") from None
