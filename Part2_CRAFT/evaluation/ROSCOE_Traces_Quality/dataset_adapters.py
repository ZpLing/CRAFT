#!/usr/bin/env python3
"""One adapter per benchmark dataset, shared by both trace-quality evaluations.

The four datasets do not agree on how a problem is stored. FLD keeps its facts in
one string and ProofWriter keeps them in a list; the logical two name the answer
`proof_label` and the mathematical two name it `answer`; only the logical two
record how long the gold proof is, and they record it two different ways. ROSCOE
and the SFT evaluation each need the same few fields out of whichever dataset a run was
generated from, so the per-dataset knowledge lives here once rather than as a
fall-through guess repeated in each adapter — which is what let a ProofWriter run
hand the scorer a list where it expected a string, and a maths run hand it an
empty hypothesis.

The shape every adapter returns:

    premises         the context the trace reasons over, always a string
    hypothesis       the claim the trace argues for — the conclusion for the
                     logical sets, the gold answer for the mathematical ones
    answer           the gold answer or label
    reference_steps  the gold proof's length, or None where the dataset has no
                     such thing, which is both mathematical sets
    domain           "logical" or "math" — what a step in this dataset looks
                     like, which is what decides how a trace is read and how a
                     judge is asked about it

`hypothesis` and `answer` are reference fields for the scorers; Module I's loader
is what keeps them out of generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class Problem:
    """One sample, in the shape the downstream evaluations read."""

    premises: str
    hypothesis: str
    answer: str
    reference_steps: Optional[int] = None
    dataset: str = ""
    domain: str = ""


def as_text(value: Any) -> str:
    """Flatten a field that one dataset stores as a string and another as a list."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(as_text(v) for v in value if v is not None).strip()
    return str(value).strip()


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def fld_proof_steps(proofs: Any) -> Optional[int]:
    """FLD's gold proof is one string: 'sent2 -> int1: …; int1 & sent4 -> hypothesis;'.

    Its length is the number of derivations in it. The published selection runs
    1–7 steps.
    """
    text = as_text(proofs)
    if not text:
        return None
    return sum(1 for segment in text.split(";") if "->" in segment) or None


# ---------------------------------------------------------------------------
# One adapter per dataset
# ---------------------------------------------------------------------------

def adapt_fld(sample: Dict[str, Any]) -> Problem:
    """FLD — facts as one string, conclusion as one string, proof as one string."""
    return Problem(
        premises=as_text(sample.get("Facts")) or as_text(sample.get("input")),
        hypothesis=as_text(sample.get("Conclusion")),
        answer=as_text(sample.get("proof_label")),
        reference_steps=fld_proof_steps(sample.get("proofs")),
        dataset="FLD",
        domain="logical",
    )


def adapt_proofwriter(sample: Dict[str, Any]) -> Problem:
    """ProofWriter — facts as a list, and QDep is the depth of the gold deduction."""
    return Problem(
        premises=as_text(sample.get("Facts")) or as_text(sample.get("input")),
        hypothesis=as_text(sample.get("Conclusion")),
        answer=as_text(sample.get("proof_label")),
        reference_steps=_int_or_none(sample.get("QDep")),
        dataset="ProofWriter",
        domain="logical",
    )


def adapt_olympiadbench(sample: Dict[str, Any]) -> Problem:
    """OlympiadBench — the problem statement is the context, the gold answer the claim.

    A competition problem has a reference solution but no step count anyone
    annotated, so reference_steps stays None rather than being guessed from the
    solution's punctuation.
    """
    return Problem(
        premises=as_text(sample.get("input")),
        hypothesis=as_text(sample.get("answer")),
        answer=as_text(sample.get("answer")),
        reference_steps=None,
        dataset="OlympiadBench",
        domain="math",
    )


def adapt_omnimath(sample: Dict[str, Any]) -> Problem:
    """Omni-MATH — same shape as OlympiadBench, without the answer_type tag."""
    return Problem(
        premises=as_text(sample.get("input")),
        hypothesis=as_text(sample.get("answer")),
        answer=as_text(sample.get("answer")),
        reference_steps=None,
        dataset="OmniMATH",
        domain="math",
    )


# The proof depth a dataset is drawn at, where that is a property of the
# selection rather than of the sample. ProofWriter here is the depth-5 slice, so
# every one of its 500 problems needs a five-step deduction and the number is
# public — the paper states it. Knowing it is knowing the configuration, not the
# answer, so a consensus may weight by how close a trace comes to it.
#
# FLD is deliberately absent. Its proof length varies per sample, so a depth
# would be gold annotation about that problem, and a run has no business seeing
# it. The mathematical sets annotate no depth at all.
#
# Measured on the 500-sample ProofWriter run: a trace landing within two steps
# of the depth is right 90% of the time, one running six steps over is right 50%
# — the accuracy of a coin on a two-way label.
EXPECTED_DEPTH: Dict[str, Optional[int]] = {
    "ProofWriter": 5,
}


def expected_depth(dataset: Any) -> Optional[int]:
    """The depth this dataset is drawn at, or None when it is not a constant."""
    return EXPECTED_DEPTH.get(dataset_name(dataset))


# What a step looks like in each set, which the downstream scorers need before
# they can read one: the logical sets state their premises as "Fact1: ..." and
# close with a __PROVED__/__DISPROVED__ marker, the mathematical ones state a
# problem in prose and close with \boxed{}.
DOMAIN: Dict[str, str] = {
    "FLD": "logical",
    "ProofWriter": "logical",
    "OlympiadBench": "math",
    "OmniMATH": "math",
}

ADAPTERS: Dict[str, Callable[[Dict[str, Any]], Problem]] = {
    "FLD": adapt_fld,
    "ProofWriter": adapt_proofwriter,
    "OlympiadBench": adapt_olympiadbench,
    "OmniMATH": adapt_omnimath,
}

# Spellings a run or a path can carry for each dataset, lowercased.
_ALIASES = {
    "fld": "FLD",
    "proofwriter": "ProofWriter",
    "proof_writer": "ProofWriter",
    "olympiadbench": "OlympiadBench",
    "olympiad": "OlympiadBench",
    "omnimath": "OmniMATH",
    "omni_math": "OmniMATH",
    "omni-math": "OmniMATH",
}


def dataset_name(raw: Any) -> str:
    """Canonical dataset name from whatever a run or a path spells it as.

    Runs record `source_dataset` as the file name ('FLD.json'), and a --dataset
    path is that same file, so both resolve through the stem.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    stem = Path(text).stem.strip()
    return _ALIASES.get(stem.lower(), "")


def domain_of(dataset: Any) -> str:
    """"logical" or "math" for a dataset, or "" when the name is not one of ours."""
    return DOMAIN.get(dataset_name(dataset), "")


def adapt(sample: Dict[str, Any], dataset: Any = None) -> Problem:
    """Normalise one sample, dispatching on the dataset it came from.

    Without a recognised name the shape decides, so a dataset added later still
    yields something usable instead of an empty record: a Conclusion beside Facts
    is read as a logical set, an `answer` beside an `input` as a mathematical one.
    """
    name = dataset_name(dataset)
    if name:
        return ADAPTERS[name](sample)

    if "Conclusion" in sample and "Facts" in sample:
        problem = adapt_fld(sample)
        problem.dataset = ""
        if problem.reference_steps is None:
            problem.reference_steps = _int_or_none(sample.get("QDep"))
        return problem
    if "answer" in sample and "input" in sample:
        problem = adapt_omnimath(sample)
        problem.dataset = ""
        return problem
    return Problem(
        premises=as_text(sample.get("input")),
        hypothesis=as_text(sample.get("Conclusion")) or as_text(sample.get("answer")),
        answer=as_text(sample.get("proof_label")) or as_text(sample.get("answer")),
        reference_steps=None,
    )
