"""
prompts.py — Prompts for the pilot study's two settings.

The study asks what changes when the model is told the answer, so the two settings
must differ by that and nothing else. Each task therefore has ONE system prompt,
shared by both settings, and one user-prompt builder whose only variable part is a
single line stating the answer. The w/o Answer form is the task as its benchmark
poses it — solve the problem and commit to a conclusion — and the w/ Answer form is
that same task with the answer supplied.

Four task shapes are needed: Entailment Bank's logical proofs, and the three that
the ROSCOE datasets fall into (reading comprehension for DROP and CosmosQA, NLI for
e-SNLI, math for GSM8K).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Entailment Bank — logical proof
# ---------------------------------------------------------------------------
SYSTEM_LOGIC = (
    "You are an expert at logical reasoning. "
    "Given a set of premises and a hypothesis, reason step-by-step to decide "
    "whether the hypothesis is proved or disproved based solely on the premises. "
    "Each step must be a single sentence. "
    "End the last line with exactly one of: __PROVED__ or __DISPROVED__."
)


def _logic_prompt(hypothesis: str, premises: list[str], proof_label: str | None = None) -> str:
    prem_block = "\n".join(f"- {p}" for p in premises)
    answer_line = f"Verified label: {proof_label}\n" if proof_label is not None else ""
    return (
        f"Premises:\n{prem_block}\n\n"
        f"Hypothesis: {hypothesis}\n"
        f"{answer_line}\n"
        "Write a numbered step-by-step reasoning chain. "
        "On the very last line, write exactly one of: __PROVED__ or __DISPROVED__."
    )


def build_prompt_with_answer(hypothesis: str, premises: list[str], proof_label: str) -> str:
    return _logic_prompt(hypothesis, premises, proof_label)


def build_prompt_wout_answer(hypothesis: str, premises: list[str]) -> str:
    return _logic_prompt(hypothesis, premises)


# ---------------------------------------------------------------------------
# DROP / CosmosQA — reading comprehension
# ---------------------------------------------------------------------------
SYSTEM_RC = (
    "You are an expert at reading comprehension and reasoning. "
    "Given a passage and a question, reason step-by-step to find the answer. "
    "Each step must be a single sentence. "
    "End with a final sentence stating the answer."
)


def _rc_prompt(passage_and_question: str, answer: str | None = None) -> str:
    """ROSCOE ships DROP and CosmosQA with the question already appended to the
    passage, and keeps the correct answer in the `hypothesis` field — the data is
    built for verification, not for asking. So the whole `premise` is the prompt's
    context, and the answer, when given, is that `hypothesis`."""
    answer_line = f"Correct answer: {answer}\n" if answer is not None else ""
    return (
        f"Passage and question:\n{passage_and_question}\n"
        f"{answer_line}\n"
        "Write a numbered step-by-step reasoning chain to answer the question. "
        "End with a final sentence stating the answer."
    )


def _rc_with_answer_prompt(passage_and_question: str, answer: str) -> str:
    return _rc_prompt(passage_and_question, answer)


def _rc_wout_answer_prompt(passage_and_question: str) -> str:
    return _rc_prompt(passage_and_question)


# ---------------------------------------------------------------------------
# e-SNLI — natural language inference
# ---------------------------------------------------------------------------
SYSTEM_NLI = (
    "You are an expert at natural language inference. "
    "Given two sentences (premise and hypothesis), reason step-by-step to determine "
    "whether the hypothesis is entailed, contradicted, or neutral with respect to "
    "the premise. Each step must be a single sentence. "
    "End with a final sentence stating your conclusion "
    "(entailment / contradiction / neutral)."
)

# e-SNLI ships the label as yes/no/maybe; the prompt names the relation instead.
_NLI_LABELS = {"yes": "entailment", "no": "contradiction", "maybe": "neutral"}


def _nli_prompt(premise: str, hypothesis: str, answer: str | None = None) -> str:
    answer_line = ""
    if answer is not None:
        answer_line = f"Correct answer: {_NLI_LABELS.get(answer.lower(), answer)}\n"
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"{answer_line}\n"
        "Write a numbered step-by-step reasoning chain to determine the relationship "
        "(entailment / contradiction / neutral) between premise and hypothesis. "
        "End with a final sentence stating your conclusion."
    )


def _nli_with_answer_prompt(premise: str, hypothesis: str, answer: str) -> str:
    return _nli_prompt(premise, hypothesis, answer)


def _nli_wout_answer_prompt(premise: str, hypothesis: str) -> str:
    return _nli_prompt(premise, hypothesis)


# ---------------------------------------------------------------------------
# GSM8K — math word problems
# ---------------------------------------------------------------------------
SYSTEM_MATH = (
    "You are an expert at solving math word problems. "
    "Given a math problem, solve it step by step. "
    "Each step must be a single sentence showing one calculation or logical deduction. "
    "End with a final sentence stating the numeric answer."
)


def _math_prompt(premise: str, answer: str | None = None) -> str:
    """ROSCOE's GSM8K `answer` field records whether the GPT-3 solution it shipped
    was correct, not what the problem's answer is; the answer lives at the end of
    the reference solution in `hypothesis`. Callers pass that, never `answer`."""
    answer_line = f"Correct answer: {answer}\n" if answer is not None else ""
    return (
        f"Problem: {premise}\n"
        f"{answer_line}\n"
        "Write a numbered step-by-step solution. "
        "End with a final sentence stating the numeric answer."
    )


def _math_with_answer_prompt(premise: str, answer: str) -> str:
    return _math_prompt(premise, answer)


def _math_wout_answer_prompt(premise: str) -> str:
    return _math_prompt(premise)


# The settings share one system prompt per task; these names keep the call sites
# in generate_traces.py reading as the pair they send.
SYSTEM_WITH_ANSWER = SYSTEM_WOUT_ANSWER = SYSTEM_LOGIC
SYSTEM_RC_WITH_ANSWER = SYSTEM_RC_WOUT_ANSWER = SYSTEM_RC
SYSTEM_NLI_WITH_ANSWER = SYSTEM_NLI_WOUT_ANSWER = SYSTEM_NLI
SYSTEM_MATH_WITH_ANSWER = SYSTEM_MATH_WOUT_ANSWER = SYSTEM_MATH
