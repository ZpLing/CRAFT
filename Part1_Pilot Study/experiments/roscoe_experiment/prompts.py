"""
prompts.py — System prompts and prompt builders for the pilot study's two settings.

Every prompt exists in a w/ Answer and a w/o Answer form; the pair differs only in
whether the correct answer or label is stated, so any difference in the traces is
attributable to that and nothing else. The ROSCOE datasets need three task shapes
(reading comprehension, NLI, math), each with its own pair.
"""
SYSTEM_WITH_ANSWER = (
    "You are an expert at logical reasoning. "
    "Given a set of premises and a known conclusion, write a clear numbered "
    "step-by-step reasoning chain that derives the conclusion from the premises. "
    "Each step must be a single sentence and logically follow from prior steps or premises."
)

SYSTEM_WOUT_ANSWER = (
    "You are an expert at logical reasoning. "
    "Given a set of premises and a hypothesis, reason step-by-step to decide "
    "whether the hypothesis is proved or disproved based solely on the premises. "
    "Each step must be a single sentence. "
    "End the last line with exactly one of: __PROVED__ or __DISPROVED__."
)


def build_prompt_with_answer(hypothesis: str, premises: list[str], proof_label: str) -> str:
    prem_block = "\n".join(f"- {p}" for p in premises)
    return (
        f"Premises:\n{prem_block}\n\n"
        f"Conclusion: {hypothesis}\n"
        f"(This conclusion is verified as: {proof_label})\n\n"
        "Write a numbered step-by-step reasoning chain that logically derives "
        "this conclusion from the premises."
    )


def build_prompt_wout_answer(hypothesis: str, premises: list[str]) -> str:
    prem_block = "\n".join(f"- {p}" for p in premises)
    return (
        f"Premises:\n{prem_block}\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Write a numbered step-by-step reasoning chain. "
        "On the very last line, write exactly one of: __PROVED__ or __DISPROVED__."
    )


# ---------------------------------------------------------------------------
# Prompts — ROSCOE datasets (domain-specific, NEW)
# ---------------------------------------------------------------------------

# DROP / CosmosQA: reading comprehension + commonsense
SYSTEM_RC_WITH_ANSWER = (
    "You are an expert at reading comprehension and reasoning. "
    "Given a passage and a question with a known correct answer, write a clear numbered "
    "step-by-step reasoning chain that derives the answer from the passage. "
    "Each step must be a single sentence."
)

SYSTEM_RC_WOUT_ANSWER = (
    "You are an expert at reading comprehension and reasoning. "
    "Given a passage and a question, reason step-by-step to find the answer. "
    "Each step must be a single sentence. "
    "End with a final sentence stating your answer."
)

# e-SNLI: natural language inference
SYSTEM_NLI_WITH_ANSWER = (
    "You are an expert at natural language inference. "
    "Given two sentences (premise and hypothesis) and the known relationship between them, "
    "write a clear numbered step-by-step explanation of why the relationship holds. "
    "Each step must be a single sentence."
)

SYSTEM_NLI_WOUT_ANSWER = (
    "You are an expert at natural language inference. "
    "Given two sentences (premise and hypothesis), reason step-by-step to determine "
    "whether the hypothesis is entailed, contradicted, or neutral with respect to the premise. "
    "Each step must be a single sentence. "
    "End with a final sentence stating your conclusion (entailment / contradiction / neutral)."
)

# GSM8K: math word problems
SYSTEM_MATH_WITH_ANSWER = (
    "You are an expert at solving math word problems. "
    "Given a math problem and its correct final answer, write a clear numbered "
    "step-by-step solution that shows how to reach the answer. "
    "Each step must be a single sentence showing one calculation or logical deduction."
)

SYSTEM_MATH_WOUT_ANSWER = (
    "You are an expert at solving math word problems. "
    "Given a math problem, solve it step by step. "
    "Each step must be a single sentence showing one calculation or logical deduction. "
    "End with a final sentence stating the numeric answer."
)


def _rc_with_answer_prompt(premise: str, hypothesis: str, answer: str) -> str:
    return (
        f"Passage:\n{premise}\n\n"
        f"Question: {hypothesis}\n"
        f"Correct answer: {answer}\n\n"
        "Write a numbered step-by-step reasoning chain that derives the answer from the passage."
    )


def _rc_wout_answer_prompt(premise: str, hypothesis: str) -> str:
    return (
        f"Passage:\n{premise}\n\n"
        f"Question: {hypothesis}\n\n"
        "Write a numbered step-by-step reasoning chain to answer the question."
    )


def _nli_with_answer_prompt(premise: str, hypothesis: str, answer: str) -> str:
    # answer in e-SNLI is 'yes'=entailment, 'no'=contradiction, 'maybe'=neutral
    label_map = {"yes": "entailment", "no": "contradiction", "maybe": "neutral"}
    label = label_map.get(answer.lower(), answer)
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Relationship: {label}\n\n"
        "Write a numbered step-by-step explanation of why this relationship holds."
    )


def _nli_wout_answer_prompt(premise: str, hypothesis: str) -> str:
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Write a numbered step-by-step reasoning chain to determine the relationship "
        "(entailment / contradiction / neutral) between premise and hypothesis."
    )


def _math_with_answer_prompt(premise: str, answer: str) -> str:
    return (
        f"Problem: {premise}\n"
        f"Answer: {answer}\n\n"
        "Write a numbered step-by-step solution showing how to reach this answer."
    )


def _math_wout_answer_prompt(premise: str) -> str:
    return (
        f"Problem: {premise}\n\n"
        "Write a numbered step-by-step solution."
    )

