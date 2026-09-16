"""
config.py — Part 1 Eval (Benchmark Evaluation)
Reads API credentials from environment variables.
Set OPENAI_API_KEY before running any evaluation script.
"""
import os
from pathlib import Path

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL",  "https://api.openai.com/v1")
DEFAULT_MODEL   = os.getenv("OPENAI_MODEL",     "gpt-4.1-mini")
REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))

# ── Output locations ─────────────────────────────────────────────────────────
# Every artifact this part produces belongs under
# <repo-root>/results/Part1_Correct_Answer_Guidance_Study/, so a rerun lands next
# to the existing score files instead of writing beside the code. Override the
# root with CRAFT_RESULTS_ROOT.
REPO_ROOT    = Path(__file__).resolve().parents[1]
RESULTS_ROOT = (Path(os.getenv("CRAFT_RESULTS_ROOT", REPO_ROOT / "results"))
                / "Part1_Correct_Answer_Guidance_Study")


def resolve_output(path) -> Path:
    """Map a relative --output onto RESULTS_ROOT and create its parent directory.

    Absolute paths pass through untouched, so an explicit destination always wins.
    """
    p = Path(path)
    if not p.is_absolute():
        p = RESULTS_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_input(path) -> Path:
    """Locate a relative --input, preferring the CWD and falling back to RESULTS_ROOT.

    Keeps paths like ../dataset/prmbench.jsonl working while letting a stage read
    the previous stage's output by bare name.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = RESULTS_ROOT / p
    return candidate if candidate.exists() else p
