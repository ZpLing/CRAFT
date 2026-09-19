"""
config.py — Part 2 CRAFT shim.
Loads API credentials and model defaults from the repo-root config.py (gitignored).
All values can be overridden at runtime via --api_key / --base_url / --model CLI args
or the OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL env vars.
"""
import importlib.util
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "config.py"
if _ROOT.exists():
    _spec = importlib.util.spec_from_file_location("_root_config", _ROOT)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    OPENAI_API_KEY    = getattr(_m, "OPENAI_API_KEY",    os.getenv("OPENAI_API_KEY", ""))
    OPENAI_BASE_URL   = getattr(_m, "OPENAI_BASE_URL",   os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    MODEL_TRACE_GEN   = getattr(_m, "MODEL_TRACE_GEN",   os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    MODEL_RKG_BUILD   = getattr(_m, "MODEL_RKG_BUILD",   MODEL_TRACE_GEN)
    MODEL_SYNTHESIS   = getattr(_m, "MODEL_SYNTHESIS",   MODEL_TRACE_GEN)
else:
    OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_TRACE_GEN   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    MODEL_RKG_BUILD   = MODEL_TRACE_GEN
    MODEL_SYNTHESIS   = MODEL_TRACE_GEN

DEFAULT_MODEL   = MODEL_SYNTHESIS
REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))

# ── Pinned endpoint ──────────────────────────────────────────────────────────
# Scripts that must reach one specific endpoint read these instead of
# OPENAI_API_KEY / OPENAI_BASE_URL, which a stray env var can redirect.
# Defined only in the gitignored repo-root config.py — never hardcode them here.
PINNED_API_KEY  = getattr(_m, "PINNED_API_KEY",  None) if _ROOT.exists() else None
PINNED_BASE_URL = getattr(_m, "PINNED_BASE_URL", None) if _ROOT.exists() else None


def require_pinned_endpoint() -> tuple:
    """Return (api_key, base_url) for the pinned endpoint, or explain what's missing."""
    if not PINNED_API_KEY or not PINNED_BASE_URL:
        raise RuntimeError(
            "PINNED_API_KEY / PINNED_BASE_URL are not set. Define them in the repo-root "
            "config.py (gitignored) — they are deliberately absent from tracked files."
        )
    return PINNED_API_KEY, PINNED_BASE_URL

# ── Part-local locations ─────────────────────────────────────────────────────
# This part is self-contained: its datasets live in Part2_CRAFT/dataset/ and every
# artifact it produces lands in Part2_CRAFT/results/, so a rerun sits next to the
# existing run directories instead of scattering JSON files into the code tree.
# Override the results root with CRAFT_RESULTS_ROOT.
PART_ROOT    = Path(__file__).resolve().parent
REPO_ROOT    = PART_ROOT.parent
DATASET_ROOT = PART_ROOT / "dataset"
RESULTS_ROOT = Path(os.getenv("CRAFT_RESULTS_ROOT", PART_ROOT / "results"))


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

    Keeps paths like dataset/logical/FLD.json working while letting a stage read the
    previous stage's output by bare name.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = RESULTS_ROOT / p
    return candidate if candidate.exists() else p
