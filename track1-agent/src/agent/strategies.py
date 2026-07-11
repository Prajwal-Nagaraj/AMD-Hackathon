"""Per-category strategy: prompt template, model tier, token limits, escalation.

Strategies name a *tier* (cheap / mid / strong / code), never a concrete model
ID -- `fireworks.infer_tiers` derives the real model from ALLOWED_MODELS at
runtime (competition rule: read model IDs from ALLOWED_MODELS, don't hardcode).

Doctrine (safe baseline, after the first submission missed the accuracy gate):
the correctness-critical categories -- factual, math, logic -- answer directly
on the *strong* tier, and code_debug / code_gen on the *code* tier, instead of
starting cheap and hoping the local validator catches a wrong answer (it only
checks format, not correctness -- see validate.py). Only the genuinely easy
extraction categories (sentiment, ner) stay on cheap, with summarization on mid.
Escalation is now a blank/format safety net -- it retries on a *different* model
when the primary's output fails local validation -- not a correctness upgrade.
Token budgets are sized to avoid truncating a full answer; reasoning-token
suppression (fireworks.py) keeps the strong/reasoning tier from paying a
hidden-reasoning tax that previously made it return blank.

Once we clear the gate, push categories back toward cheaper tiers as
eval/sweep.py measurements justify it.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .router import Category

# Tier names; concrete models are inferred from ALLOWED_MODELS at runtime.
CHEAP, MID, STRONG, CODE = "cheap", "mid", "strong", "code"


@dataclass
class Strategy:
    primary_tier: str
    max_tokens: int
    build_user: Callable[[str], str]
    system: Optional[str] = None
    stop: Optional[list] = None
    escalation_tier: Optional[str] = None


def _factual_user(prompt: str) -> str:
    return f"{prompt}\nAnswer in 1-3 sentences. No preamble."


def _math_user(prompt: str) -> str:
    return f"{prompt}\nSolve with minimal steps. End with a final line exactly: ANSWER: <value>"


def _sentiment_user(prompt: str) -> str:
    return (
        f"{prompt}\n"
        "Respond in the form: <label> — <reason in 5 words or fewer>. "
        "Label must be one of: positive, negative, neutral."
    )


def _summarization_user(prompt: str) -> str:
    return f"{prompt}\nFollow the requested format and length exactly. No preamble."


def _ner_user(prompt: str) -> str:
    return (
        f"{prompt}\n"
        "List entities as `type: value` pairs, one per line. "
        "Types are limited to person, org, location, date. No preamble."
    )


def _code_debug_user(prompt: str) -> str:
    return (
        f"{prompt}\n"
        "Return the corrected code in a single fenced code block, "
        "then one line starting with 'Cause:' naming the bug. No other text."
    )


def _logic_user(prompt: str) -> str:
    return f"{prompt}\nReason minimally. End with a final line exactly: ANSWER: <value>"


def _code_gen_user(prompt: str) -> str:
    return f"{prompt}\nOutput only the function in a single fenced code block. No prose."


# Primary tier / escalation tier per category. The escalation tier is only
# reached when the local validator predicts a gate failure (main.run_task).
STRATEGIES = {
    Category.FACTUAL: Strategy(
        primary_tier=STRONG,
        max_tokens=300,
        build_user=_factual_user,
        escalation_tier=MID,
    ),
    Category.MATH: Strategy(
        primary_tier=STRONG,
        max_tokens=400,
        build_user=_math_user,
        escalation_tier=MID,
    ),
    Category.SENTIMENT: Strategy(
        primary_tier=CHEAP,
        max_tokens=120,
        stop=["\n"],
        build_user=_sentiment_user,
        escalation_tier=MID,
    ),
    Category.SUMMARIZATION: Strategy(
        primary_tier=MID,
        max_tokens=250,
        build_user=_summarization_user,
        escalation_tier=STRONG,
    ),
    Category.NER: Strategy(
        primary_tier=CHEAP,
        max_tokens=150,
        build_user=_ner_user,
        escalation_tier=MID,
    ),
    Category.CODE_DEBUG: Strategy(
        primary_tier=CODE,
        max_tokens=520,
        build_user=_code_debug_user,
        escalation_tier=STRONG,
    ),
    Category.LOGIC: Strategy(
        primary_tier=STRONG,
        max_tokens=420,
        build_user=_logic_user,
        escalation_tier=MID,
    ),
    Category.CODE_GEN: Strategy(
        primary_tier=CODE,
        max_tokens=520,
        build_user=_code_gen_user,
        escalation_tier=STRONG,
    ),
}
