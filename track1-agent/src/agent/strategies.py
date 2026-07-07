"""Per-category strategy: prompt template, model, token limits, escalation.

Model choices follow the Gemma-first doctrine from TRACK-1-MODEL-RESEARCH.md:
the Gemma variants are non-reasoning by default and share the densest
tokenizer, so they're cheapest on every axis. `minimax-m3` / `kimi-k2p7-code`
are reserved as escalation targets for tasks the Gemma pass fails to clear
the accuracy gate on -- every reasoning-model call is a token tax taken on
purpose, not the default.

These are starting hypotheses (plan doc SS5) -- tune model choice, max_tokens
and prompts against eval/sweep.py once real Fireworks access is available.

TODO (confirm at kickoff, see plan SS13 / research SS10): once the exact
Fireworks extra_body param for toggling `minimax-m3` "thinking" off is
known, set it in MATH/LOGIC's `extra_body` so the default (non-escalated)
path never pays the reasoning-token tax. Left empty until confirmed rather
than guessing an unverified param name.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from .router import Category


@dataclass
class Strategy:
    model: str
    max_tokens: int
    build_user: Callable[[str], str]
    system: Optional[str] = None
    stop: Optional[list] = None
    extra_body: dict = field(default_factory=dict)
    escalation_model: Optional[str] = None
    escalation_extra_body: dict = field(default_factory=dict)


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


STRATEGIES = {
    Category.FACTUAL: Strategy(
        model="gemma-4-26b-a4b-it",
        max_tokens=120,
        stop=["\n\n"],
        build_user=_factual_user,
        escalation_model="gemma-4-31b-it",
    ),
    Category.MATH: Strategy(
        model="gemma-4-31b-it",
        max_tokens=300,
        build_user=_math_user,
        escalation_model="minimax-m3",
    ),
    Category.SENTIMENT: Strategy(
        model="gemma-4-26b-a4b-it",
        max_tokens=20,
        stop=["\n"],
        build_user=_sentiment_user,
        escalation_model="gemma-4-31b-it",
    ),
    Category.SUMMARIZATION: Strategy(
        model="gemma-4-31b-it-nvfp4",
        max_tokens=250,
        build_user=_summarization_user,
        escalation_model="gemma-4-31b-it",
    ),
    Category.NER: Strategy(
        model="gemma-4-26b-a4b-it",
        max_tokens=150,
        build_user=_ner_user,
        escalation_model="gemma-4-31b-it",
    ),
    Category.CODE_DEBUG: Strategy(
        model="gemma-4-31b-it",
        max_tokens=400,
        build_user=_code_debug_user,
        escalation_model="kimi-k2p7-code",
    ),
    Category.LOGIC: Strategy(
        model="gemma-4-31b-it",
        max_tokens=300,
        build_user=_logic_user,
        escalation_model="minimax-m3",
    ),
    Category.CODE_GEN: Strategy(
        model="gemma-4-31b-it",
        max_tokens=400,
        build_user=_code_gen_user,
        escalation_model="kimi-k2p7-code",
    ),
}
