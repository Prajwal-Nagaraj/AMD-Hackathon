"""Local, free, CPU-only task classification.

Misrouting only wastes tokens (the local validator + escalation ladder
catches correctness problems downstream), so this stays a lightweight
heuristic set rather than a trained classifier. Tune categories/patterns
against the eval devset before adding any ML here (see eval/sweep.py).
"""

import re
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    LOGIC = "logic"
    CODE_GEN = "code_gen"


_CODE_FENCE_RE = re.compile(r"```")
_CODE_SYNTAX_RE = re.compile(r"\b(def |function |class \w+|=>|#include|public static|void main)\b")
_BUG_WORDS_RE = re.compile(
    r"\b(bug|fix|error|exception|traceback|doesn'?t work|incorrect output|debug|not working)\b", re.I
)
_GEN_WORDS_RE = re.compile(r"\b(write|implement|create a function|generate a function)\b", re.I)
_SUMMARY_WORDS_RE = re.compile(
    r"(summari[sz]e|summary|condense|shorten|abridge|tl;?dr|in a nutshell"
    r"|key (points|facts|takeaways))",
    re.I,
)
_SENTIMENT_WORDS_RE = re.compile(r"\b(sentiment|tone|positive or negative|how does .* feel)\b", re.I)
_NER_WORDS_RE = re.compile(
    r"(entities|named entit|extract (people|persons|organi[sz]ations|locations|dates)"
    r"|who\b.*mentioned|list (the )?(people|organizations|locations))",
    re.I,
)
_LOGIC_WORDS_RE = re.compile(
    r"\b(if .* then|exactly one|at least one|no two|must be (true|false)|puzzle|constraint"
    r"|who (is|owns|lives)|order them|based only on|yes or no|all \w+ are \w+)\b",
    re.I,
)
_MATH_RE = re.compile(
    r"(\d\s*[-+*/%^]\s*\d|percent|%|how (much|many)|average|\bmean\b|sum of|calculate"
    r"|km/h|km/hr|\bmph\b|per (hour|second|minute|day|week|month|year)"
    r"|total (journey )?(time|cost|distance|price|amount|weight)|what is the total)",
    re.I,
)


def classify(prompt: str) -> Category:
    text = prompt.strip()

    has_code_fence = bool(_CODE_FENCE_RE.search(text))
    has_code_syntax = bool(_CODE_SYNTAX_RE.search(text))
    if has_code_fence or has_code_syntax:
        if _BUG_WORDS_RE.search(text):
            return Category.CODE_DEBUG
        if _GEN_WORDS_RE.search(text) or has_code_fence:
            return Category.CODE_GEN
        return Category.CODE_GEN

    if _SUMMARY_WORDS_RE.search(text):
        return Category.SUMMARIZATION
    if _SENTIMENT_WORDS_RE.search(text):
        return Category.SENTIMENT
    if _NER_WORDS_RE.search(text):
        return Category.NER
    if _LOGIC_WORDS_RE.search(text):
        return Category.LOGIC
    if _MATH_RE.search(text):
        return Category.MATH
    return Category.FACTUAL
