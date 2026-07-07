"""Local, free validation + coercion of model output.

Turns "wrong format" failures into passes at zero token cost: strip
preambles, extract the structured part, sanity-check code. Each validator
returns (ok, answer) -- `ok=False` signals the caller should escalate to a
stronger model; `answer` is always the best coerced text to fall back on
even when `ok` is False.
"""

import ast
import re

_PREAMBLE_RE = re.compile(
    r"^(sure|okay|ok|certainly|here'?s|here is|the answer is)[,:]?\s*", re.I
)
_FENCE_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.S)
_ANSWER_LINE_RE = re.compile(r"ANSWER:\s*(.+)", re.I)


def _strip_preamble(text: str) -> str:
    return _PREAMBLE_RE.sub("", text.strip(), count=1).strip()


def _extract_fenced_code(text: str):
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_answer_line(text: str):
    m = _ANSWER_LINE_RE.search(text)
    return m.group(1).strip() if m else None


def validate_factual(raw: str):
    text = _strip_preamble(raw)
    return bool(text), text


def validate_math(raw: str):
    ans = _extract_answer_line(raw)
    if ans:
        return True, ans
    text = _strip_preamble(raw)
    return False, text


def validate_sentiment(raw: str):
    first_line = _strip_preamble(raw).splitlines()[0] if raw.strip() else ""
    ok = bool(re.search(r"\b(positive|negative|neutral)\b", first_line, re.I))
    return ok, first_line


def validate_summarization(raw: str):
    text = _strip_preamble(raw)
    return bool(text), text


def validate_ner(raw: str):
    text = _strip_preamble(raw)
    return bool(text), text


def validate_code_debug(raw: str):
    code = _extract_fenced_code(raw)
    text = _strip_preamble(raw)
    if not code:
        return False, text
    try:
        ast.parse(code)
        return True, text
    except SyntaxError:
        return False, text


def validate_logic(raw: str):
    ans = _extract_answer_line(raw)
    if ans:
        return True, ans
    text = _strip_preamble(raw)
    return False, text


def validate_code_gen(raw: str):
    code = _extract_fenced_code(raw) or _strip_preamble(raw)
    try:
        ast.parse(code)
        return True, code
    except SyntaxError:
        return False, code


VALIDATORS = {
    "factual": validate_factual,
    "math": validate_math,
    "sentiment": validate_sentiment,
    "summarization": validate_summarization,
    "ner": validate_ner,
    "code_debug": validate_code_debug,
    "logic": validate_logic,
    "code_gen": validate_code_gen,
}
