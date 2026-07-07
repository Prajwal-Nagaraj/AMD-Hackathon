from agent.validate import (
    validate_code_debug,
    validate_code_gen,
    validate_factual,
    validate_logic,
    validate_math,
    validate_ner,
    validate_sentiment,
    validate_summarization,
)


def test_validate_factual_strips_preamble():
    ok, ans = validate_factual("Sure, here's the answer: Paris is the capital of France.")
    assert ok
    assert not ans.lower().startswith("sure")


def test_validate_factual_empty_fails():
    ok, ans = validate_factual("   ")
    assert not ok


def test_validate_math_extracts_answer_line():
    ok, ans = validate_math("Step 1: ...\nStep 2: ...\nANSWER: 42")
    assert ok
    assert ans == "42"


def test_validate_math_missing_answer_line_fails():
    ok, ans = validate_math("The result is probably around 42.")
    assert not ok


def test_validate_sentiment_requires_label():
    ok, ans = validate_sentiment("positive — food was great")
    assert ok
    ok2, _ = validate_sentiment("I liked it")
    assert not ok2


def test_validate_summarization_strips_preamble():
    ok, ans = validate_summarization("Certainly: the article discusses climate policy.")
    assert ok
    assert not ans.lower().startswith("certainly")


def test_validate_ner_passthrough():
    ok, ans = validate_ner("person: John Smith\norg: Google\nlocation: Paris\ndate: July 4th")
    assert ok
    assert "John Smith" in ans


def test_validate_code_gen_valid_python():
    raw = "```python\ndef add(a, b):\n    return a + b\n```"
    ok, code = validate_code_gen(raw)
    assert ok
    assert "def add" in code


def test_validate_code_gen_invalid_python():
    raw = "```python\ndef add(a, b)\n    return a + b\n```"
    ok, code = validate_code_gen(raw)
    assert not ok


def test_validate_code_debug_extracts_and_checks_code():
    raw = (
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "Cause: was subtracting instead of adding."
    )
    ok, text = validate_code_debug(raw)
    assert ok
    assert "Cause:" in text


def test_validate_code_debug_no_fence_fails():
    ok, text = validate_code_debug("The bug was a subtraction instead of addition.")
    assert not ok


def test_validate_logic_extracts_answer_line():
    ok, ans = validate_logic("Alice owns the cat, so...\nANSWER: Alice")
    assert ok
    assert ans == "Alice"
