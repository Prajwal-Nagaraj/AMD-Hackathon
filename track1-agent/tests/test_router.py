from agent.router import Category, classify


def test_factual():
    assert classify("What is the capital of France?") == Category.FACTUAL


def test_math():
    assert (
        classify("If a shirt costs $40 and is discounted 25%, what is the final price?")
        == Category.MATH
    )


def test_sentiment():
    assert (
        classify("What is the sentiment of this review: 'The food was amazing!'")
        == Category.SENTIMENT
    )


def test_summarization():
    assert classify("Summarize the following text in one sentence: ...") == Category.SUMMARIZATION


def test_ner():
    assert (
        classify(
            "Extract all named entities (person, org, location, date) from: "
            "John Smith met with Google in Paris on July 4th."
        )
        == Category.NER
    )


def test_code_debug():
    prompt = "This function has a bug:\n```python\ndef add(a, b):\n    return a - b\n```\nFix it."
    assert classify(prompt) == Category.CODE_DEBUG


def test_code_gen():
    prompt = "Write a Python function that returns the nth Fibonacci number."
    assert classify(prompt) == Category.CODE_GEN


def test_logic():
    prompt = (
        "Three friends Alice, Bob, and Carol each own exactly one pet. "
        "If Alice does not own the dog and Bob owns the cat, who owns the dog?"
    )
    assert classify(prompt) == Category.LOGIC


# Regression cases: real testset prompts that previously fell through to FACTUAL.
def test_logic_syllogism_not_factual():
    prompt = (
        "All roses are flowers. All flowers need water. Based only on these "
        "statements, does a rose need water? Answer Yes or No."
    )
    assert classify(prompt) == Category.LOGIC


def test_math_rate_word_problem_not_factual():
    prompt = (
        "A train travels 240 km at 80 km/h, then 150 km at 50 km/h. "
        "What is the total journey time in hours and minutes?"
    )
    assert classify(prompt) == Category.MATH


def test_summarization_condense_not_factual():
    prompt = (
        "Condense the following paragraph into a single bullet-point list of "
        "no more than four key facts: Blockchain is a distributed ledger ..."
    )
    assert classify(prompt) == Category.SUMMARIZATION
