"""Proxy LLM-judge: approximates the hidden accuracy gate locally so we can
sweep strategies without burning real submissions. Scores an agent answer
against a gold description of the expected intent on a 0.0-1.0 scale.

This is dev-time tooling -- it spends real Fireworks tokens, on purpose, to
estimate whether a strategy would clear the actual gate.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.fireworks import FireworksClient  # noqa: E402

DEFAULT_JUDGE_MODEL = "minimax-m3"

JUDGE_SYSTEM = "You are a strict grader. Output only a JSON object, nothing else."

_JSON_RE = re.compile(r"\{.*\}", re.S)


def build_judge_prompt(prompt: str, gold: str, answer: str) -> str:
    return (
        f"Task given to the agent:\n{prompt}\n\n"
        f"Expected intent / gold answer:\n{gold}\n\n"
        f"Agent's answer:\n{answer}\n\n"
        "Score how well the agent answer satisfies the expected intent, on a 0.0-1.0 scale "
        "(1.0 = fully correct and complete, 0.0 = wrong, empty, or irrelevant). "
        'Respond with only JSON: {"score": <0-1 float>, "reason": "<=8 words"}'
    )


async def judge_answer(client, prompt: str, gold: str, answer: str, judge_model: str = DEFAULT_JUDGE_MODEL):
    result = await client.complete(
        model=judge_model,
        system=JUDGE_SYSTEM,
        user=build_judge_prompt(prompt, gold, answer),
        max_tokens=60,
        temperature=0.0,
    )
    m = _JSON_RE.search(result.text)
    if not m:
        return 0.0, "unparseable judge output"
    try:
        data = json.loads(m.group(0))
        return float(data.get("score", 0.0)), data.get("reason", "")
    except (ValueError, TypeError):
        return 0.0, "unparseable judge JSON"


def _cli():
    parser = argparse.ArgumentParser(description="Ad hoc single-triple judge call")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    client = FireworksClient()
    score, reason = asyncio.run(judge_answer(client, args.prompt, args.gold, args.answer, args.model))
    print(json.dumps({"score": score, "reason": reason}))


if __name__ == "__main__":
    _cli()
