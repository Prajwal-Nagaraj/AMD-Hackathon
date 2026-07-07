"""Strategy sweep: for each category x candidate model, run the devset
through the category's prompt template and record (avg judge score, avg
tokens). Requires real Fireworks credentials -- this is dev-time cost, not
part of the submission.

Usage:
    export FIREWORKS_API_KEY=... FIREWORKS_BASE_URL=... ALLOWED_MODELS=...
    python eval/sweep.py
    python eval/report.py 0.7
"""

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.fireworks import FireworksClient  # noqa: E402
from agent.router import Category  # noqa: E402
from agent.strategies import STRATEGIES  # noqa: E402
from agent.validate import VALIDATORS  # noqa: E402
from judge import judge_answer  # noqa: E402

DEVSET_DIR = ROOT / "eval" / "devset"

# Candidate models to A/B per category -- start with the plan's pick plus the
# research doc's "try Gemma before the reasoning model" recommendation.
CANDIDATES = {
    Category.FACTUAL: ["gemma-4-26b-a4b-it", "gemma-4-31b-it"],
    Category.MATH: ["gemma-4-31b-it", "minimax-m3"],
    Category.SENTIMENT: ["gemma-4-26b-a4b-it", "gemma-4-31b-it"],
    Category.SUMMARIZATION: ["gemma-4-31b-it-nvfp4", "gemma-4-31b-it"],
    Category.NER: ["gemma-4-26b-a4b-it", "gemma-4-31b-it"],
    Category.CODE_DEBUG: ["gemma-4-31b-it", "kimi-k2p7-code"],
    Category.LOGIC: ["gemma-4-31b-it", "minimax-m3"],
    Category.CODE_GEN: ["gemma-4-31b-it", "kimi-k2p7-code"],
}


def load_devset(category: Category) -> list:
    path = DEVSET_DIR / f"{category.value}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


async def evaluate_category_model(client, judge_model: str, category: Category, model_id: str) -> dict:
    tasks = load_devset(category)
    strategy = STRATEGIES[category]
    validator = VALIDATORS[category.value]
    scores, tokens = [], []

    for t in tasks:
        user_msg = strategy.build_user(t["prompt"])
        result = await client.complete(
            model=model_id,
            system=strategy.system,
            user=user_msg,
            max_tokens=strategy.max_tokens,
            stop=strategy.stop,
        )
        _, answer = validator(result.text)
        score, _reason = await judge_answer(client, t["prompt"], t["gold"], answer, judge_model)
        scores.append(score)
        tokens.append(result.prompt_tokens + result.completion_tokens)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0
    return {
        "category": category.value,
        "model": model_id,
        "n": len(tasks),
        "avg_score": avg_score,
        "avg_tokens": avg_tokens,
    }


async def run_sweep(judge_model: str) -> list:
    client = FireworksClient()
    rows = []
    for category, models in CANDIDATES.items():
        for model_id in models:
            row = await evaluate_category_model(client, judge_model, category, model_id)
            rows.append(row)
            print(
                f"{row['category']:14s} {row['model']:22s} "
                f"n={row['n']:3d} avg_score={row['avg_score']:.2f} avg_tokens={row['avg_tokens']:.1f}"
            )
    return rows


def main():
    judge_model = os.environ.get("JUDGE_MODEL", "minimax-m3")
    rows = asyncio.run(run_sweep(judge_model))
    out_path = ROOT / "eval" / "sweep_results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
