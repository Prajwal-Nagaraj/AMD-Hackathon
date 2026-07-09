"""Orchestrator: load tasks, route+answer+validate each one under a bounded
concurrent pool, enforce a global deadline, and always write valid
/output/results.json before exiting 0.

A partial-but-valid result beats a timeout that produces malformed/missing
output (which scores zero) -- see TRACK-1-IMPLEMENTATION-PLAN.md SS7.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from .fireworks import FireworksClient
from .router import classify
from .strategies import STRATEGIES
from .telemetry import Telemetry
from .validate import VALIDATORS

INPUT_PATH = Path(os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json"))
DEADLINE_SECONDS = float(os.environ.get("DEADLINE_SECONDS", "570"))  # 9m30s of the 10m budget
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "8"))
PER_TASK_TIMEOUT = float(os.environ.get("PER_TASK_TIMEOUT", "28"))  # under the 30s per-request rule


def load_tasks(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    tmp.replace(path)


async def run_task(client, telemetry: Telemetry, task: dict, deadline: float) -> dict:
    task_id = task.get("task_id")
    prompt = task.get("prompt", "")

    try:
        category = classify(prompt)
        strategy = STRATEGIES[category]
        validator = VALIDATORS[category.value]
        user_msg = strategy.build_user(prompt)

        primary_model = client.model_for_tier(strategy.primary_tier)
        timeout = min(PER_TASK_TIMEOUT, max(1.0, deadline - time.monotonic()))
        result = await client.complete(
            model=primary_model,
            system=strategy.system,
            user=user_msg,
            max_tokens=strategy.max_tokens,
            stop=strategy.stop,
            timeout=timeout,
        )
        ok, answer = validator(result.text)
        telemetry.record(
            task_id, category.value, result.model, result.prompt_tokens, result.completion_tokens
        )

        if not ok and strategy.escalation_tier and time.monotonic() < deadline:
            esc_model = client.model_for_tier(strategy.escalation_tier)
            # Skip a pointless retry when the tier collapses onto the same model
            # (e.g. a short ALLOWED_MODELS list).
            if esc_model != result.model:
                timeout = min(PER_TASK_TIMEOUT, max(1.0, deadline - time.monotonic()))
                esc_result = await client.complete(
                    model=esc_model,
                    system=strategy.system,
                    user=user_msg,
                    max_tokens=strategy.max_tokens,
                    stop=strategy.stop,
                    timeout=timeout,
                )
                esc_ok, esc_answer = validator(esc_result.text)
                telemetry.record(
                    task_id,
                    category.value,
                    esc_result.model,
                    esc_result.prompt_tokens,
                    esc_result.completion_tokens,
                    escalated=True,
                )
                if esc_ok or esc_answer:
                    answer = esc_answer

        if not answer:
            answer = result.text.strip()
        return {"task_id": task_id, "answer": answer}
    except Exception:
        return {"task_id": task_id, "answer": ""}


async def run_all(tasks: list, client=None) -> list:
    if client is None:
        client = FireworksClient()
    tiers = getattr(client, "tiers", None)
    if tiers:
        sys.stderr.write(json.dumps({"tiers": tiers}) + "\n")
    telemetry = Telemetry()
    deadline = time.monotonic() + DEADLINE_SECONDS
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def guarded(task):
        async with sem:
            if time.monotonic() >= deadline:
                return {"task_id": task.get("task_id"), "answer": ""}
            return await run_task(client, telemetry, task, deadline)

    futures = [asyncio.ensure_future(guarded(t)) for t in tasks]
    total_timeout = max(1.0, deadline - time.monotonic())
    done, pending = await asyncio.wait(futures, timeout=total_timeout)

    results_by_id = {}
    for fut in done:
        r = fut.result()
        results_by_id[r["task_id"]] = r
    for fut in pending:
        fut.cancel()

    ordered = []
    for t in tasks:
        tid = t.get("task_id")
        ordered.append(results_by_id.get(tid, {"task_id": tid, "answer": ""}))

    sys.stderr.write(json.dumps({"telemetry": telemetry.summary()}) + "\n")
    return ordered


def main() -> None:
    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception:
        write_results(OUTPUT_PATH, [])
        sys.exit(0)
        return

    try:
        results = asyncio.run(run_all(tasks))
    except Exception:
        results = [{"task_id": t.get("task_id"), "answer": ""} for t in tasks]

    write_results(OUTPUT_PATH, results)
    sys.exit(0)


if __name__ == "__main__":
    main()
