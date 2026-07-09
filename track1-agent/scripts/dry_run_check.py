"""Dry-run readiness checker (no Fireworks key needed).

Reads sample_input/tasks.json + sample_output/results.json (produced by a real
`python -m agent.main` run against your Fireworks creds) and reports, per task:
router category, non-empty?, passes the category's local format validator?, and
the testset gold answer alongside ours for eyeballing.

This is a *format/coverage* gate, not the real LLM-judge -- it tells us the
container is producing well-formed, on-category, non-empty answers before we
spend a submission slot. Run:

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe scripts\\dry_run_check.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.router import classify  # noqa: E402
from agent.validate import VALIDATORS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TASKS = os.path.join(ROOT, "sample_input", "tasks.json")
RESULTS = os.path.join(ROOT, "sample_output", "results.json")
TESTSET_GLOB = os.path.join(ROOT, "..", "testset", "tasks_*.json")


def load_gold():
    gold = {}
    for f in glob.glob(TESTSET_GLOB):
        for t in json.load(open(f, encoding="utf-8")):
            gold[t["prompt"].strip()] = (t.get("category", "?"), t.get("answer", ""))
    return gold


def main():
    tasks = {t["task_id"]: t["prompt"] for t in json.load(open(TASKS, encoding="utf-8"))}
    results = {r["task_id"]: r.get("answer", "") for r in json.load(open(RESULTS, encoding="utf-8"))}
    gold = load_gold()

    non_empty = fmt_ok = 0
    n = len(tasks)
    print(f"{'id':<5}{'router-cat':<14}{'gold-cat':<24}{'empty':<7}{'fmt':<5}chars")
    print("-" * 70)
    for tid, prompt in tasks.items():
        ans = (results.get(tid) or "").strip()
        cat = classify(prompt)
        if cat.value in ("math", "logic"):
            # main.py already extracts the ANSWER: value, so re-running the
            # validator (which looks for the ANSWER: prefix) would double-penalise
            # a correctly-coerced answer. A non-empty coerced value is the pass.
            ok = bool(ans)
        else:
            ok, _ = VALIDATORS[cat.value](ans) if ans else (False, "")
        gcat = gold.get(prompt.strip(), ("?", ""))[0]
        non_empty += bool(ans)
        fmt_ok += bool(ok)
        flag = "" if ans else "EMPTY"
        print(f"{tid:<5}{cat.value:<14}{gcat:<24}{flag:<7}{'ok' if ok else 'FAIL':<5}{len(ans)}")

    print("-" * 70)
    print(f"non-empty: {non_empty}/{n}   format-pass: {fmt_ok}/{n}")
    if non_empty < n:
        print("\n!! Some answers are EMPTY -> likely missing/failed Fireworks calls "
              "(bad key/base_url, Gemma not deployed, or a MODEL_VIOLATION). Fix before submitting.")
    elif fmt_ok < n:
        print("\n~ All answered but some fail local format checks -> inspect those "
              "categories' strategies/validators; may still pass the judge, but worth a look.")
    else:
        print("\nOK: every task answered and well-formed. Spot-check a few answers vs gold, "
              "then this baseline is safe to push.")
    # Dump full answers for eyeballing
    print("\n=== full answers ===")
    for tid, prompt in tasks.items():
        print(f"\n[{tid}] {prompt[:80]}")
        print("  OURS:", (results.get(tid) or "").strip()[:400])


if __name__ == "__main__":
    main()
