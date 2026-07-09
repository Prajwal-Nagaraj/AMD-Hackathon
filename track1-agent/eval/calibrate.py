"""Local-model calibration -- the empirical arbiter for the local-model choice.

Runs the bundled local model (default: Gemma-3-4b-it Q4_K_M) over eval/devset,
one category at a time, and reports per category:

  * valid%   -- fraction whose output passed the free local validator/coercer
  * judge    -- mean proxy-judge score + pass-rate at --threshold (optional; the
                --judge flag spends real Fireworks tokens on the grading calls)
  * mean_s   -- mean local inference wall-clock per task (the real constraint on
                a 2 vCPU CPU box; local tokens are not scored)

plus the model load time and the projected wall-clock for a full 19-task run.

This is what decides, per category, whether the local model answers a task for
free or we escalate it to Fireworks -- and whether a candidate model fits the
10-minute budget at all on a 4 GB / 2 vCPU box. It reuses the real per-category
prompt templates (strategies.py) and validators (validate.py) so the numbers
reflect the actual agent path. See TRACK-1-IMPLEMENTATION-PLAN.md.

    python eval/calibrate.py                    # free: validation + wall-clock
    python eval/calibrate.py --judge            # + accuracy (spends Fireworks tokens)
    python eval/calibrate.py --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --judge
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.local_llm import LocalLLM  # noqa: E402
from agent.router import Category  # noqa: E402
from agent.strategies import STRATEGIES  # noqa: E402
from agent.validate import VALIDATORS  # noqa: E402

DEVSET_DIR = Path(__file__).resolve().parent / "devset"
FULL_RUN_TASKS = 19  # the real grader runs a fixed set of 19 tasks
DEADLINE_SECONDS = 570  # what main.py budgets of the 10-minute wall-clock


def load_devset():
    devset = {}
    for cat in Category:
        path = DEVSET_DIR / f"{cat.value}.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                devset[cat] = json.load(f)
    return devset


def run_local(llm, devset):
    """Run every devset task through the local model; return per-task records."""
    records = []
    for cat, tasks in devset.items():
        strat = STRATEGIES[cat]
        validator = VALIDATORS[cat.value]
        for t in tasks:
            user = strat.build_user(t["prompt"])
            res = llm.complete_sync(
                user=user,
                system=strat.system,
                max_tokens=strat.max_tokens,
                stop=strat.stop,
            )
            ok, answer = validator(res.text)
            records.append(
                {
                    "category": cat.value,
                    "prompt": t["prompt"],
                    "gold": t.get("gold", ""),
                    "answer": answer or res.text,
                    "valid": ok,
                    "latency_s": res.latency_s,
                    "completion_tokens": res.completion_tokens,
                }
            )
    return records


async def judge_records(records, threshold):
    from agent.fireworks import FireworksClient
    from judge import judge_answer  # eval/judge.py (script dir is on sys.path)

    client = FireworksClient()
    sem = asyncio.Semaphore(8)

    async def one(rec):
        async with sem:
            score, reason = await judge_answer(client, rec["prompt"], rec["gold"], rec["answer"])
            rec["judge_score"] = score
            rec["judge_pass"] = score >= threshold
            rec["judge_reason"] = reason

    await asyncio.gather(*(one(r) for r in records))


def summarize(records, judged):
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    header = f"{'category':<15}{'n':>3}  {'valid%':>7}"
    if judged:
        header += f"  {'judge':>6}  {'pass%':>6}"
    header += f"  {'mean_s':>7}  {'max_s':>7}"
    print()
    print(header)
    print("-" * len(header))

    all_lat = []
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        n = len(rs)
        lat = [r["latency_s"] for r in rs]
        all_lat += lat
        valid = 100.0 * sum(r["valid"] for r in rs) / n
        line = f"{cat:<15}{n:>3}  {valid:>6.0f}%"
        if judged:
            js = [r.get("judge_score", 0.0) for r in rs]
            jp = 100.0 * sum(r.get("judge_pass", False) for r in rs) / n
            line += f"  {sum(js) / n:>6.2f}  {jp:>5.0f}%"
        line += f"  {sum(lat) / n:>7.2f}  {max(lat):>7.2f}"
        print(line)

    n_all = len(all_lat)
    mean_lat = sum(all_lat) / n_all if n_all else 0.0
    proj = mean_lat * FULL_RUN_TASKS
    print("-" * len(header))
    overall_valid = 100.0 * sum(r["valid"] for r in records) / len(records)
    tail = f"OVERALL        {len(records):>3}  {overall_valid:>6.0f}%"
    if judged:
        js = [r.get("judge_score", 0.0) for r in records]
        jp = 100.0 * sum(r.get("judge_pass", False) for r in records) / len(records)
        tail += f"  {sum(js) / len(records):>6.2f}  {jp:>5.0f}%"
    tail += f"  {mean_lat:>7.2f}"
    print(tail)
    print()
    print(f"projected 19-task wall-clock: {proj:.0f}s  (budget {DEADLINE_SECONDS}s)", end="")
    print("  -- OVER BUDGET, will time out" if proj > DEADLINE_SECONDS else "  -- fits")


def main():
    ap = argparse.ArgumentParser(description="Calibrate a local model on the devset")
    ap.add_argument("--model", default=None, help="GGUF path (default: $LOCAL_MODEL_PATH or Gemma-3-4b-it)")
    ap.add_argument("--judge", action="store_true", help="score accuracy via the Fireworks proxy judge (spends tokens)")
    ap.add_argument("--threshold", type=float, default=0.7, help="judge pass threshold (default 0.7)")
    ap.add_argument("--dump", default=None, help="write per-task records to this JSON path")
    args = ap.parse_args()

    devset = load_devset()
    total = sum(len(v) for v in devset.values())
    if not total:
        print("no devset found under eval/devset/", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    llm = LocalLLM(model_path=args.model)
    load_s = time.monotonic() - t0
    print(f"loaded {llm.model_name} in {load_s:.1f}s  (startup budget is 60s incl. load)")
    print(f"running {total} devset tasks across {len(devset)} categories...")

    records = run_local(llm, devset)
    if args.judge:
        asyncio.run(judge_records(records, args.threshold))

    summarize(records, judged=args.judge)

    if args.dump:
        Path(args.dump).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
