"""Turn eval/sweep_results.json into a per-category recommendation: the
token-minimal candidate that clears the accuracy-gate proxy threshold.

Usage: python eval/report.py [threshold]   (threshold defaults to 0.7)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    path = ROOT / "eval" / "sweep_results.json"
    if not path.exists():
        print(f"No sweep results at {path} -- run eval/sweep.py first.")
        return

    rows = json.loads(path.read_text(encoding="utf-8"))
    by_category = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r)

    for category, entries in by_category.items():
        print(f"\n{category}")
        passing = [e for e in entries if e["avg_score"] >= threshold]
        candidates = sorted(passing or entries, key=lambda e: e["avg_tokens"])
        for e in candidates:
            marker = "PASS" if e["avg_score"] >= threshold else "FAIL"
            print(f"  [{marker}] {e['model']:22s} score={e['avg_score']:.2f} tokens={e['avg_tokens']:.1f}")
        if passing:
            best = candidates[0]
            print(f"  -> recommend: {best['model']} (score={best['avg_score']:.2f}, tokens={best['avg_tokens']:.1f})")
        else:
            print("  -> no candidate cleared the threshold; needs escalation or prompt tuning")


if __name__ == "__main__":
    main()
