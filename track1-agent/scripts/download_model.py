"""Download a local GGUF into track1-agent/models/ for calibration.

Default: Gemma-3-4b-it Q4_K_M (~2.5 GB, the local primary). Override --repo /
--pattern to pull a fallback candidate for the calibrate.py bake-off. The Qwen3
repo IDs below are the intended fallbacks -- confirm they exist before relying
on them (naming drifts between quant uploaders).

    python scripts/download_model.py                       # Gemma-3-4b-it Q4_K_M
    python scripts/download_model.py --candidate qwen3-4b
    python scripts/download_model.py --repo unsloth/Qwen3-1.7B-GGUF --pattern "*Q4_K_M.gguf"

Requires: pip install -r requirements-local.txt
"""

import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# (repo_id, glob pattern). Fallback repo IDs are best-guess -- verify on HF.
CANDIDATES = {
    "gemma-3-4b": ("unsloth/gemma-3-4b-it-GGUF", "*Q4_K_M.gguf"),
    "qwen3-4b": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "*Q4_K_M.gguf"),
    "qwen3-1.7b": ("unsloth/Qwen3-1.7B-GGUF", "*Q4_K_M.gguf"),
}


def main():
    ap = argparse.ArgumentParser(description="Download a GGUF into models/")
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), default="gemma-3-4b",
                    help="named candidate (default: gemma-3-4b)")
    ap.add_argument("--repo", default=None, help="override HF repo id")
    ap.add_argument("--pattern", default=None, help="override file glob (e.g. '*Q4_K_S.gguf')")
    ap.add_argument("--out", default=str(MODELS_DIR), help="output dir (default: models/)")
    args = ap.parse_args()

    repo, pattern = CANDIDATES[args.candidate]
    repo = args.repo or repo
    pattern = args.pattern or pattern

    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo} [{pattern}] -> {out} ...")
    snapshot_download(repo_id=repo, allow_patterns=[pattern], local_dir=str(out))

    ggufs = sorted(out.glob("*.gguf"))
    if not ggufs:
        print("WARNING: no .gguf matched -- check --repo / --pattern against the HF repo.")
        return
    for f in ggufs:
        print(f"  {f.name}  ({f.stat().st_size / 1e9:.2f} GB)")
    print(f"\nrun: python eval/calibrate.py --model {ggufs[0].as_posix()}")


if __name__ == "__main__":
    main()
