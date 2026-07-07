# Track 1 — Token-Efficient General-Purpose Agent: Implementation Plan

> Our entry for AMD Developer Hackathon ACT II, Track 1.
> Source of truth: `Participant Guide_ AMD Developer Hackathon (ACT II).md`. Re-confirm anything marked ❓ at kickoff / on Discord.
> Build window: **Jul 6 → Jul 11, 2026 (submissions close Jul 11, 9:30 PM IST).**

---

## 1. What we're actually building (and what "winning" means)

Track 1 is **not** just a router. It's a **general-purpose agent** that answers arbitrary natural-language tasks spanning **8 capability categories**, using Fireworks AI models as token-efficiently as possible.

**The 8 categories (evaluated on all of them):**

| # | Category | Output shape |
|---|---|---|
| 1 | Factual knowledge | Concise explanation/definition |
| 2 | Mathematical reasoning | Correct final value (multi-step) |
| 3 | Sentiment classification | Label **+ short justification** |
| 4 | Text summarisation | Condensed to a given format/length constraint |
| 5 | Named entity recognition | Labelled entities (person/org/location/date) |
| 6 | Code debugging | Bug identified + corrected code |
| 7 | Logical / deductive reasoning | Answer satisfying all constraints |
| 8 | Code generation | Correct function from a spec |

**Scoring is two-stage — this dictates everything:**

1. **Accuracy gate** — an LLM-Judge scores each answer against expected intent. Below the threshold ⇒ **excluded from the leaderboard entirely** (a hard cutoff, not a weighted term).
2. **Token efficiency** — everyone who clears the gate is ranked by **total tokens recorded by the judging proxy, ascending. Fewer = better.** This is **raw token count (prompt + completion), not weighted by model price.**

> **The whole strategy in one line:** *Clear the gate on every single task with the maximum margin our token budget allows, then win by emitting the fewest possible input + output tokens.* Accuracy is a wall you must get over; tokens are the race among those who cleared it.

**Allowed models (read from `ALLOWED_MODELS` at runtime — never hardcode):**

- `minimax-m3` — presumed strongest general reasoner → hard math/logic.
- `kimi-k2p7-code` — code specialist → debugging + generation.
- `gemma-4-31b-it` — strong general-purpose.
- `gemma-4-26b-a4b-it` — MoE (~4B active), cheap/fast → easy high-volume categories.
- `gemma-4-31b-it-nvfp4` — 4-bit quant of the 31B, faster, near-same quality.

> The three Gemma variants **share the Gemma tokenizer** ⇒ identical token count for identical output text; choose among them by **accuracy + latency**, not tokens. `minimax` and `kimi` have their own tokenizers, so the *same answer* may cost a different number of tokens on them — worth a quick measurement at kickoff (2nd-order).

---

## 2. Guiding principles (the token ⇄ accuracy tension)

Every design choice is a trade between clearing the gate (wants more reasoning/tokens) and ranking well (wants fewer tokens). Our doctrine:

> **Bright-line invariant (never cross):** *Every answer's content must be produced by a Fireworks call to an `ALLOWED_MODELS` model* — including trivial answers and "plain code" for simple generation tasks. Local logic may **route, validate, and format**, but must **never author, compute, or shortcut the answer** (no local `eval("2+2")`, no locally-emitted boilerplate). Anything not produced through `FIREWORKS_BASE_URL` isn't metered → doesn't count → effectively invalid. When in doubt, the model writes it; we only reshape what the model returned.

1. **One Fireworks call per task by default.** No self-consistency voting, no multi-agent debate, no re-asking — those multiply tokens. Multiple calls only via *escalation* (below), and only when a local check predicts a gate failure.
2. **Do all routing, classification, and post-processing locally — it's free.** The moderator confirmed local models/logic count as zero. Exploit this ruthlessly for everything *except the answer-producing inference*.
3. **Minimize input tokens:** terse or no system prompt, zero-shot by default, never echo the task back, no chain-of-thought scaffolding in the prompt. We must send the task content (passage/code) — everything *around* it is overhead to cut.
4. **Minimize output tokens:** instruct "output only X, no explanation" wherever justification isn't required; set tight per-category `max_tokens`; use `stop` sequences to cut generation the moment the answer is complete.
5. **`temperature = 0` (greedy) everywhere.** Deterministic, reproducible, removes any need for multi-sampling, and stabilizes gate-clearing.
6. **Route to the *minimum-capable* model per category.** Use the cheapest model that reliably clears the gate; reserve `minimax`/`kimi` for tasks that genuinely need them.
7. **Never emit malformed or missing output.** Malformed JSON / missing answers score zero. A global deadline guard guarantees we always write valid `/output/results.json` and exit 0.

---

## 3. System architecture

```
/input/tasks.json ──▶ main.py (orchestrator)
                         │
                         ├─▶ router.py        (LOCAL, free): category + difficulty → strategy
                         │
                         ├─▶ strategies.py    per-category: prompt template, model, max_tokens, stop
                         │
                         ├─▶ fireworks.py     OpenAI-compatible client → FIREWORKS_BASE_URL (metered)
                         │
                         ├─▶ validate.py      (LOCAL, free): parse/plausibility check + coerce format
                         │        │
                         │        └─(fail)─▶ escalate: stronger model / richer prompt (rare)
                         │
                         └─▶ writer            format → /output/results.json ──▶ exit 0
```

**Components:**

- **`main.py`** — load tasks, run them concurrently (async, bounded pool) to fit the 10-min budget, enforce a global deadline, write results, exit 0. Always writes valid JSON even on partial completion.
- **`router.py`** — **local, CPU-only.** Classifies each task into one of the 8 categories (+ a coarse difficulty signal) and returns a strategy handle. Misrouting only wastes tokens (validation+escalation catches correctness), so this stays lightweight.
- **`strategies.py`** — a per-category spec: system/user template, chosen model, `max_tokens`, `stop`, and whether justification/structure is required.
- **`fireworks.py`** — OpenAI-compatible client pointed at `FIREWORKS_BASE_URL`, keyed by `FIREWORKS_API_KEY`, models from `ALLOWED_MODELS`. Retries with backoff. Records our own token telemetry for tuning.
- **`validate.py`** — **local, free.** Per-category checks (JSON parses? number present? code `ast.parse`s? within length constraint?) and **coercion** (strip "Sure, here's…" preambles, extract the JSON/answer, truncate a one-sentence summary at the first period). This fixes format-mismatch "failures" at zero token cost.

**Concurrency & limits:** Process tasks in parallel with a bounded async pool. Keep each task's wall-clock < 30s (per-request rule). Add backoff for any proxy-side rate limiting. Respect the 10-min *total* budget via the deadline guard below.

---

## 4. Local router (free, CPU-only, <60s startup)

Start simple; add ML only if heuristics misroute too often. Because the escalation net catches correctness errors, the router only needs to be *good*, not perfect.

- **Tier A — heuristics (default):** code fences / `def`/`function` ⇒ code (debug vs generate by "fix/bug" vs "write/implement"); "summari[sz]e" ⇒ summarisation; "sentiment/tone" ⇒ sentiment; digits + operators / "how much/percent" ⇒ math; "who/where/when/list entities" ⇒ NER; constraint-puzzle phrasing ("if… then", "exactly one", ordering) ⇒ logic; else ⇒ factual.
- **Tier B — tiny embedding classifier (optional, likely skip):** a small ONNX sentence-embedder (~20–80 MB) + logistic regression trained on synthetic per-category examples, used only for low-confidence heuristic cases. **No local model is required at all** — the moderator confirmed local models are dev-only and don't count, and pure Python heuristics keep the image tiny and startup instant. Add Tier B only if measured heuristic misroute rate is costing real tokens; otherwise ship Tier A alone. Never a local *LLM* (too slow on CPU, and it can't author answers anyway per the bright-line invariant).

**Difficulty / model choice:** prefer **cheap-first + escalate** over a difficulty predictor — it's self-correcting. But note: escalation = 2 calls = *more* tokens than one strong call. So for any category where the cheap model's gate-pass rate is low (measured on our dev set), **go straight to the strong model.** The cheap-first-vs-strong-first decision per category is **tuned empirically at kickoff**, not guessed.

---

## 5. Per-category strategy (initial hypotheses — tune at kickoff)

| Category | Model (start) | Output policy | max_tokens | Notes |
|---|---|---|---|---|
| Factual | `gemma-4-26b-a4b-it` | 1–3 sentence answer, no preamble | low | Risk: too terse fails judge — tune length up to margin. |
| Math | `minimax-m3` | brief work, **final answer only** on last line | low-med | Reasoning tokens are counted — instruct minimal steps. Local plausibility check. |
| Sentiment | `gemma-4-26b-a4b-it` | `label — <one clause>` | very low | Justification required but keep to a clause. |
| Summarisation | `gemma-4-31b-it-nvfp4` | obey the stated format/length exactly | task-bounded | Locally truncate/enforce the length constraint (free). |
| NER | `gemma-4-26b-a4b-it` | compact structured list | low | Choose plain compact format vs JSON by token count. |
| Code debugging | `kimi-k2p7-code` | corrected code + 1-line cause | med | Validate the fix `ast.parse`s. |
| Logic | `minimax-m3` | minimal reasoning, final answer | low-med | Hardest accuracy/token trade; escalate on validation fail. |
| Code generation | `kimi-k2p7-code` | function only, no prose | med | Validate it parses; optionally run against spec-derived asserts locally. |

**Escalation ladder (per task):**
`route → single terse call to cheapest capable model → local validate → (only if fail) retry with stronger model or richer prompt.` By design, escalation is rare — most tasks resolve in one call.

---

## 6. Token-minimization tactics (the race)

- **Terse/absent system prompt**; fold instructions into a compact per-category template.
- **Zero-shot default**; add *one* tiny few-shot example only where it measurably lifts the gate.
- **`stop` sequences** (e.g., `\n\n`, closing fence) to end generation at the answer.
- **Tight `max_tokens`** per category; never let the model ramble.
- **Local output coercion** (free) instead of a re-ask: extract the answer, strip preambles, enforce length. Turns many "wrong-format" fails into passes at zero cost.
- **Prefer non-reasoning models with terse output**; use a reasoning model only where needed to clear the gate (its internal reasoning tokens are *counted*).
- **❓ Batching (advanced, measure carefully):** for tiny high-overhead categories (sentiment, NER) where instruction overhead ≫ task content, sending several tasks per call amortizes the overhead → big input-token savings. Risk: one derailed task fails several answers (and the gate is unforgiving). Batch **only within a category, only where accuracy holds** on the dev set.
- **❓ Prompt-prefix caching:** if the proxy discounts cached shared prefixes, identical templates help; if it counts raw tokens regardless, short prompts still win. Confirm at kickoff — either way, short is robust.
- **Tokenizer arbitrage (2nd-order):** measure which model's tokenizer is cheapest for the *same* answer text; among Gemmas it's identical.

---

## 7. Reliability & the 10-minute guard (non-negotiable)

- **Global deadline** (e.g., 9m30s): once hit, stop launching new calls, fill any unfinished task with best-effort/placeholder answers, **write valid `/output/results.json`, exit 0.** A partial-but-valid result beats a timeout that produces malformed/missing output (which scores zero).
- **Per-task try/except** ⇒ never let one task crash the run; a failed task still gets a (possibly empty) answer entry so the JSON stays well-formed and complete.
- **Validate our own output JSON** against the required schema before writing.
- **Exit code discipline:** 0 on success; only non-zero for truly unrecoverable startup failures.

---

## 8. Evaluation & tuning harness — our real edge

Most teams will hand-tune prompts. We win by **measuring**. Build this early; it turns the 5-day sprint into a rapid optimize loop.

- **Synthetic dev set:** ~30–50 tasks per category with known-good answers, plus adversarial/edge variants (since eval uses *unseen variants*, we must generalize).
- **Our own LLM-Judge:** a proxy for the hidden accuracy gate (a strong model) that scores answers for intent-match. Lets us estimate gate-clearing locally.
- **Strategy sweep:** for each category × model × output-policy × max_tokens, record **(judge score, tokens)**. Pick the **token-minimal strategy that clears the gate with margin.** Since the exact threshold is ❓, aim for comfortably-high accuracy at minimal tokens and keep a dial.
- **Two operating modes:** `safe` (more tokens, higher accuracy — guarantees we're on the leaderboard) and `lean` (pushes tokens down). Start safe, then push lean while watching the leaderboard.

---

## 9. Submission / leaderboard loop (10 submissions/hour/team)

1. **First submission = `safe` baseline** — confirm we clear the gate and appear on the leaderboard end-to-end (I/O contract, Docker, env, JSON all correct).
2. **Iterate down:** each cycle, lower tokens on the categories with the most margin; re-submit; verify we're still on the leaderboard. Keep the last known-good config pinned.
3. Use the generous rate limit for real A/B tests on the true judge — but always retreat to the pinned config if a lean change drops us off the board.

---

## 10. Docker & ops requirements (bake in from day 1)

- **`linux/amd64` manifest** (build `--platform linux/amd64` if ever on ARM). Wrong arch ⇒ fails to pull ⇒ zero.
- **Slim base** (e.g., `python:3.12-slim`); image well under **10 GB** (we're MBs unless we bundle the optional embedder — keep it small).
- **Reads env only:** `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`. **No `.env` in the image, no hardcoded keys/models.** Route *all* calls through `FIREWORKS_BASE_URL` (bypassing it ⇒ zero tokens recorded ⇒ invalid).
- **Startup < 60s**, **per-request < 30s**, English-only outputs.
- Public registry (GHCR/Docker Hub), **MIT-licensed public repo.**
- I/O exactly: read `/input/tasks.json` (`[{task_id, prompt}]`), write `/output/results.json` (`[{task_id, answer}]`).

---

## 11. Repo structure

```
track1-agent/
├── Dockerfile                 # linux/amd64, slim, env-only
├── LICENSE                    # MIT
├── README.md
├── pyproject.toml
├── src/
│   ├── main.py                # orchestrator + deadline guard + JSON writer
│   ├── router.py              # local classification (heuristics + optional ONNX)
│   ├── strategies.py          # per-category prompt/model/limits
│   ├── fireworks.py           # OpenAI-compatible client → FIREWORKS_BASE_URL
│   ├── validate.py            # local validation + coercion
│   └── telemetry.py           # token accounting for tuning
├── eval/
│   ├── devset/                # synthetic tasks per category (+ gold answers)
│   ├── judge.py               # our proxy LLM-Judge
│   ├── sweep.py               # strategy × model × policy → (score, tokens)
│   └── report.py              # leaderboard-style summary
└── tests/                     # I/O contract, JSON schema, deadline guard
```

---

## 12. Timeline (Jul 6 → Jul 11)

- **Day 0 (Jul 6, kickoff):** confirm ❓ items (CPU-only? gate threshold? token accounting; caching). Register/approval done, Fireworks dev key ready, repo + MIT + Dockerfile skeleton pushed. Smoke-test each of the 5 models through the proxy.
- **Day 1:** end-to-end skeleton passing the I/O contract with a dumb single-model baseline; first Docker build/push; **first `safe` submission on the leaderboard.**
- **Day 2:** router + per-category strategies + validation/coercion; build the eval harness + synthetic dev set + proxy judge.
- **Day 3:** run strategy sweeps; lock in token-minimal per-category strategies; begin lean submissions.
- **Day 4:** optimization pass (batching experiments, escalation tuning, tokenizer arbitrage); harden the deadline guard; robustness/edge cases.
- **Day 5 (Jul 11):** final lean push with `safe` fallback pinned; freeze; **presentation package** (video + slides + write-up); submit well before 9:30 PM IST.

---

## 13. ❓ Confirm at kickoff / on Discord

- Is final scoring really **CPU-only**? (Affects how heavy a local router we can afford.)
- **Accuracy gate threshold** value and whether it's per-task or aggregate.
- Exact **token accounting**: prompt+completion summed? cached-prefix discount? per-model weighting (spec says raw count — confirm)?
- Any **per-call rate limits** on the proxy (vs. the 10-submissions/hour team limit).
- Model context limits / pricing (for latency planning within the 10-min budget).

---

## 14. Prize layer (parallel workstream)

The leaderboard determines Track-1 ranking, but the **$10k cash goes to top overall projects** judged on *Application of Technology, Presentation, Business Value, Originality*. So budget real time for a **crisp demo video + slides + narrative**: frame the agent as a **cost-optimizing inference router** with obvious business value (LLM cost reduction), show the token savings vs. a naive baseline as the money shot, and highlight that it runs efficiently on AMD-served models. Presentation is where many strong builds lose — don't leave it to the last hour.

---

### One-paragraph summary

Build a containerized general-purpose agent that **routes every task locally (free) into one of 8 category strategies**, answers it with **a single, tightly-constrained Fireworks call** to the cheapest model that clears the accuracy gate, **validates and coerces the output locally**, and **escalates only when a local check predicts failure**. A deadline guard guarantees valid output within 10 minutes. An **eval harness with a proxy judge** lets us empirically drive tokens down while staying above the gate, and we iterate on the real leaderboard from a pinned `safe` baseline toward a `lean` optimum.
