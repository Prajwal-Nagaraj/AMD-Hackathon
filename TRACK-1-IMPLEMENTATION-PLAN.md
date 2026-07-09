# Track 1 — Token-Efficient General-Purpose Agent: Implementation Plan

> Our entry for AMD Developer Hackathon ACT II, Track 1.
> Source of truth: `Participant Guide_ AMD Developer Hackathon (ACT II).md`. Re-confirm anything marked ❓ at kickoff / on Discord.
> Build window: **Jul 6 → Jul 11, 2026 (submissions close Jul 11, 9:30 PM IST).**

> **⚡ STRATEGY UPDATE (Jul 9 2026 — official organizer clarification, inverts the old doctrine).**
> A **local model bundled in our image may author answers, and those answers count fully toward accuracy.** The token score records **only Fireworks-routed tokens**, so **a task answered correctly by a local model costs zero tokens — the best possible ranking outcome.** The previous "every answer must be produced by a Fireworks call" invariant is now the *opposite* of optimal. New doctrine: **answer as many tasks as we reliably can with a bundled 2–3B local model at zero cost; call Fireworks only where the small model can't clear the gate.**
> Newly-confirmed facts: **accuracy gate = 80 %**, **19 fixed tasks** (scores are n/19 ⇒ ~3-task error budget), grading env is **4 GB RAM / 2 vCPU / no GPU / no model runtime pre-installed**, and the **judge is not perfectly deterministic** (keep margin). Gemma on Fireworks is **on-demand — deploy it first** (a 404 = not deployed, not banned).

---

## 1. What we're actually building (and what "winning" means)

Track 1 is a **general-purpose agent** that answers arbitrary natural-language tasks spanning **8 capability categories** across **19 fixed tasks**, spending **as few Fireworks tokens as possible** — ideally zero.

**The 8 categories (evaluated on all of them, ~2–3 tasks each across the 19):**

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

**Scoring is two-stage — and the token metric is the pivot:**

1. **Accuracy gate — 80 % (confirmed).** An LLM-Judge scores each answer against expected intent. Below 80 % ⇒ **excluded from the leaderboard entirely** (a hard cutoff). With **19 tasks** every score is **n/19**, so **16/19 = 84.2 % is the lowest passing rung** and **15/19 = 78.9 % already fails.** Practical **error budget: 3 wrong answers, and the judge is nondeterministic** ⇒ aim for **~17–18/19 (≈90–95 %)** to keep real margin, wherever the tokens allow.
2. **Token efficiency — Fireworks tokens only, ascending.** Everyone who clears the gate is ranked by **total tokens recorded by the judging proxy**, and **the proxy only records tokens routed through `FIREWORKS_BASE_URL`.** **Local answers contribute zero.** Raw prompt + completion count, not price-weighted.

> **The whole strategy in one line:** *Clear the 80 % gate with margin, then win by answering the maximum number of tasks locally at **zero Fireworks tokens** — reserving Fireworks for only the tasks a small local model can't reliably clear.* The theoretical optimum is **0 Fireworks tokens** (all 19 answered locally and still ≥ 80 %); the realistic optimum is **as-local-as-possible with a lean Fireworks fallback** for the hard categories.

**Two token sources, only one is counted:**

- **Local (bundled 2–3B 4-bit model)** — CPU inference inside the container. **Zero counted tokens.** Bounded by a small model's quality, 4 GB RAM, 2 vCPU, and the 10-minute wall-clock.
- **Fireworks (`ALLOWED_MODELS` via `FIREWORKS_BASE_URL`)** — every token here is counted and hurts our rank. Fast (network-bound, parallelizable) and high-quality. Use only where local can't clear the gate. Governed by `TRACK-1-MODEL-RESEARCH.md` (Gemma-first).

**Allowed *Fireworks* models (read from `ALLOWED_MODELS` at runtime — never hardcode):** `minimax-m3`, `kimi-k2p7-code`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it-nvfp4`. The three Gemma variants share a tokenizer (identical token count for identical text). **Local models are NOT restricted to this list** — locally we may bundle any open-weight model.

---

## 2. Guiding principles (the new local ⇄ Fireworks tension)

Every design choice now trades three things off: **clearing the gate** (wants quality), **ranking well** (wants zero Fireworks tokens ⇒ answer locally), and **finishing in 10 minutes** (local CPU inference is slow and serial).

> **Bright-line invariants (never cross):**
> 1. **Never emit malformed or missing output.** Malformed JSON / missing answers score zero for that task. A deadline guard guarantees a valid `/output/results.json` and exit 0.
> 2. **Everything we send to Fireworks must be an `ALLOWED_MODELS` model routed through `FIREWORKS_BASE_URL`.** An out-of-list model is a `MODEL_VIOLATION`; a call bypassing the base URL isn't metered and effectively doesn't count. (Local models are *allowed* — they are not a violation.)
> 3. **Never hardcode or cache answers to specific inputs.** A bundled *model* is fine; a bundled *answer key* is not. Evaluation uses unseen variants.

1. **Local-first.** For each task, prefer a local answer at **zero Fireworks cost**. Escalate to Fireworks **only** when a local check predicts the local answer would miss the gate, or when local inference is too slow/uncertain for that task type.
2. **Answer authorship is now allowed locally — exploit it ruthlessly.** The local model may *author* summaries, labels, entities, simple code, easy arithmetic, and short factual answers. (Old plan forbade this; the clarification reverses it.) Local routing, validation, and formatting remain free as before.
3. **One inference per task by default** (local *or* Fireworks, not both). No self-consistency voting or debate. A second call happens only via *escalation* when a local validator predicts a gate failure.
4. **Minimize Fireworks tokens on the tasks that do escalate:** terse/near-empty system prompt, zero-shot, tight per-category `max_tokens`, `stop` sequences, no CoT scaffolding, Gemma-first, suppress reasoning traces. (Full toolbox: `TRACK-1-MODEL-RESEARCH.md`.)
5. **`temperature = 0` (greedy) everywhere**, local and Fireworks — deterministic, reproducible, no multi-sampling.
6. **Budget wall-clock like a scarce resource.** Local CPU inference on 2 vCPU is slow and effectively serial; Fireworks calls are fast and parallel. Under time pressure, **spend tokens to avoid a `TIMEOUT`** (a timeout produces no/partial output = far worse than a few tokens).
7. **Route to the minimum-sufficient answerer per category** — local where a 2–3B model reliably clears the gate; the cheapest Fireworks model that clears it otherwise; a reasoning model only where genuinely required.

---

## 3. System architecture

```
/input/tasks.json ──▶ main.py (orchestrator)
                         │
                         ├─▶ router.py        (LOCAL, free): category + difficulty + local-vs-Fireworks decision
                         │
                         ├─▶ local_llm.py     (LOCAL, free tokens): bundled 2–3B Q4 model on CPU (llama.cpp)
                         │        │
                         │        └─▶ validate.py ─(local answer looks good)─▶ accept (0 tokens)
                         │                          └─(fail / low-confidence / too slow)─▶ escalate ▼
                         │
                         ├─▶ strategies.py    per-category: prompt template, model, max_tokens, stop
                         │
                         ├─▶ fireworks.py     OpenAI-compatible client → FIREWORKS_BASE_URL (metered)  ◀── escalation target
                         │
                         └─▶ writer           format → /output/results.json ──▶ exit 0
```

**Components:**

- **`main.py`** — load tasks; schedule work under a **global deadline** and a **time-vs-tokens policy** (below); write results; exit 0. Always writes valid JSON even on partial completion.
- **`router.py`** — **local, CPU-only.** Classifies each task into one of the 8 categories, estimates difficulty, and makes the **local-vs-Fireworks decision** from a measured per-category policy table (§5). Misrouting only wastes work; validation+escalation catches correctness.
- **`local_llm.py`** — **NEW. The bundled local model** (2–3B, 4-bit GGUF) run on CPU via llama.cpp (`llama-cpp-python`), loaded **once** at startup. Produces zero-counted-token answers for the categories where it's reliable. Hard-capped `max_tokens` and a per-call wall-clock guard keep each inference under the 30 s/request rule.
- **`strategies.py`** — per-category spec for **both** paths: local prompt/limits and (for escalation) the Fireworks model, template, `max_tokens`, `stop`, and whether justification/structure is required.
- **`fireworks.py`** — OpenAI-compatible client → `FIREWORKS_BASE_URL`, keyed by `FIREWORKS_API_KEY`, models from `ALLOWED_MODELS`. Retries with backoff. Records our own token telemetry for tuning.
- **`validate.py`** — **local, free.** Per-category checks (JSON parses? number present? code `ast.parse`s? within length constraint?) + **coercion** (strip preambles, extract the answer, enforce length). Doubles as the **escalation trigger**: a local answer that fails validation is re-tried on Fireworks.

**Concurrency model (changed by the local tier):**

- **Local inferences are effectively serial** (2 vCPU; one llama.cpp call saturates both cores). Run them one at a time, cheapest/fastest categories first.
- **Fireworks escalations run concurrently** (network-bound, bounded async pool).
- Keep each task < 30 s wall-clock (local cap enforced in `local_llm.py`). Respect the 10-min *total* budget via the deadline guard.

---

## 4. Local router + the local-vs-Fireworks decision (free, CPU-only)

The router does two jobs, both free: **classify** the task, and **decide who answers it**.

- **Tier A — heuristics (default classifier):** code fences / `def`/`function` ⇒ code (debug vs generate by "fix/bug" vs "write/implement"); "summari[sz]e" ⇒ summarisation; "sentiment/tone" ⇒ sentiment; digits + operators / "how much/percent" ⇒ math; "who/where/when/list entities" ⇒ NER; constraint-puzzle phrasing ⇒ logic; else ⇒ factual.
- **Tier B — tiny embedding classifier (optional):** a small ONNX sentence-embedder + logistic regression for low-confidence heuristic cases. Likely unnecessary; add only if measured misroute rate matters.
- **Local-vs-Fireworks policy:** a **measured** per-category table (built by the eval harness, §8) says, for each category, whether the bundled local model clears the gate reliably enough to answer at zero cost, or whether we escalate straight to Fireworks. Low-confidence *within* a category (e.g. an unusually long summarisation input, or a math problem with many steps) also biases toward Fireworks.

**Cheap-first + local-validate + escalate** replaces the old cheap-model ladder:
`route → local answer (0 tokens) → local validate → (only if fail/low-confidence) → Fireworks call (Gemma-first) → local validate.`
By design most tasks resolve locally in one pass; Fireworks is the exception, not the rule.

---

## 5. Per-category strategy (initial hypotheses — tune at kickoff)

The key new column is **"answer locally?"** — driven by whether a 2–3B model clears the 80 % gate for that category on our proxy judge. Start conservative (escalate the hard reasoning/code), then push categories local as measurements justify it.

| Category | Default answerer | If escalated (Fireworks) | Output policy | Notes |
|---|---|---|---|---|
| Factual | **Local** (fall back to Fireworks on low confidence) | `gemma-4-26b-a4b-it` → `gemma-4-31b-it` | 1–3 sentences, no preamble | Small models hallucinate facts — watch accuracy; escalate obscure/long-tail questions. |
| Math | **Fireworks** first (small models are weak at multi-step) | `gemma-4-31b-it` (thinking-off) → `minimax-m3` (thinking-on) | brief work, **final answer only** on last line | Try local for *simple* arithmetic; escalate multi-step. Local plausibility check. |
| Sentiment | **Local** | `gemma-4-26b-a4b-it` | `label — <one clause>` | Ideal local category: short, easy, high-volume. |
| Summarisation | **Local** (short inputs); Fireworks for long inputs | `gemma-4-31b-it-nvfp4` | obey stated format/length exactly | CPU prompt-eval of a long passage is slow — route long inputs to Fireworks. Enforce length locally (free). |
| NER | **Local** | `gemma-4-26b-a4b-it` | compact structured list | Ideal local category. Choose plain compact format vs JSON by token count *only if escalated*. |
| Code debugging | **Fireworks** first | `gemma-4-31b-it` → `kimi-k2p7-code` | corrected code + 1-line cause | Small models miss subtle bugs; escalate. Validate the fix `ast.parse`s locally. |
| Logic | **Fireworks** first | `gemma-4-31b-it` → `minimax-m3` | minimal reasoning, final answer | Hardest for a 2–3B model — usually escalate. Condense CoT carefully. |
| Code generation | **Local** for simple specs; Fireworks for complex | `gemma-4-31b-it` → `kimi-k2p7-code` | function only, no prose | Validate it parses (and, where possible, run spec-derived asserts) locally — free. |

**Escalation ladder (per task):**
`route → local answer → local validate → (only if fail) Fireworks: cheapest capable model → local validate → (only if fail) stronger model.` Escalation is rare by design.

> **Calibration is everything.** With a 3-task error budget and a nondeterministic judge, answer a category locally **only** where its measured local gate-pass rate is high enough that the *aggregate* stays ≥ ~90 %. When in doubt, escalate — a few tokens are cheaper than falling below 80 %.

---

## 6. Token-minimization tactics (only the escalated tasks cost tokens)

These apply to the **Fireworks fallback path** — the minority of tasks the local model can't clear. Full detail + measurements in `TRACK-1-MODEL-RESEARCH.md`.

- **Answer locally whenever the gate allows — the single biggest token win is a task that never hits Fireworks at all.**
- **Gemma-first** on escalation: non-reasoning, terse, and the densest tokenizer (262k vocab) ⇒ fewest counted tokens. Reserve `minimax-m3` (thinking togglable off) / `kimi-k2p7-code` for tasks only they clear.
- **Terse/absent system prompt**; fold instructions into a compact per-category template.
- **Zero-shot default**; add one tiny few-shot example only where it measurably lifts the gate.
- **`stop` sequences** (`\n\n`, closing fence, first period) + **tight `max_tokens`** per category.
- **Local output coercion** (free) instead of a re-ask: extract the answer, strip preambles, enforce length.
- **Suppress reasoning traces** where the gate allows (avoids the 3–5× CoT tax); don't over-compress math/logic on weak models.
- **❓ Batching (advanced):** for tiny high-overhead escalated categories, batch within a category where accuracy holds. Risk: one derailed task fails several answers.
- **Tokenizer arbitrage (2nd-order):** among Gemmas identical; measure minimax/kimi at kickoff.

---

## 7. Reliability, the 10-minute guard, and the local-model constraints (non-negotiable)

- **Startup < 60 s including model load.** Loading a 2–3B Q4 GGUF (~1.5–2 GB) off local disk is a few seconds — well within budget. Load **once** at process start; never per-task.
- **Grading env is 4 GB RAM / 2 vCPU / no GPU.** A **2–3B 4-bit** model + short-context KV cache + our agent code fits; **a 7B 4-bit model fills RAM and leaves no room** — do not exceed 3B. **No model runtime is pre-installed** ⇒ we bundle the weights *and* the CPU runtime (llama.cpp / `llama-cpp-python`) in the image.
- **Per-request < 30 s.** Local CPU inference is the risk here: hard-cap local `max_tokens` and add a per-call wall-clock guard; if a local inference would blow 30 s (e.g. a long summarisation input), route it to Fireworks instead.
- **Global deadline (e.g. 9m30s) with a time-vs-tokens policy:** track elapsed time; if we're falling behind (local inference too slow to finish all tasks), **stop starting new local inferences and push remaining tasks to fast parallel Fireworks calls.** Paying tokens beats a `TIMEOUT`. Once the hard deadline hits, fill any unfinished task with a best-effort answer, **write valid `/output/results.json`, exit 0.**
- **Per-task try/except** ⇒ one task's failure (local *or* Fireworks) never crashes the run; it still gets a (possibly escalated or placeholder) answer so the JSON stays complete.
- **Validate our own output JSON** against the required schema before writing (avoids `INVALID_RESULTS_SCHEMA`).
- **Exit code discipline:** 0 on success; non-zero only for truly unrecoverable startup failures (avoids `RUNTIME_ERROR`).

**Failure-status map (from the participant guide) → our mitigation:**

| Status | Our guard |
|---|---|
| `PULL_ERROR` | public registry + `linux/amd64` manifest (§10) |
| `RUNTIME_ERROR` | per-task try/except, exit 0, deadline guard |
| `TIMEOUT` | time-vs-tokens policy → Fireworks fallback under pressure |
| `INVALID_RESULTS_SCHEMA` | schema-validate output before writing |
| `MODEL_VIOLATION` | only `ALLOWED_MODELS` via `FIREWORKS_BASE_URL`; local model is legal |
| `IMAGE_TOO_LARGE` | slim base + 4-bit weights, keep well under 10 GB (§10) |
| `ACCURACY_GATE_FAILED` | conservative local-vs-Fireworks calibration + margin |

---

## 8. Evaluation & tuning harness — our real edge

Most teams will hand-tune. We win by **measuring** — and the new central measurement is **per-category local gate-pass rate**, which decides what we can answer for free.

- **Synthetic dev set:** ~30–50 tasks per category with known-good answers + adversarial/edge variants (eval uses unseen variants, so we must generalize). Keep the total shaped like the real 19-task, 8-category mix.
- **Our own LLM-Judge:** a proxy for the hidden 80 % gate (a strong model) scoring intent-match. Lets us estimate gate-clearing locally.
- **Local-vs-Fireworks calibration (new):** run each category through **the bundled local model** and record **(judge pass-rate, latency)**. Answer locally only where the pass-rate keeps the aggregate comfortably ≥ 90 %; escalate the rest. This produces the §5 policy table empirically, not by guess.
- **Strategy sweep (escalated path):** for each escalated category × Fireworks model × output-policy × `max_tokens`, record **(judge score, tokens)**; pick the token-minimal strategy that clears the gate with margin.
- **Wall-clock model:** measure real local tokens/sec on a 2-vCPU box to project whether N local inferences fit in 10 min; tune how many tasks we dare answer locally vs. must escalate for time.
- **Two operating modes:** `safe` (more Fireworks, higher accuracy — guarantees the leaderboard) and `lean` (max local, tokens toward zero). Start `safe`, push `lean` while watching the leaderboard.

---

## 9. Submission / leaderboard loop (10 submissions/hour/team)

1. **First submission = `safe` baseline (pure Fireworks, lean prompts).** Simplest robust config: confirm we clear the 80 % gate and appear on the leaderboard end-to-end (I/O contract, Docker, env, JSON). No local model yet — de-risk the pipeline first.
2. **Layer in local answering:** enable local answers for the categories with proven local pass-rates (sentiment, NER, summarisation, easy factual first). Re-submit; verify accuracy holds and tokens drop. Keep the last known-good config pinned.
3. **Push toward zero tokens:** move more categories local as calibration justifies; each cycle lower Fireworks reliance on the categories with the most margin. Always retreat to the pinned config if a lean change drops us below the gate.
4. Use the download-counter tip to confirm the graders have actually pulled each new image before drawing conclusions from a score.

---

## 10. Docker & ops requirements (bake in from day 1)

- **`linux/amd64` manifest** (build `--platform linux/amd64` if ever on ARM). Wrong arch ⇒ `PULL_ERROR` ⇒ zero.
- **Bundle the local model + runtime:** the 2–3B 4-bit GGUF weights (~1.5–2 GB) **and** `llama-cpp-python` (no Ollama in the env). Still comfortably under **10 GB compressed** — but watch the total (avoid `IMAGE_TOO_LARGE`).
- **Slim base** (e.g. `python:3.12-slim`); prune build caches. Ensure the CPU wheel of `llama-cpp-python` (no CUDA) to keep the image small and portable.
- **Fits 4 GB RAM / 2 vCPU:** never exceed a 3B model; keep contexts short; free the model's KV cache between tasks if needed.
- **Reads env only:** `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`. **No `.env` in the image, no hardcoded keys/models/answers.** Route *all Fireworks* calls through `FIREWORKS_BASE_URL`.
- **Startup < 60 s** (incl. model load), **per-request < 30 s**, English-only outputs.
- Public registry (GHCR/Docker Hub), **MIT-licensed public repo.**
- I/O exactly: read `/input/tasks.json` (`[{task_id, prompt}]`), write `/output/results.json` (`[{task_id, answer}]`).

---

## 11. Repo structure

```
track1-agent/
├── Dockerfile                 # linux/amd64, slim, env-only, bundles local model + llama.cpp
├── LICENSE                    # MIT
├── README.md
├── pyproject.toml
├── models/
│   └── local-2-3b-q4.gguf     # bundled local weights (4-bit)
├── src/
│   ├── main.py                # orchestrator + deadline/time-vs-tokens guard + JSON writer
│   ├── router.py              # local classification + local-vs-Fireworks decision
│   ├── local_llm.py           # bundled 2–3B Q4 model on CPU (llama.cpp), loaded once  ← NEW
│   ├── strategies.py          # per-category local + Fireworks prompt/model/limits
│   ├── fireworks.py           # OpenAI-compatible client → FIREWORKS_BASE_URL (escalation)
│   ├── validate.py            # local validation + coercion + escalation trigger
│   └── telemetry.py           # token accounting for tuning
├── eval/
│   ├── devset/                # synthetic tasks per category (+ gold answers)
│   ├── judge.py               # our proxy LLM-Judge
│   ├── calibrate.py           # per-category LOCAL gate-pass rate → policy table  ← NEW
│   ├── sweep.py               # escalated: strategy × Fireworks model × policy → (score, tokens)
│   └── report.py              # leaderboard-style summary
└── tests/                     # I/O contract, JSON schema, deadline guard, router
```

---

## 12. Timeline (Jul 6 → Jul 11)

- **Day 0 (Jul 6, kickoff):** confirm ❓ items (§13). **Deploy the Gemma models on Fireworks (on-demand — a 404 = not deployed).** Register/approval done, Fireworks dev key ready, repo + MIT + Dockerfile skeleton. Smoke-test each of the 5 models through the proxy. Pick + download the local 2–3B Q4 model.
- **Day 1:** end-to-end skeleton passing the I/O contract with a **pure-Fireworks `safe` baseline**; first Docker build/push; **first submission on the leaderboard** using the practice tasks to validate I/O first.
- **Day 2:** bundle the local model + `llama.cpp`; wire the local tier + router local-vs-Fireworks decision + validation/coercion; build the eval harness + synthetic dev set + proxy judge.
- **Day 3:** **local calibration sweep** — measure per-category local gate-pass rate + latency; lock the policy table; move the safe local categories on; begin lean submissions.
- **Day 4:** push more categories local; strategy sweeps on the escalated path; harden the time-vs-tokens deadline guard; robustness/edge cases; verify image size + RAM fit on a 2-vCPU/4 GB box.
- **Day 5 (Jul 11):** final lean push with the `safe` config pinned; freeze; **presentation package** (video + slides + write-up); submit well before 9:30 PM IST.

---

## 13. ❓ Confirm at kickoff / on Discord

Resolved by the Jul 9 clarification: **gate = 80 %**, **19 tasks (n/19)**, **grading env 4 GB / 2 vCPU / no GPU / no runtime pre-installed**, **local models legal and count as zero Fireworks tokens**, **judge is nondeterministic**. Still to confirm:

- **Local-model runtime works in the grading env:** does `llama-cpp-python` (CPU) run cleanly in the sandbox, and can a 2–3B Q4 model finish enough tasks within 10 min on 2 vCPU? (Measure real tokens/sec.)
- **Any restriction on *which* local model** we bundle? (Clarification says any bundled local model is fine; sanity-check no hidden constraint.)
- **Does the judging proxy count reasoning/CoT tokens?** (Assumed yes; governs Gemma-first on the escalated path.)
- **Do served `gemma-4-*-it` endpoints emit a `<think>` trace by default?** Force non-thinking if so.
- **Can `minimax-m3` thinking be toggled off** via the OpenAI-compatible API (param name)? Default it off.
- **Gemma deployment:** confirm the Gemma endpoints are deployed (app.fireworks.ai/models) before relying on the Gemma-first fallback.
- **Tokenizer sanity check** on the real proxy: send one fixed answer string through all 5 models; record counted tokens.
- Any **per-call rate limits** on the proxy (vs. the 10-submissions/hour team limit).

---

## 14. Prize layer (parallel workstream)

The leaderboard determines Track-1 ranking, but the **$10k cash goes to top overall projects** judged on *Application of Technology, Presentation, Business Value, Originality*. Budget real time for a **crisp demo video + slides + narrative**: the story is now even stronger — frame the agent as a **hybrid cost-optimizing inference router that answers most tasks on a tiny local model (zero API cost) and reaches for a frontier API only when it must.** The money shot: tokens/cost vs. a naive all-API baseline, plus the accuracy held above the gate. Highlight it runs on a 2-vCPU/4 GB CPU box yet stays accurate — obvious business value (LLM cost reduction at the edge). Don't leave presentation to the last hour.

---

### One-paragraph summary

Build a containerized general-purpose agent that **routes every task locally (free)**, **answers as many as it reliably can with a bundled 2–3B 4-bit local model at zero Fireworks-token cost**, and **escalates to a single, tightly-constrained Gemma-first Fireworks call only for the tasks a small model can't clear** — validating and coercing every output locally. A time-aware deadline guard guarantees valid output within 10 minutes and spends tokens rather than time out. An **eval harness with a proxy judge** calibrates, per category, what we can answer locally while staying comfortably above the 80 % (16/19) gate, and we iterate on the real leaderboard from a pinned pure-Fireworks `safe` baseline toward an as-local-as-possible, near-zero-token `lean` optimum.
