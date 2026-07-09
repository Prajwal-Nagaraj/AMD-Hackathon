# Track 1 — Model Capability & Token-Minimization Research

> Companion to `TRACK-1-IMPLEMENTATION-PLAN.md`. Produced by a fan-out deep-research pass (26 sources, 212 extracted claims, adversarial verification). Dates on the underlying models are 2026 — treat specific benchmark numbers as **directional**, and re-confirm the `ALLOWED_MODELS` strings + token accounting at kickoff.
>
> **Confidence key:** ✅ well-corroborated across ≥2 independent sources · ⚠️ single-source / marketing / disputed · ❓ must confirm at kickoff.

> **⚡ SCOPE UPDATE (Jul 9 2026 — official clarification).** A **bundled local model may now author answers, counted toward accuracy at zero Fireworks-token cost** (see the implementation plan's local-first doctrine). So **this document now governs the *Fireworks fallback path* — the minority of tasks a small local model can't clear.** Everything below (Gemma-first, the reasoning-token trap, tokenizer arbitrage, terse-output tactics) is how to make those *escalated* calls as cheap as possible. **The biggest token win is upstream of all of it: a task answered locally never spends a Fireworks token.** Two additions triggered by the clarification: **§0.5 (choosing the local model)** and the **Gemma-is-on-demand caveat** (deploy it before Gemma-first means anything).

---

## 0.5 Choosing the LOCAL model (the zero-token answerer)

The local model is bundled in our image and run on CPU; **it is NOT restricted to `ALLOWED_MODELS`** (that list only governs Fireworks calls). Pick the strongest **2–3B, 4-bit** instruct model — the grading env is **4 GB RAM / 2 vCPU / no GPU**, so **≤ 3B** (a 7B 4-bit fills RAM with no room for agent code), served via **llama.cpp / `llama-cpp-python`** (no runtime is pre-installed).

| Candidate (2–3B, Q4) | Why consider | Watch-outs |
|---|---|---|
| **Qwen2.5-3B-Instruct** | Strong all-rounder for its size — good instruction-following, math, and code; solid default | Confirm RAM headroom at Q4_K_M + short KV cache |
| **Llama-3.2-3B-Instruct** | Good general/factual + summarisation; broad tooling support | Weaker at math/code than Qwen for size |
| **Gemma-2-2B-it** | Smallest/fastest ⇒ most tasks fit the 10-min wall-clock; terse | 2B ceiling — escalate more reasoning/code |
| **Phi-3.5-mini (3.8B)** | Punchy reasoning/code for size | **~3.8B — borderline on 4 GB RAM**; test carefully, may not fit |

**Selection method (not by vibes):** run the eval harness's **local calibration** (`eval/calibrate.py`) — per category, record **(proxy-judge pass-rate, latency on 2 vCPU)** for each candidate. Choose the model that maximizes the number of categories we can answer locally while keeping the aggregate comfortably ≥ 90 % (margin over the 80 % / 16-of-19 gate) **and** finishing 19 tasks within 10 min. Q4_K_M is the usual quality/size sweet spot; the weights are ~1.5–2 GB (3B) — well under the 10 GB image cap even with the runtime bundled.

> **Division of labour:** the *local* model wants breadth + speed + instruction-following (it handles sentiment, NER, summarisation, easy factual, simple code). The *Fireworks* models below are the escalation targets for what it can't clear (hard math/logic, complex debugging, obscure factual). Optimize each independently.

---

## 0. TL;DR — the three findings that change the plan

1. **The reasoning-token trap is the whole game.** `minimax-m3` and `kimi-k2p7-code` are **reasoning models that emit counted chain-of-thought tokens by default**; the Gemma variants are **terse / non-reasoning by default**. In a *raw-token* race, a reasoning model's internal `<think>` trace can be **3–5× the answer's tokens** (Kimi Thinking) — a silent, massive tax. **Default to Gemma; only escalate to a reasoning model when the gate genuinely needs it, and suppress thinking when you do.** ✅

2. **Gemma also wins the tokenizer arbitrage.** Gemma's tokenizer has a **262k vocabulary** vs MiniMax **~200k** and Kimi **~160k**. Larger vocab ⇒ **the same answer text costs fewer counted tokens on Gemma.** So Gemma is cheaper on *three* independent axes: non-reasoning, terse, and most token-dense encoding. The three Gemmas share this tokenizer (as the plan already notes). ✅

3. **Do NOT import the `caveman` or `ponytail` skills. The *principle* is right; the *packaged skills are net-negative for single-shot metered scoring.*** `caveman` compresses **output only** but **adds ~1–1.5k input tokens per turn** for its own system prompt → on a single metered call it measured **~10% *more* expensive**, and one independent benchmark found it **+7% tokens**. `ponytail` isn't output compression at all (it's a code-minimization discipline) and multiple benchmarks found it **increases total tokens/cost by 6–55%**. Re-implement the *good idea* (terse-output instructions) as your **own 1–2 line prompt**, not by pulling in a heavyweight skill. ✅

---

## 1. Model identity — the rename strings map to real Fireworks models

The `ALLOWED_MODELS` strings are **not hackathon-only aliases**; they resolve to real Fireworks-hosted models (confirmed by a Fireworks blog + an independent cline PR that added them to its model registry). ✅

| `ALLOWED_MODELS` string | Real model | Arch | Context | Reasoning? |
|---|---|---|---|---|
| `minimax-m3` | **MiniMax M3** | Sparse MoE, ~230B total / **~10B active** | 512K (adv. 1M) | **Reasoning — togglable** ✅ |
| `kimi-k2p7-code` | **Moonshot Kimi K2.7 Code** | MoE, 1T total / **32B active** | 256K | **Reasoning (has `Preserve_thinking`)** ✅ |
| `gemma-4-31b-it` | **Gemma 4 31B** (dense) | Dense ~30.7B | 256K | **Reasoning *engine* exists (`<\|think\|>`) but off/terse by default** ⚠️ |
| `gemma-4-26b-a4b-it` | **Gemma 4 26B A4B** | MoE, 25.2B total / **~3.8B active** | 256K | Same — optional reasoning, terse default ⚠️ |
| `gemma-4-31b-it-nvfp4` | 4-bit **NVFP4 quant** of 31B | Dense, quantized | 256K | Same as 31B; faster, ~equal quality |

> ⚠️ **Nuance flagged by adversarial verification:** the blanket claim "Gemma 4 31B is a non-reasoning model" was *contested* — Gemma 4 ships a `<|think|>` reasoning mode. The reconciled truth: **Gemma 4 *can* reason but does not by default in the `-it` path**, so it behaves as the terse option *unless you trigger thinking*. **Verify at kickoff that the served `-it` endpoint does not silently emit a reasoning trace** — if it does, that changes the routing math. ❓

> 🚩 **Gemma is on-demand — deploy it before Gemma-first means anything (official clarification).** The Gemma Fireworks models are **not** auto-deployed: **deploy them at <https://app.fireworks.ai/models> first.** A **404 is "not deployed," not "banned."** Our entire escalated-path doctrine is Gemma-first, so **deploy all three Gemma endpoints at kickoff** — otherwise every Gemma route 404s and we're forced onto the pricier reasoning models (`minimax-m3` / `kimi-k2p7-code`) by default. Add "deploy Gemma" to the Day-0 checklist.

---

## 2. Per-model profile (capability + token behavior)

### `gemma-4-26b-a4b-it` — the cheap/fast default workhorse
- **~3.8B active params** ⇒ cheapest & fastest to serve. Terse, non-reasoning by default. ✅
- Surprisingly capable *with thinking on*: AIME 2026 **88.3%**, LiveCodeBench v6 **77.1%**, MMLU-Pro **82.6%**, GPQA **82.3%**. ⚠️ (vendor/DeepInfra numbers)
- **Best home for high-volume terse categories:** sentiment, NER, short factual, summarization.
- **Risk:** small models can be *too* terse and trip the judge on "explain" tasks, and the Gemma line historically has **weak factual recall / hallucination** (Gemma-3-class SimpleQA ~10.0). Watch factual-QA and NER accuracy. ⚠️

### `gemma-4-31b-it` — the strong terse generalist
- Dense 31B, non-reasoning default, **LiveCodeBench v6 80.0%**, AIME 2026 **89.2%**. ⚠️
- **Sleeper pick for code:** it's a *non-reasoning* coder, so it may clear code-debug/gen at **far fewer tokens than the reasoning `kimi-k2p7-code`.** Worth a direct A/B (see §6). ✅ (reasoning-token logic) / ⚠️ (which wins on accuracy)

### `gemma-4-31b-it-nvfp4` — same text, faster
- 4-bit NVFP4 quant → faster/cheaper latency, **near-same quality, identical tokenizer** ⇒ **identical token count for identical output.** Use it wherever 31B is chosen and latency matters; token score is unchanged. ✅

### `minimax-m3` — the strongest general reasoner (spends tokens)
- **~10B active** (best activation ratio of the candidates) ⇒ compute-cheap, but it's a **reasoning model → counted CoT tokens.** ✅
- **Toggle:** MiniMax M3's **thinking mode can be turned on/off at request time, same price** — so you can run it *non-thinking* for a terse answer, or *thinking* only for the hardest math/logic. This is the key knob. ✅
- Strengths: long-horizon agentic, hard math/logic, long-doc, multimodal (irrelevant here). Reserve for hard reasoning where Gemma fails the gate. ✅

### `kimi-k2p7-code` — the code specialist (also a reasoning model)
- MoE 1T/32B active, code-tuned; **K2.7 Code emits ~30% fewer reasoning tokens than K2.6** *but still a reasoning model* with a persistent-thinking mode. ✅
- Roughly matches MiniMax M3 on agentic coding (SWE-Bench-Pro ~58.6% vs 59.0%). ⚠️
- **Token caution:** its output is the expensive side ($4.00/1M out vs $0.95/1M in on Fireworks), it's **verbose/over-engineers by default** (needs "simplify" nudges), and it's **~3–3.3× the per-token price of `minimax-m3`.** Price ≠ your score (you're on raw count), but the *verbosity* and *reasoning trace* directly cost you tokens. Constrain hard if used. ✅

---

## 3. The reasoning-token lever (biggest single win)

- Reasoning/CoT tokens are **billed as output tokens** and count against a total-token budget. ✅
- Kimi-Thinking-class models generate **up to 2.5–5× more tokens** than a non-reasoning peer for the same task. ✅
- **Therefore:** every task you can route to a **non-reasoning Gemma** (or to `minimax-m3` **with thinking off**) avoids the entire CoT tax.
- **Concise-CoT is the fallback when you *must* reason:** forcing concise reasoning cuts response length **~49%** with **negligible** accuracy loss *in general* — **BUT on math the weaker model lost 27.69% accuracy** from concise prompting (GPT-4-class was resilient). ✅
  → **Rule:** suppress/condense reasoning freely on easy categories; on **math & logic**, condense *carefully* and let the accuracy gate + escalation decide. Don't strangle the reasoning on the exact tasks that need it.

---

## 4. Tokenizer arbitrage (the 2nd-order edge, quantified)

Larger vocabulary ⇒ fewer tokens for identical text (lower "fertility"). ✅

| Family | Vocab | Implication for a *raw-token* race |
|---|---|---|
| **Gemma 4** | **262k** (SentencePiece, shared w/ Gemini) | **Fewest tokens per identical answer** — structural advantage |
| MiniMax M3 | ~200k | Middle |
| Kimi K2.7 | ~160k (Tiktoken/BPE) | **Most tokens per identical answer** |

- So even setting verbosity aside, **the same sentence is cheapest on Gemma, priciest on Kimi.** Reinforces "Gemma-first."
- The three Gemmas are **token-identical** for identical text → choose among them purely on **accuracy + latency**, never tokens (matches the plan).
- **Measure at kickoff:** run one fixed answer string through each of the 5 models via the proxy and record the counted tokens — confirms the vocab math on the *actual* judging proxy. ❓

---

## 5. `caveman` skill — verdict: **borrow the idea, not the skill**

**What it does:** a Claude-Code "skill" that instructs the model to answer in terse telegraphic style — drop articles (a/an/the), filler (just/really/basically), pleasantries (sure/certainly), and hedging; keep code/commands/errors **byte-for-byte exact.** Graded levels (lite/full/ultra/wenyan). ✅

**The numbers, honestly:**
- Headline: **~65% output-token reduction** (range 22–87%). ⚠️ (self-reported)
- Independent benchmark (BetterStack): **45%** output reduction, **39%** vs merely asking "be terse." ✅
- **Critical for us:** it reduces **output only** — **input & reasoning tokens untouched**, and the **skill itself adds ~1–1.5k input tokens/turn.** On a **single one-off metered call it came out ~10% *more* expensive.** ✅
- Another benchmark (ponytail's suite) found caveman **+7% tokens ("terse but expensive").** ✅
- Its savings **depend on prompt-caching amortizing the big system prompt across a multi-turn session** — a condition **Track 1 does not have** (single-shot, per-task metering). ✅

**Accuracy under a gate:** self-reports **100% technical accuracy** (code preserved exactly), and cites a Mar-2026 paper that brevity constraints *improved* accuracy ~26 pts. ⚠️ **But** caveman deletes **hedging/qualifiers** — which can change *judged meaning* on sentiment ("slightly negative" → "negative"), summarization, and nuanced factual answers. Risky exactly where your gate is subjective.

> **Verdict:** ❌ Don't install the skill. ✅ Do steal its *content rule* as a **~1-line instruction** you author yourself (no 1.5k-token overhead): e.g. *"Answer in minimal words. No articles/filler/preamble. Keep any code, numbers, and names exact."* Test it per-category on your proxy judge; keep it only where the gate still passes.

---

## 6. `ponytail` skill — verdict: **wrong tool, skip entirely**

- **It is NOT output/answer compression.** It's a **code-minimization discipline** for agentic coding — a "laziest senior dev" decision-ladder that makes the agent *write less code* (reuse, stdlib, one-liners). Its own README: *"The rule was never fewest tokens."* ✅
- Token reality: **~54% less code but only ~22% fewer tokens / ~20% cost** on agentic runs — and that's **only** code-generation workloads, not QA/sentiment/NER/summarization. ✅
- **It frequently *increases* total tokens:** by driving more skill-reading/exploration **before** answering, benchmarks measured **+6% to +55% cost** with it ON (one case 187k→308k tokens). A GitHub issue documents the regression directly. ✅
- Safety: it *does* preserve correctness on code (100% on adversarial security tests), so it's a fine coding-quality tool — **just not a token-minimizer, and irrelevant to 6 of your 8 categories.**

> **Verdict:** ❌ Not applicable to Track 1. Ignore for scoring. (If anything, it's a cautionary tale: "fewer output lines" ≠ "fewer total tokens.")

---

## 7. The safe, evidence-backed token toolbox (use these instead)

All of these are single-call, input+output-honest tactics — no skill overhead, no multi-turn caching assumption.

| Tactic | Measured effect | Notes / safety |
|---|---|---|
| **Terse/near-absent system prompt** | 847→156 tok rewrite = **~81% system cost cut** | Fold instructions into the per-category user template. ✅ |
| **Zero-shot over few-shot** | 5-shot sentiment **400+ → 32 tokens** | Add one tiny example back only where the gate demands it. ✅ |
| **Explicit length cap** ("under N words", "one sentence") | **75–85%** output cut; specific > "be concise" | Vague "be concise" underperforms hard limits. ✅ |
| **"Answer only. Don't repeat/paraphrase the question. No preamble."** | **~34%** | Kills "Great question!…" and question echoing. ✅ |
| **`max_tokens` caps per category** | 20–40% (with format ctrl) | Suggested: yes/no ≈5, classification ≈10, extraction ≈100, summary ≈200. ✅ |
| **`stop` sequences** (`\n\n`, closing fence, first period) | ends gen at the answer | Combine with max_tokens as a hard backstop. ✅ |
| **Compact structure when prose would ramble** | sentiment prose **89 → 34 tok as short-key JSON**; bullets **62%** < prose | ✅ **Nuance:** structure is cheaper *only if the structure is short*. Verbose JSON keys are worse; use `s`/`l` not `sentiment_classification`. Short keys cut 50–55%. |
| **Concise-CoT (not full CoT)** on reasoning tasks | **~49%** length, ~23% cost, negligible acc loss *generally* | ⚠️ **Except math on weak models (−27.69%)** — don't over-compress math/logic. ✅ |
| **Suppress reasoning mode** where the gate allows | avoids the entire 3–5× CoT tax | `minimax-m3` thinking is togglable same-price; keep Gemma non-thinking. ✅ |

---

## 8. Revised per-category routing (research-informed deltas vs the plan)

Your plan's table is sound. The research suggests these **specific changes to A/B at kickoff**:

| Category | Plan's pick | Research-informed recommendation |
|---|---|---|
| Factual | `gemma-4-26b-a4b-it` | **Keep — but watch hallucination** (Gemma factual recall is the family's weak spot). If gate-fail rate is high, escalate to `gemma-4-31b-it` (still terse) *before* reaching for a reasoning model. |
| Math | `minimax-m3` | **Test `gemma-4-31b-it` first (thinking-off, terse).** It scores AIME ~89% and is non-reasoning ⇒ *far* cheaper tokens. Reserve `minimax-m3` **with thinking on** only for the hardest items that Gemma fails. Don't over-condense CoT here. |
| Sentiment | `gemma-4-26b-a4b-it` | **Keep.** `label — <one clause>`, `max_tokens`≈10–15, `stop="\n"`. Avoid caveman-style hedging removal (can flip nuance). |
| Summarisation | `gemma-4-31b-it-nvfp4` | **Keep.** Enforce the length constraint *locally* (free) as the plan says; cap `max_tokens` to the format. |
| NER | `gemma-4-26b-a4b-it` | **Keep.** Compact `type:value` list beats JSON *if keys are long*; measure both — short structure can win. |
| Code debugging | `kimi-k2p7-code` | **⚠️ Re-evaluate.** `kimi` is a *reasoning* model (CoT tax) **and** its 160k tokenizer is the priciest per-token **and** it over-engineers. **A/B against `gemma-4-31b-it` (non-reasoning, 262k vocab, LiveCodeBench 80%).** If Gemma clears the gate it likely wins on tokens by a wide margin. |
| Logic | `minimax-m3` | **Keep as escalation target**, but try `gemma-4-31b-it` first. Condense reasoning *carefully*. |
| Code generation | `kimi-k2p7-code` | **Same as debugging — test `gemma-4-31b-it` first.** Constrain to "function only, no prose"; validate it `ast.parse`s locally (free). |

**Cross-cutting rule:** *Gemma-first for everything; escalate to `minimax-m3` (thinking-on) or `kimi-k2p7-code` only where your proxy judge shows the Gemma route missing the gate.* Every reasoning-model call is a token tax you take on purpose.

---

## 9. Concrete prompt scaffolds (starting points to sweep)

Author these yourself — no external skill. Keep the system prompt near-empty; put the terse rule inline.

```
# Sentiment  (gemma-4-26b-a4b-it, max_tokens≈12, stop="\n")
<passage>
Sentiment (positive/negative/neutral) + 3-word reason. Format: label — reason.

# Factual  (gemma-4-26b-a4b-it → escalate 31b, max_tokens≈40)
<question>
Answer in one sentence. No preamble.

# Math  (gemma-4-31b-it first; escalate minimax-m3 thinking-on)
<problem>
Solve. Show minimal steps. Final answer on the last line as: ANSWER: <value>
# stop="\n\n" ; parse the ANSWER: line locally

# Code gen  (gemma-4-31b-it vs kimi-k2p7-code — A/B)
<spec>
Output only the function. No prose, no examples, no markdown fences.
# stop="```" as backstop ; validate ast.parse locally
```

- `temperature=0` everywhere (plan §2.5) — deterministic, no multi-sample.
- Never echo the task; the passage/code is the only unavoidable input — everything else is trimmable overhead.

---

## 10. Confirm at kickoff (❓ — these change the math)

1. **Does the judging proxy count reasoning/CoT tokens?** The whole "Gemma-first" thesis assumes yes (industry default: CoT bills as output). If the proxy somehow *excludes* thinking, reasoning models get much more attractive. **Highest-priority confirm.**
2. **Do the served `gemma-4-*-it` endpoints emit a `<think>` trace by default?** If yes, they lose their terse advantage — force non-thinking.
3. **Can `minimax-m3` thinking actually be toggled off via the Fireworks/OpenAI-compatible API** (param name)? Confirm and default it off.
4. **Tokenizer sanity check on the real proxy:** send one fixed answer string through all 5 models; record counted tokens → validates the 262k/200k/160k vocab arbitrage on the actual scorer.
5. Prompt-prefix caching discount on the proxy? (If none, short prompts still win — robust either way. And it kills any lingering case for caveman.)
6. **Deploy the Gemma endpoints** (app.fireworks.ai/models) and confirm all three respond (a 404 = not deployed). Gemma-first collapses to the pricier reasoning models without this. **Highest-priority ops step.**
7. **Local model reality-check on a 4 GB / 2 vCPU box:** which 2–3B Q4 candidate (§0.5) gives the best per-category pass-rate, and does it finish 19 tasks within 10 min on CPU? This decides how many categories we answer for free vs. escalate.

---

## Sources

**Model identity & specs**
- Fireworks — Kimi K2.7 Code launch · https://fireworks.ai/blog/kimi-k2p7-code
- Fireworks — MiniMax M3 launch · https://fireworks.ai/blog/minimax-m3-launch
- cline PR #11554 (registry confirmation of rename strings) · https://github.com/cline/cline/pull/11554
- Fireworks — Best LLMs for coding 2026 · https://fireworks.ai/blog/best-llms-for-coding
- Fireworks — Best open-source LLMs 2026 · https://fireworks.ai/blog/best-open-source-llms
- Flowtivity — Kimi K2.7 Code vs MiniMax M3 · https://flowtivity.ai/blog/kimi-k2-7-code-vs-minimax-m3/
- DeepInfra — Gemma 4 overview · https://deepinfra.com/blog/gemma-4-model-overview

**Reasoning-token behavior & verbosity**
- Artificial Analysis — Kimi K2 Thinking vs MiniMax-M2 · https://artificialanalysis.ai/models/comparisons/kimi-k2-thinking-vs-minimax-m2
- Artificial Analysis — Gemma 4 31B (Non-reasoning) vs Kimi K2 Thinking · https://artificialanalysis.ai/models/comparisons/gemma-4-31b-non-reasoning-vs-kimi-k2-thinking
- Artificial Analysis — Gemma 4 26B A4B vs Kimi K2 Thinking · https://artificialanalysis.ai/models/comparisons/gemma-4-26b-a4b-vs-kimi-k2-thinking
- kimi-k2.org — Kimi K2 Thinking vs MiniMax M2 · https://kimi-k2.org/blog/17-kimi-k2-thinking-vs-minimax-m2
- Interconnects (N. Lambert) — Kimi K2 Thinking · https://www.interconnects.ai/p/kimi-k2-thinking-what-it-means
- Artificial Analysis — Fireworks provider page · https://artificialanalysis.ai/providers/fireworks

**Tokenizers**
- Google Developers — Gemma 3 (262k SentencePiece) · https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/
- IntuitionLabs — Kimi K2 analysis (160k Tiktoken/BPE) · https://intuitionlabs.ai/articles/kimi-k2-open-weight-llm-analysis
- MiniMax-M2 technical report (200k vocab) · https://arxiv.org/html/2605.26494v1
- Rohan Paul — Balancing vocabulary size · https://www.rohan-paul.com/p/tutorial-balancing-vocabulary-size

**caveman / ponytail**
- caveman README · https://github.com/JuliusBrussee/caveman/blob/main/README.md
- caveman SKILL.md (literal prompt) · https://github.com/JuliusBrussee/caveman/blob/main/skills/caveman/SKILL.md
- BetterStack — Caveman independent analysis (45% / net-negative single-call) · https://betterstack.com/community/guides/ai/caveman-llm/
- ponytail README · https://github.com/DietrichGebert/ponytail/blob/main/README.md
- ponytail agentic benchmark (−22% tokens, caveman +7%) · https://github.com/DietrichGebert/ponytail/blob/main/benchmarks/results/2026-06-18-agentic.md
- ponytail Issue #121 (ON → more tokens/cost) · https://github.com/DietrichGebert/ponytail/issues/121

**General token-minimization tactics**
- Renze & Guven 2024, "Benefits of Concise CoT" (FLLM 2024) · https://arxiv.org/pdf/2401.05618
- Pristren — Reduce output tokens 40–60% · https://pristren.com/blog/reduce-output-tokens-guide/
- Burnwise — Token optimization guide · https://www.burnwise.io/blog/token-optimization-guide
- LLM Reference — 2026 benchmarks by task · https://www.llmreference.com/benchmarks

**Killed by adversarial verification (do not over-rely):**
- The blanket "AA classifies MiniMax + Kimi on Fireworks as reasoning models" and "Gemma 4 31B is (categorically) non-reasoning" claims were each out-voted 2–1 — the *nuance* (which specific variant; Gemma's optional `<|think|>`) is why. Reflected in §1's ⚠️ note; confirm the served endpoints' actual behavior at kickoff.


Done. The deep-research pass completed (26 sources, 212 claims, adversarial verification), and I've synthesized it into TRACK-1-MODEL-RESEARCH.md in your repo. The workflow's own synthesis step returned placeholder junk ("test", c1/e1/s1) â a bug in that stage â so I recovered the real findings from the run's journal and wrote the report myself.

The three findings that change your plan

1. The reasoning-token trap is the whole game. minimax-m3 and kimi-k2p7-code are reasoning models that emit counted chain-of-thought tokens by default â a Kimi-Thinking-class trace can be 3â5Ã the answer's tokens. The Gemma variants are terse / non-reasoning by default. In a raw-token race, every reasoning-model call is a tax you take on purpose. minimax-m3 lets you toggle thinking off at request time (same price) â that's the key knob when you must use it.

2. Gemma also wins the tokenizer arbitrage. Gemma vocab 262k vs MiniMax ~200k vs Kimi ~160k â the same answer text costs fewest counted tokens on Gemma. So Gemma is cheaper on three independent axes: non-reasoning, terse, and densest encoding. â Gemma-first for everything; escalate to a reasoning model only where your proxy judge shows Gemma missing the gate. Concretely: test gemma-4-31b-it for the code categories before kimi â kimi is a reasoning model and has the priciest tokenizer and over-engineers by default.

3. Your instinct on caveman/ponytail was right â don't import either skill.
- caveman compresses output only but adds ~1â1.5k input tokens/turn for its own system prompt. On a single metered call it measured ~10% more expensive; one independent benchmark found +7% tokens. Its savings depend on multi-turn prompt caching â which Track 1's single-shot scoring doesn't have.
- ponytail isn't output compression at all (it's a code-minimization discipline) and benchmarks found it increases total tokens/cost by 6â55%.
- The idea is sound though â re-implement it as your own 1-line instruction ("Answer in minimal words. No articles/filler/preamble. Keep code, numbers, names exact."), not a heavyweight skill. Caution: caveman-style hedging removal can flip sentiment nuance ("slightly negative" â "negative")inst your proxy judge.

The report also has: the confirmed rename-string â real-model mapping, per-model profiles, a revised per-category routing table with what to A/B, ready-to-sweep prompt scaffolds, and a Â§10 "confirm at kilist (the top one: does the judging proxy actually count reasoning tokens? â the whole Gem).

Two caveats on trust: these models are 2026-dated and many numbers come from vendor blogs / a benchmark aggregator with "estimate forthcoming" gaps â treat specific benchmark figures as directional. And adversarial verification killed the blanket "Gemma 4 31B is categorically non-reasoning" cip an optional <|think|> mode â so verify the served -it endpoint doesn't silently emit areasoning trace before you bank on it being terse.

I've also updated your track-1-decision memory with the Gemma-first doctrine and the cavemries across sessions. Want me to fold the routing-table deltas directly intoTRACK-1-IMPLEMENTATION-PLAN.md Â§5, or keep the research as a standalone companion?