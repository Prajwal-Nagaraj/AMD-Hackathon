# AMD Developer Hackathon: ACT II — Build Guide

> Personal reference doc. Researched from the official lablab.ai page on **2026-06-22**.
> Always re-check the live page for the latest dates/rules: <https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii>

---

## TL;DR

- **What:** Online AI-agent hackathon by **AMD × lablab.ai (NativelyAI)**. Build AI agents / high-performance AI apps on **AMD GPUs in the cloud**.
- **Cost:** Free.
- **Kick-off:** **Mon, Jul 6, 2026, 9:30 PM IST**
- **Submissions close:** **Sat, Jul 11, 2026, 9:30 PM IST** → a **~5-day build sprint**.
- **Cash prizes:** **$10,000** total → 🥇 $5,000 · 🥈 $3,000 · 🥉 $2,000 (awarded to **top overall projects**, not per track).
- **3 tracks:** (1) Token-Efficient Routing Agent, (2) Video Captioning, (3) Unicorn (open product).
- **Hard rules:** Submission must be **containerized (Docker)**, repo **public + MIT-licensed**, plus **video + slides + demo URL**.
- **Action item:** **Register ASAP** — enrollment needs approval, which takes time.

---

## ⏰ Timeline & Critical Dates

| Date (IST) | Event |
|---|---|
| **Now → Jul 6** | Register, get approved, set up environment, skill up |
| **Jul 6, 9:30 PM** | Hackathon kick-off |
| Jul 6, 9:35 PM | lablab.ai opening words (Pawel Czech) |
| Jul 6, 9:40 PM | AMD opening words |
| Jul 6, 9:45 PM | **Introduction to the Challenge** (Track 1 & 2 tasks/clips revealed here) |
| Jul 6, 10:05 PM | Hackathon guide |
| Jul 6, 10:30 PM | Discord Q&A |
| **Jul 11, 9:30 PM** | **END OF SUBMISSIONS** |

> ⚠️ The actual specific tasks (Track 1) and video clips (Track 2) are **revealed only at kick-off**. You can pre-build scaffolding/infra, but not the final solution. Track 3 is fully open — you can plan/design now.

---

## 📝 Registration & What You Get

**How to register:** On the hackathon page, click **"Sign up with AMD"** → join the **AMD AI Developer Program (ADP)**. New members create an account. **Enrollment requires approval**, so do this early.

**Perks for joining ADP (new members):**
- ☁️ **$100 AMD Developer Cloud credits** + access to AMD GPUs
- 🔥 **$50 Fireworks AI API credits**
- 🎓 **AMD AI Academy** (courses, docs, tutorials)
- 📘 **1-month DeepLearning.AI Pro** (complimentary)
- 🧑‍💻 **AMD expert access** (engineers, office hours)
- 💬 Private community channels (Discord)
- 🏆 Project recognition (possible showcase on AMD channels/events)
- 🚀 **Extra compute + API credits** dropped to all participants at launch

Everything runs **fully in the cloud** — no local hardware required.

---

## 🛤️ The Three Tracks

### Track 1 — Hybrid Token-Efficient Routing Agent  ⭐ Beginner-friendly · AI Agent
Build an agent that completes a **fixed task set** (revealed at kickoff) using the **fewest tokens / lowest cost**.
- Route intelligently between a **local model** (your machine / local AMD resource) and a **remote model** (Fireworks AI), picking whichever is cheaper **without dropping below a set accuracy threshold**.
- **Scored on a standardized CPU-only environment (no GPU).** Develop on anything, but final scoring is CPU-only → **routing intelligence wins, not raw compute**.
- **Models:** restricted to the **two AMD-hardware models on Fireworks AI**.
- **Fine-tuning:** explicitly allowed (you may fine-tune a router). Prompt-based and fine-tuned approaches scored identically on **token count + output accuracy** only.
- **Build ideas:** a minimal-token task agent; a model-router/cost-optimizer that picks the cheapest suitable endpoint per query.

### Track 2 — Video Captioning via Fireworks AI API  🎬 API-only
Build a pipeline that captions a **fixed set of short clips** in **four styles**: **formal, sarcastic, humorous-tech, humorous-non-tech**.
- Runs **entirely via Fireworks AI API** — no other compute.
- **Models:** the **two AMD-hardware models on Fireworks** only.
- **Fine-tuning:** allowed; may use **open or custom datasets** to shape style/quality.
- Lowest setup overhead, no GPU needed.

### Track 3 — Unicorn Track  🦄 No fixed benchmark
Build a **product- / startup-oriented** project using **any** open-source models, frameworks, and AMD infra.
- Use **AMD Compute pods with Radeon GPUs** and/or **Fireworks AI API credits** (AMD-specified models).
- **No performance benchmark** — not scored on speed/tokens/accuracy.
- **Judged on creativity, originality, product/market potential.**
- Open to all skill levels and any stack. **Highest ceiling for the cash prizes.**

---

## 🧰 Technology & Resources Available

| Resource | What it gives you |
|---|---|
| **AMD Developer Cloud** | On-demand **AMD Instinct GPUs** (e.g., **MI300X, 192 GB VRAM** → big models / long context / multimodal) and **Radeon GPU pods** (Track 3). $100 credits. Cloud-based. |
| **ROCm** | AMD's open-source GPU compute stack. Run **PyTorch / TensorFlow** on AMD, **port CUDA→AMD (HIP)**, high-perf AI/ML. [Docs](https://rocm.docs.amd.com) · [GitHub](https://github.com/ROCm/ROCm) |
| **Fireworks AI API** | $50 credits. **Two AMD-hardware-served models** (the only models allowed for Tracks 1 & 2). |
| **AMD AI Academy** | Courses, tutorials, hands-on labs with real GPU access. |
| **DeepLearning.AI Pro** | 1-month free for advanced content. |
| **Community / Experts** | Discord (teaming + most activity), AMD engineer office hours. |

Key links:
- AMD AI Developer Program — sign-up via the hackathon page
- AMD Developer Cloud overview + Getting Started Guide (linked on hackathon page)
- ROCm Documentation / Installation Guide / GitHub
- "From Zero to AI Builder with AMD" article (on lablab.ai)

---

## 🏆 Prizes

**Cash pool: $10,000** (awarded to **top overall projects**, across all tracks):
- 🥇 1st — **$5,000**
- 🥈 2nd — **$3,000**
- 🥉 3rd — **$2,000**

**Referral pool: $1,000** (top 3 referrers get cash + Natively credits) — **requires 100+ approved referrals to qualify.** Not worth chasing for most participants.

> Some marketing lists a larger "$21k+" figure; that bundles in the per-participant credits/perks. The actual **cash** pool is **$10,000**.

---

## ⚖️ Judging Criteria

1. **Application of Technology** — how effectively the chosen model(s) are integrated.
2. **Presentation** — clarity and effectiveness of the project presentation.
3. **Business Value** — impact / practical value; fit into business areas.
4. **Originality** — uniqueness & creativity of the solution.

> Because prizes go to the **top overall project** judged on these four, **originality + business value + presentation matter a lot**. A polished product (Track 3) has the highest ceiling; benchmark tracks (1 & 2) still benefit from strong presentation.

---

## 📦 What to Submit

- **Basic info:** project title, short description, long description, technology & category tags
- **Media:** cover image, **video presentation**, **slide presentation**
- **Code & hosting:** **public GitHub repository**, demo application platform, **application URL**
- **Hard requirements:**
  - ✅ Submission must be **containerized (Docker)**
  - ✅ Code must be **original and MIT-compliant** (license your repo MIT)
- **Deadline:** Jul 11, 9:30 PM IST (always confirm on the Event Schedule tab — shown in your local timezone)
- Submission guidelines are linked on the hackathon page.

---

## 🎯 Strategy Notes

- **Prizes are overall, not per-track.** A compelling, well-presented product that *visibly* uses AMD hardware is the strongest bet for the $5k. The rubric rewards originality + business value + presentation.
- **Track choice by goal:**
  - Want the **best shot at the money / a portfolio piece** → **Track 3 (Unicorn)**.
  - **Newer**, want a **contained, objectively-scored** challenge with low presentation pressure → **Track 1**.
  - Want **minimal setup**, API-only, a little creative → **Track 2**.
- **Show off the hardware** for "Application of Technology" points: use MI300X's 192 GB VRAM for something a small GPU can't do (large model, long context, full-res multimodal, multi-agent).
- **Presentation is a scored category** — budget real time for the video + slides + a clean demo. Many good projects lose here.
- **A teammate helps** (esp. for demo/video). Find teammates on the **lablab.ai Discord**.
- **Containerize from day one** — don't leave Dockerization to the final hour; it's mandatory.

### Project idea seeds (Track 3 / Unicorn)
- Multi-agent **research → report** or **ops-automation** assistant (needs long context / big model on MI300X).
- **Domain agent** fine-tuned on AMD GPUs (e.g., clinical / legal / finance — a past winner did clinical AI).
- **Multimodal agent** (vision + text at full resolution).
- **Coding / DevOps agent** with tool use.
> Pick one with a *crisp demo* and *clear business value*.

---

## ✅ Pre-Kickoff Prep Checklist (do before Jul 6)

- [ ] **Register / "Sign up with AMD"** and get enrollment **approved**
- [ ] Join the **lablab.ai Discord**; look for teammates if going team
- [ ] Activate **AMD Developer Cloud** credits; spin up a GPU instance once to confirm access
- [ ] Set up **Fireworks AI** account + API key; confirm which **two AMD models** are available; run a test call
- [ ] Get a **Docker** workflow ready (base image, build, run locally)
- [ ] Create a **GitHub repo** with an **MIT LICENSE** and a skeleton README
- [ ] Skim **ROCm** + PyTorch-on-AMD basics; do one AMD AI Academy / DeepLearning.AI module
- [ ] Decide a **track** (or shortlist) and draft a project concept
- [ ] (Track 1/2) Pre-build a **harness/scaffold** so you can drop in the kickoff tasks/clips fast

---

## ❓ To Confirm at Kickoff / On the Page

- Exact identities of the **two AMD-hardware models on Fireworks** (which models, context limits, pricing).
- **Team size** limits (page doesn't specify — Discord/guidelines will).
- Whether grand prizes are split per-track or pooled (page says "top overall project").
- Track 1's exact **accuracy threshold** and the standardized CPU eval container spec.

---

## 🔗 Sources

- Official page: <https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii>
- AMD article — "Build Across the AI Stack": <https://www.amd.com/en/developer/resources/technical-articles/2026/build-across-the-ai-stack--join-the-amd-x-lablab-ai-hackathon-.html>
- "From Zero to AI Builder with AMD": <https://lablab.ai/ai-articles/from-zero-to-ai-builder-amd-developer-program>
- AMD Developer events: <https://developer.amd.com/events/>
- ROCm docs: <https://rocm.docs.amd.com>
