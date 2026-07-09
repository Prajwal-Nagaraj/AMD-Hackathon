**AMD Developer Hackathon: Participant Submission Guide** 

This document covers everything you need to build and submit a competitive entry. Exact evaluation inputs are intentionally omitted: your agent must be genuinely capable, not hardcoded to specific answers. 

**Track 1: General-Purpose AI Agent** 

What you are building 

An AI agent that handles a wide variety of natural language tasks across multiple capability domains, using Fireworks AI models as efficiently as possible. 

Capability categories 

Your agent will be evaluated across all eight categories. Build for all of them:

| \#  | Category  | What it covers |
| :---- | :---- | :---- |
| 1  | **Factual knowledge**  | Explaining concepts,  definitions, and how things work |
| 2  | **Mathematical reasoning**  | Multi-step arithmetic,  percentages, word problems, projections |
| 3  | **Sentiment classification**  | Labelling sentiment and justifying the classification |
| 4  5  | **Text summarisation  Named entity recognition**  | Condensing passages to a specific format or length constraint  Extracting and labelling  entities (person, org, location, date) |

| \#  | Category  | What it covers |
| :---- | :---- | :---- |
| 6  | **Code debugging**  | Identifying bugs in code  snippets and providing  corrected implementations |
| 7  | **Logical / deductive  reasoning** | Constraint-based puzzles where all conditions must be satisfied |
| 8  | **Code generation**  | Writing correct,  well-structured functions from a spec |

What to submit 

A **Docker image** pushed to a public registry (e.g. GitHub Container Registry, Docker Hub). Check out the Image Architecture requirement at the bottom of this document 

Your container must: 

1\. Read tasks from /input/tasks.json on startup 

JSON 

\[ 

 { "task\_id": "t1", "prompt": "Summarise the following text in one sentence: ..." }, 

 { "task\_id": "t2", "prompt": "..." } 

\] 

2\. Write results to /output/results.json before exiting 

JSON 

\[ 

 { "task\_id": "t1", "answer": "..." }, 

 { "task\_id": "t2", "answer": "..." } 

\]  
**Environment variables:** 

The harness injects these at runtime. Read them from the environment: do not hardcode values or bundle a .env file in your image. 

Python 

import os 

api\_key \= os.environ\["FIREWORKS\_API\_KEY"\] \# provided by harness — do not use your own 

base\_url \= os.environ\["FIREWORKS\_BASE\_URL"\] \# route ALL Fireworks calls through this URL 

models \= os.environ\["ALLOWED\_MODELS"\].split(",") \# exact model IDs published on launch day 

For local development you can use a .env file, but your submitted container must read these purely from the environment: the harness will inject the real values at evaluation time. 

| Variable  | Description |
| :---- | :---- |
| FIREWORKS\_API\_KEY  | Provided by the harness — use this key, not your own |
| FIREWORKS\_BASE\_URL  | Base URL for all Fireworks API calls — must be used to configure your client |
| ALLOWED\_MODELS  | Comma-separated list of permitted Fireworks AI model IDs, published on launch day |

**Important:** All API calls must go through FIREWORKS\_BASE\_URL. Calls that bypass this URL will not be recorded and the submission will score zero tokens. Do not hardcode model IDs: read from ALLOWED\_MODELS at runtime.  
Rules 

\- Exit code 0 on success, non-zero on failure 

\- Maximum runtime: **10 minutes** 

\- Only models in ALLOWED\_MODELS are permitted **for Fireworks calls**; calling any other model *through Fireworks* invalidates the submission (a `MODEL_VIOLATION`). This restricts the **Fireworks** path only — **local open-weight models bundled in your image are exempt** (see "Local models are a valid scoring strategy" below). 

\- /output/results.json must be valid JSON, malformed output scores zero 

\- **Local models are a valid scoring strategy (official clarification).** Your container may answer tasks with a **bundled local model**, and those answers **count fully toward accuracy**. **Only tokens routed through `FIREWORKS_BASE_URL` count toward your token score** — a task a local model answers correctly uses **zero Fireworks tokens, the best possible outcome for ranking.** You are *not* required to route inference through Fireworks; do so only where it helps you clear the accuracy gate. Any call you *do* make to Fireworks must still use an `ALLOWED_MODELS` model via `FIREWORKS_BASE_URL` (bypassing it isn't recorded; an out-of-list model is a `MODEL_VIOLATION`). 

\- Do not hardcode or cache answers; evaluation uses unseen prompt variants \- Image compressed size must not exceed 10GB — larger images are rejected before pulling 

\- Submissions are rate-limited to 10 per hour per team 

Scoring 

1\. **Accuracy gate**: LLM-Judge evaluates each answer against the expected intent. Submissions below the accuracy threshold are excluded from the leaderboard. 2\. **Token efficiency**: submissions that pass the accuracy gate are ranked ascending by total tokens recorded by the judging proxy — **and only Fireworks-routed tokens are recorded**, so locally-answered tasks contribute zero. Fewer tokens \= higher rank. 

**Scoring — clarified (organizers):**

- **The accuracy gate is 80%.** Below that you won't appear on top of the leaderboard regardless of token count.
- There are exactly **19 fixed tasks**, so every score is **n/19**. That's why teams share identical percentages (84.2 % = 16/19, 78.9 % = 15/19, 73.7 % = 14/19 …). **16/19 = 84.2 % is the lowest passing score** — miss a 4th task and you fall to 15/19 = 78.9 %, below the gate. Practical error budget: **at most 3 wrong answers.**
- The **LLM judge is not perfectly deterministic** run-to-run: identical code can score slightly differently. Known tradeoff, not rigging. **Keep margin above 80 %; don't sit on the boundary.**
- **Ops tip:** your registry's download counter (GitHub Packages / Docker Hub) shows whether the graders have pulled your image yet.

**Using a local model (Track 1) — practical limits**

The grading environment your container runs in:

| Constraint | Value |
| :---- | :---- |
| RAM | **4 GB** |
| CPU | **2 vCPU** (no GPU) |
| Fits comfortably | **2B–3B, 4-bit quantized** model |
| 7B 4-bit | Fills the entire RAM budget — no room left for your agent code |
| Model runtime | **None pre-installed** (no Ollama). Bundle the **weights *and* a CPU runtime** (e.g. llama.cpp / llama-cpp-python) directly in your Docker image |
| Image size | **≤ 10 GB compressed** (unchanged) |

Local open-weight models are **not** restricted to `ALLOWED_MODELS` — that list only governs calls made *through Fireworks*. Locally you may bundle any open-weight model. You still must not hardcode or cache answers to specific inputs.

**Troubleshooting: why did my submission fail? (Track 1)**

The Track 1 pipeline is live and scoring normally. A failed submission returns a specific status:

| Status | Meaning | How to fix |
| :---- | :---- | :---- |
| `PULL_ERROR` | The image couldn't be pulled | Push to a **public** registry; include a **linux/amd64** manifest; verify the tag is correct and public |
| `RUNTIME_ERROR` | Container crashed / exited non-zero / threw | Exit **0** on success; wrap each task in try/except; always write a valid results file, even on partial failure |
| `TIMEOUT` | Exceeded the **10-minute** runtime | Deadline guard; parallelize Fireworks calls; don't let slow CPU-only local inference overrun — fall back to Fireworks under time pressure |
| `INVALID_RESULTS_SCHEMA` | `/output/results.json` malformed or wrong shape | Emit exactly `[{"task_id","answer"}]`; validate the JSON/schema before writing |
| `MODEL_VIOLATION` | Called a model **not** in `ALLOWED_MODELS`, or made a Fireworks call bypassing `FIREWORKS_BASE_URL` | Read model IDs from `ALLOWED_MODELS`; route every Fireworks call through `FIREWORKS_BASE_URL`. **Local open-weight models are allowed and are not a violation.** |
| `IMAGE_TOO_LARGE` | Compressed image > 10 GB | Slim base image; 4-bit quantized local model; prune build caches/artifacts |
| `ACCURACY_GATE_FAILED` | Below the 80 % gate | Improve answers; escalate more tasks to Fireworks / a stronger model; leave margin for judge nondeterminism |

**Practice tasks (Track 1)**

The organizers publish **illustrative example tasks** (found in the participant guide after *What to submit*) — **not** the real 19-task grading set — so you can validate your container's input/output handling locally before spending a real submission slot. Use them to confirm the I/O contract, schema, env-var reading, and Docker start — not to hardcode answers.

**Gemma is allowed, but on-demand**

Gemma is permitted, but the Gemma Fireworks models are **on-demand**: **deploy them at <https://app.fireworks.ai/models> first.** A **404 means "not deployed," not "banned."** If your routing is Gemma-first, deploy the Gemma endpoints at kickoff or those routes will 404.

**Track 2: Video Captioning Agent** 

What you are building 

An AI agent that watches a video clip and generates a caption in a requested style. Build your pipeline to handle a wide variety of video content: different scenes, settings, subjects, and visual styles. 

Styles your agent must support

| Style  | Description |
| :---- | :---- |
| formal  | Professional, objective, factual tone |
| sarcastic  | Dry, ironic, lightly mocking |
| humorous\_tech  | Funny, with technology or programming references |

| Style  | Description |
| :---- | :---- |
| humorous\_non\_tech  | Funny, everyday humour with no technical jargon |

What to submit 

A **Docker image** pushed to a public registry. 

Your container must: 

1\. Read tasks from /input/tasks.json on startup 

JSON 

\[ 

 { 

 "task\_id": "v1", 

 "video\_url": "https://storage.example.com/clips/clip1.mp4", 

 "styles": \["formal", "sarcastic", "humorous\_tech", "humorous\_non\_tech"\]  } 

\] 

2\. Write results to /output/results.json before exiting 

JSON 

\[ 

 { 

 "task\_id": "v1", 

 "captions": { 

 "formal": "...", 

 "sarcastic": "...", 

 "humorous\_tech": "...", 

 "humorous\_non\_tech": "..." 

 } 

 } 

\] 

No inference log is required for Track 2\. 

Check out the Image Architecture requirement at the bottom of this document  
Environment variables 

No API key or model restriction is injected. You may call any model, API, or framework: use your own credentials inside the container. 

Example clips 

Use these to develop and test your agent. Evaluation uses a hidden set your agent has never seen, ensure your pipeline generalises beyond these examples. 

| Clip  | URL  | Content |
| :---- | :---- | :---- |
| v1  | https://storage.google apis.com/amd-hackathon \-clips/1860079-uhd\_256 0\_1440\_25fps.mp4 | Urban autumn boulevard with golden trees and city traffic |
| v2  | https://storage.google apis.com/amd-hackathon \-clips/13825391-uhd\_38 40\_2160\_30fps.mp4 | Orange kitten among green foliage in a garden |
| v3  | https://storage.google apis.com/amd-hackathon \-clips/3044693-uhd\_384 0\_2160\_24fps.mp4 | Office worker at a desktop computer in a modern  open-plan office |

Rules 

\- Exit code 0 on success, non-zero on failure 

\- Maximum runtime: **10 minutes** 

\- No model restriction: any model, framework, or API is permitted 

\- **Video clips in the hidden evaluation set are 30 seconds to 2 minutes in length** \- Output must include a caption for every requested style for every clip, missing styles score zero for that clip 

\- /output/results.json must be valid JSON, malformed output scores zero \- Image compressed size must not exceed 10GB — larger images are rejected before pulling 

\- Submissions are rate-limited to 10 per hour per team 

Scoring 

Each caption is scored on two dimensions by an LLM-Judge:  
1\. **Caption accuracy** (0–1): how faithfully the caption reflects the video content 2\. **Style match** (0–1): how well the caption matches the requested tone 

Final score \= weighted average across all clips and all four styles. Evaluation uses a hidden set of \~12 clips spanning varied content (nature, urban, animals, people, sports, food, weather, technology). Agents that only work on the three example clips above will score poorly. 

**Track 3: Unicorn (Open Innovation)** 

What you are building 

An original AI application that uses AMD compute resources. There is no fixed task, we are looking for the most innovative, technically impressive, and practically useful projects. 

What to submit 

| Item  | Required |
| :---- | :---- |
| GitHub repository URL  | Yes |
| Demo video  | Yes |
| Slide deck  | Yes |
| Live demo / hosted URL  | Optional but recommended |

Note: automated pre-screening only inspects the GitHub repository, slide deck (PDF), and live demo/hosted URL — it does not process the demo video. 

No Docker image is required for Track 3\. 

Judging 

Submissions are pre-screened automatically for AMD resource usage and originality, then reviewed by human judges. AMD compute usage is a requirement: projects that do not demonstrate it will be disqualified. 

General rules (all tracks) 

\- Your container must start and be ready within **60 seconds**  
\- Response time per request must be under **30 seconds** 

\- All responses must be in **English** 

\- Do not hardcode or cache answers to specific inputs — evaluation uses unseen variants \- Container images must be publicly pullable at submission time 

**Image architecture requirement** 

The judging VM runs ***linux/amd64***. Your image must include a ***linux/amd64*** manifest or it will fail to pull and score zero. 

If you build on Apple Silicon (M1/M2/M3), add ***\--platform linux/amd64*** to your build command: 

Shell 

docker buildx build \--platform linux/amd64 \--tag your-image:latest \--push . 

Standard ***linux/amd64*** builds (e.g. built on Intel/AMD or GitHub Actions) are fine without any changes.


Question that I asked: Hi team! Quick clarification question regarding Track 1 (AI Agent):The stream slide mentioned that "local tokens count as zero," but the official announcement specifies that tasks must be handled "using Fireworks AI models" from the list of 5 allowed endpoints (Minimax, Kimi, Gemma).Does the "local tokens count as zero" rule mean we are allowed to use an ultra-lightweight, local open-weight model on our own hardware purely to categorize/route the incoming prompts before sending them to the allowed Fireworks endpoints? Or must all text processing happen strictly within those 5 Fireworks models? Thank you! @Steve | lablab.ai


Answer from the moderator: Yes you're allowed to use local models, and then send them to the fireworks endpoints.

Allowed models (Track 1):
minimax-m3
kimi-k2p7-code
gemma-4-31b-it
gemma-4-26b-a4b-it
gemma-4-31b-it-nvfp4