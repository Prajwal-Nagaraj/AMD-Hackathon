# Track 1 Agent

Token-efficient general-purpose agent for AMD Developer Hackathon ACT II,
Track 1. Routes each task locally (free) into one of 8 category strategies,
answers with a single tightly-constrained Fireworks call to the cheapest
model that clears the accuracy gate, validates/coerces the output locally,
and escalates to a stronger model only when local validation predicts a
gate failure. See `../TRACK-1-IMPLEMENTATION-PLAN.md` and
`../TRACK-1-MODEL-RESEARCH.md` for the design rationale.

## Layout

```
src/agent/
  main.py        orchestrator: deadline guard, concurrency, JSON I/O
  router.py      local heuristic classification into 8 categories
  strategies.py  per-category prompt/model/max_tokens/stop/escalation
  fireworks.py   OpenAI-compatible client -> FIREWORKS_BASE_URL
  validate.py    local validation + coercion (free format fixes)
  telemetry.py   in-process token accounting (stderr only, not scored)
eval/
  devset/        starter synthetic tasks per category (expand before relying on it)
  judge.py       proxy LLM-judge (scores an answer against a gold intent)
  sweep.py       category x model sweep -> (avg judge score, avg tokens)
  report.py      turns sweep_results.json into a per-category recommendation
tests/           router/validate unit tests + a fake-client I/O contract test
```

## Environment variables

Read only from the environment, never hardcoded or bundled:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS` (comma-separated)

Optional tuning knobs (defaults shown):

- `TASKS_INPUT_PATH` (`/input/tasks.json`)
- `RESULTS_OUTPUT_PATH` (`/output/results.json`)
- `DEADLINE_SECONDS` (`570`, i.e. 9m30s of the 10-minute budget)
- `MAX_CONCURRENCY` (`8`)
- `PER_TASK_TIMEOUT` (`28`, under the 30s per-request rule)

## Local development

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests are network-free (router/validate are pure functions; the
I/O contract test injects a fake Fireworks client). No API key is required
to run `pytest`.

To run the agent against a local `tasks.json` with real Fireworks calls:

```bash
export FIREWORKS_API_KEY=...
export FIREWORKS_BASE_URL=...
export ALLOWED_MODELS=minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4
export TASKS_INPUT_PATH=./sample_tasks.json
export RESULTS_OUTPUT_PATH=./sample_results.json
python -m agent.main
```

## Docker

```bash
docker buildx build --platform linux/amd64 --tag track1-agent:latest .
docker run --rm \
  -e FIREWORKS_API_KEY=... -e FIREWORKS_BASE_URL=... -e ALLOWED_MODELS=... \
  -v "$(pwd)/sample_input:/input:ro" -v "$(pwd)/sample_output:/output" \
  track1-agent:latest
```

## Eval harness

Requires real Fireworks credentials (it spends tokens -- this is dev-time
cost, not part of a submission). Expand `eval/devset/*.json` before trusting
its recommendations; the starter set is a handful of examples per category,
not the 30-50 the plan calls for.

```bash
export FIREWORKS_API_KEY=... FIREWORKS_BASE_URL=... ALLOWED_MODELS=...
python eval/sweep.py       # writes eval/sweep_results.json
python eval/report.py 0.7  # 0.7 = accuracy-gate proxy threshold
```
