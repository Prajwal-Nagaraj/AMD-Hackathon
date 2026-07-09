"""Zero-dependency OpenAI-compatible mock server for FREE, offline pipeline testing.

You cannot deploy Gemma without billing, but the grading harness supplies the
models in ITS environment -- so locally you only need to prove the container
mechanics (routing, validation, escalation, token accounting, output schema,
deadline). Point the agent at this mock instead of Fireworks:

    # terminal 1
    .venv\\Scripts\\python.exe scripts\\mock_fireworks.py 8000

    # terminal 2 (from track1-agent\\)
    $env:FIREWORKS_API_KEY="noop"
    $env:FIREWORKS_BASE_URL="http://localhost:8000/v1"
    $env:ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"
    $env:TASKS_INPUT_PATH="sample_input\\tasks.json"; $env:RESULTS_OUTPUT_PATH="sample_output\\results.json"
    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m agent.main

The answers are canned placeholders shaped to each category's requested format,
so they exercise the real validators/escalation -- NOT a measure of answer
quality (only the grading env's real Gemma tells you that). It proves the
pipeline never crashes and always emits valid, non-empty, on-format output.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def canned_answer(user: str) -> str:
    u = user.lower()
    if "answer:" in u:  # math / logic
        return "Working through it briefly.\nANSWER: 42"
    if "positive, negative, neutral" in u or "label must be one of" in u:  # sentiment
        return "positive — clearly upbeat tone"
    if "fenced code block" in u or "corrected code" in u:  # code debug / gen
        return "```python\ndef solution(n):\n    return n * 2\n```\nCause: off-by-one in the loop bound"
    if "type: value" in u or "person, org, location, date" in u:  # ner
        return "person: Ada Lovelace\norg: Analytical Engine Co\nlocation: London\ndate: 1843"
    # factual / summarisation / fallback
    return ("This is a concise placeholder answer from the offline mock server, "
            "used only to validate the agent pipeline end to end.")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console quiet
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        model = body.get("model", "mock")
        messages = body.get("messages", [])
        user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        content = canned_answer(user)

        prompt_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        completion_tokens = max(1, len(content) // 4)
        resp = {
            "id": "mock-cmpl",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"mock Fireworks (OpenAI-compatible) on http://localhost:{port}/v1  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
