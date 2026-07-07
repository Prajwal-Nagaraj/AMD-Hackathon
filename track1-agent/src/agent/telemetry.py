"""In-process token accounting, for our own tuning -- not part of the
required output contract. Written to stderr as a single JSON line at the
end of a run so it never touches /output/results.json.
"""

import threading


class Telemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.calls = []

    def record(self, task_id, category, model, prompt_tokens, completion_tokens, escalated=False):
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.calls.append(
                {
                    "task_id": task_id,
                    "category": category,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "escalated": escalated,
                }
            )

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "num_calls": len(self.calls),
                "num_escalations": sum(1 for c in self.calls if c["escalated"]),
            }
