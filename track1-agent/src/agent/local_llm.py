"""Local CPU inference via llama.cpp -- the primary, zero-token answering engine.

Loads a small quantized GGUF (default: Gemma-3-4b-it Q4_K_M, ~2.5 GB) once at
startup and answers tasks on CPU. Every task this resolves costs zero Fireworks
tokens, which is the best possible ranking outcome (see
TRACK-1-IMPLEMENTATION-PLAN.md: rank by Fireworks tokens ascending, local = 0).

Mirrors fireworks.CallResult so the orchestrator can treat a local answer and a
Fireworks answer uniformly. Local tokens are NOT scored, but local inference is
serial and slow on 2 vCPU -- so calls are serialized behind a lock and every
call is timed. On this path wall-clock, not tokens, is the binding constraint.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL_PATH = "models/gemma-3-4b-it-Q4_K_M.gguf"


@dataclass
class LocalResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    latency_s: float


class LocalLLM:
    """One loaded GGUF, reused for every task. Not concurrency-safe -- the async
    wrapper serializes calls (CPU inference can't truly parallelize on 2 vCPU
    anyway; concurrent generations would only thrash the cache)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_batch: Optional[int] = None,
        chat_format: Optional[str] = None,
        verbose: bool = False,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover - dep-presence guard
            raise ImportError(
                "llama-cpp-python is not installed. Run: "
                "pip install -r requirements-local.txt"
            ) from e

        self.model_path = model_path or os.environ.get("LOCAL_MODEL_PATH", DEFAULT_MODEL_PATH)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"local model not found at {self.model_path!r}. "
                "Download it first: python scripts/download_model.py"
            )
        self.model_name = os.path.basename(self.model_path)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx or int(os.environ.get("LOCAL_CTX", "4096")),
            n_threads=n_threads or int(os.environ.get("LOCAL_THREADS", str(os.cpu_count() or 2))),
            n_batch=n_batch or int(os.environ.get("LOCAL_BATCH", "256")),
            n_gpu_layers=0,  # force CPU; the grading box has no GPU
            chat_format=chat_format or os.environ.get("LOCAL_CHAT_FORMAT") or None,
            logits_all=False,
            verbose=verbose,
        )
        self._lock = asyncio.Lock()

    def complete_sync(
        self,
        *,
        user: str,
        system: Optional[str] = None,
        max_tokens: int = 256,
        stop: Optional[list] = None,
        temperature: float = 0.0,
    ) -> LocalResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        t0 = time.monotonic()
        resp = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        latency = time.monotonic() - t0

        choice = resp["choices"][0]
        text = (choice.get("message", {}).get("content") or "").strip()
        usage = resp.get("usage", {}) or {}
        return LocalResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            model=self.model_name,
            latency_s=latency,
        )

    async def complete(self, **kwargs) -> LocalResult:
        # llama.cpp is blocking and not safe under concurrent calls: serialize
        # and run off the event loop so the orchestrator stays responsive.
        async with self._lock:
            return await asyncio.to_thread(self.complete_sync, **kwargs)
