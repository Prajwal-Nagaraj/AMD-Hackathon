"""OpenAI-compatible client pointed at FIREWORKS_BASE_URL.

Every answer-producing call in this project must go through here: it is the
only component allowed to talk to the network, and it is the only source of
truth for token counts (the judging proxy meters exactly these calls).
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


@dataclass
class CallResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class FireworksClient:
    def __init__(self):
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        raw_models = os.environ["ALLOWED_MODELS"]
        self.allowed_models = [m.strip() for m in raw_models.split(",") if m.strip()]
        if not self.allowed_models:
            raise RuntimeError("ALLOWED_MODELS is empty")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def resolve_model(self, name: str) -> str:
        """Map a strategy's requested model name onto an exact ALLOWED_MODELS entry.

        Exact match first; falls back to substring match so strategy code
        doesn't break if the published model IDs gain/lose a suffix.
        """
        if name in self.allowed_models:
            return name
        for m in self.allowed_models:
            if name in m or m in name:
                return m
        raise ValueError(f"model {name!r} not resolvable against ALLOWED_MODELS: {self.allowed_models}")

    async def complete(
        self,
        *,
        model: str,
        user: str,
        system: Optional[str] = None,
        max_tokens: int,
        stop: Optional[list] = None,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
        timeout: float = 25.0,
        retries: int = 2,
    ) -> CallResult:
        model_id = self.resolve_model(model)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                        extra_body=extra_body or {},
                    ),
                    timeout=timeout,
                )
                choice = resp.choices[0]
                text = choice.message.content or ""
                usage = resp.usage
                return CallResult(
                    text=text,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    model=model_id,
                )
            except Exception as exc:  # noqa: BLE001 - retry on anything, surface the last
                last_exc = exc
                if attempt < retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
        raise RuntimeError(f"Fireworks call failed after {retries + 1} attempts: {last_exc}") from last_exc
