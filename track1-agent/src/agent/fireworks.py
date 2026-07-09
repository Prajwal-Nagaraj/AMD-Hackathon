"""OpenAI-compatible client pointed at FIREWORKS_BASE_URL.

Every answer-producing call in this project must go through here: it is the
only component allowed to talk to the network, and it is the only source of
truth for token counts (the judging proxy meters exactly these calls).

Model IDs are never hardcoded (competition rule: read them from ALLOWED_MODELS
at runtime). Strategies ask for a *tier* -- cheap / mid / strong / code -- and
`infer_tiers` derives which allowed model fills each tier by parsing the
published IDs. Reasoning-token suppression also lives here so every call gets
it uniformly.
"""

import asyncio
import os
import re
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


# --- Model tiering -----------------------------------------------------------
# Tiers are derived from the ALLOWED_MODELS ID strings, never hardcoded:
#   strong = largest general model (unsized IDs are treated as frontier-class),
#   code   = code-specialised model (else falls back to strong),
#   cheap  = fewest ACTIVE params (MoE-aware), quantized preferred on ties,
#   mid    = densest general model below `strong`, non-quantized preferred --
#            the reliable workhorse and escalation target for the cheap tier.

TIERS = ("cheap", "mid", "strong", "code")

_MOE_RE = re.compile(r"(\d+)\s*x\s*(\d+)\s*b\b")   # mixtral-8x7b -> 56
_ACTIVE_RE = re.compile(r"\ba(\d+)b\b")            # gemma-...-a4b -> 4 active
_DENSE_RE = re.compile(r"(\d+)\s*b\b")             # llama-...-8b  -> 8
_CODE_RE = re.compile(r"code|coder", re.I)
_QUANT_RE = re.compile(r"nvfp4|fp4|fp8|int8|int4|awq|gptq|gguf", re.I)


def _total_params(model_id: str) -> int:
    mid = model_id.lower()
    moe = _MOE_RE.search(mid)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))
    sizes = [int(m.group(1)) for m in _DENSE_RE.finditer(mid)]
    return max(sizes) if sizes else 100  # unsized IDs are frontier-class


def _active_params(model_id: str) -> int:
    m = _ACTIVE_RE.search(model_id.lower())
    return int(m.group(1)) if m else _total_params(model_id)


def _is_code(model_id: str) -> bool:
    return bool(_CODE_RE.search(model_id))


def _is_quant(model_id: str) -> bool:
    return bool(_QUANT_RE.search(model_id))


def infer_tiers(models: list) -> dict:
    """Map cheap/mid/strong/code onto concrete ALLOWED_MODELS entries.

    Every tier always resolves to a real model, even for short lists (tiers
    collapse onto the same model rather than raising).
    """
    general = [m for m in models if not _is_code(m)] or list(models)
    strong = max(general, key=lambda m: (_total_params(m), not _is_quant(m)))
    code_models = [m for m in models if _is_code(m)]
    code = max(code_models, key=_total_params) if code_models else strong
    cheap = min(general, key=lambda m: (_active_params(m), not _is_quant(m)))
    mid_pool = [m for m in general if m != strong] or [strong]
    mid = max(mid_pool, key=lambda m: (_total_params(m), not _is_quant(m)))
    return {"cheap": cheap, "mid": mid, "strong": strong, "code": code}


# --- Reasoning-token suppression ---------------------------------------------
# Reasoning models (e.g. minimax-m3) otherwise spend the whole max_tokens budget
# on hidden reasoning and return a BLANK answer -- zero accuracy AND worst-case
# token cost. `reasoning_effort` is sent verbatim as a top-level JSON field via
# extra_body, so it works regardless of the installed OpenAI SDK version; any
# model that rejects it is remembered and retried without it.
DEFAULT_REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")
_NO_REASONING_PARAM: set = set()


class FireworksClient:
    def __init__(self):
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        raw_models = os.environ["ALLOWED_MODELS"]
        self.allowed_models = [m.strip() for m in raw_models.split(",") if m.strip()]
        if not self.allowed_models:
            raise RuntimeError("ALLOWED_MODELS is empty")
        self.tiers = infer_tiers(self.allowed_models)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def resolve_model(self, name: str) -> str:
        """Map a requested model name onto an exact ALLOWED_MODELS entry.

        Exact match first; falls back to substring match so an override or a
        published ID that gains/loses a suffix still resolves.
        """
        if name in self.allowed_models:
            return name
        for m in self.allowed_models:
            if name in m or m in name:
                return m
        raise ValueError(f"model {name!r} not resolvable against ALLOWED_MODELS: {self.allowed_models}")

    def model_for_tier(self, tier: str) -> str:
        """Concrete model for a tier.

        Precedence: MODEL (global override) > MODEL_<TIER> > inferred from
        ALLOWED_MODELS. Overrides are resolved against the allowed list too.
        """
        override = os.environ.get("MODEL") or os.environ.get(f"MODEL_{tier.upper()}")
        if override:
            return self.resolve_model(override)
        try:
            return self.tiers[tier]
        except KeyError:
            raise ValueError(f"unknown tier {tier!r}; known tiers: {TIERS}")

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

        base_body = dict(extra_body or {})
        want_reasoning = bool(DEFAULT_REASONING_EFFORT) and "reasoning_effort" not in base_body

        last_exc: Optional[Exception] = None
        attempt = 0
        while attempt <= retries:
            body = dict(base_body)
            if want_reasoning and model_id not in _NO_REASONING_PARAM:
                body["reasoning_effort"] = DEFAULT_REASONING_EFFORT
            try:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                        extra_body=body,
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
                # Model rejects reasoning_effort: forget it for this model and
                # retry immediately without it. This doesn't consume a transient
                # retry, and can only fire once per model (the guard flips off).
                if (
                    want_reasoning
                    and model_id not in _NO_REASONING_PARAM
                    and "invalid_request_error" in str(exc)
                ):
                    _NO_REASONING_PARAM.add(model_id)
                    continue
                if attempt < retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    attempt += 1
                    continue
                break
        raise RuntimeError(
            f"Fireworks call failed after {retries + 1} attempts: {last_exc}"
        ) from last_exc
