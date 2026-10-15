"""All LLM calls go through OpenRouter (OPENROUTER_API_KEY), via its OpenAI-compatible API.

Model IDs are checked against a blocklist of Chinese model providers before any
request is sent, so a misconfigured model name fails loudly instead of quietly
calling a disallowed model.
"""
from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from . import config

# Providers whose models must never be used in this project.
BLOCKED_MARKERS = (
    "qwen", "deepseek", "moonshot", "kimi", "z-ai", "zhipu", "glm", "baichuan", "minimax",
    "01-ai", "yi-", "/yi", "tencent", "hunyuan", "baidu", "ernie", "bytedance", "doubao",
    "stepfun", "xiaomi", "inclusionai", "internlm", "sensetime", "thudm", "chatglm",
)

_client: OpenAI | None = None


class BlockedModelError(ValueError):
    pass


def assert_allowed(model: str) -> str:
    low = model.lower()
    for marker in BLOCKED_MARKERS:
        if marker in low:
            raise BlockedModelError(f"Model '{model}' is from a blocked provider ({marker}).")
    return model


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
        # Explicit timeout: the SDK default (600 s x 3 attempts) let one stalled OpenRouter
        # request freeze a pipeline run for up to 30 minutes.
        _client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY,
                         default_headers={"X-Title": "RNGD AI Studio"}, timeout=90.0, max_retries=2)
    return _client


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def call_json(model: str, system: str, user: str, max_retries: int = 2,
              temperature: float = 0.2) -> tuple[dict[str, Any], float]:
    """JSON-mode call; feeds malformed-JSON errors back to the model and retries."""
    assert_allowed(model)
    client = _get_client()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    start = time.perf_counter()
    last_error = None
    for _ in range(max_retries + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(_strip_fences(raw))
            return parsed, round((time.perf_counter() - start) * 1000, 1)
        except json.JSONDecodeError as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                f"That was not valid JSON ({exc}). Reply again with ONLY a single valid JSON object, "
                "no prose, no markdown fences.")})
    raise ValueError(f"JSON call failed after {max_retries + 1} attempts: {last_error}")


def call_chat(model: str, messages: list[dict[str, str]], temperature: float = 0.4) -> tuple[str, float]:
    assert_allowed(model)
    start = time.perf_counter()
    resp = _get_client().chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content or "", round((time.perf_counter() - start) * 1000, 1)


def call_text(model: str, system: str, user: str, temperature: float = 0.3) -> tuple[str, float]:
    return call_chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}], temperature)
