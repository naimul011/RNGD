"""Groq chat wrapper: JSON-mode calls with a self-correction retry loop.

Groq hosts fast open-weight text models (openai/gpt-oss-*, qwen*, etc). At the
time this was built the account behind GROQ_API_KEY exposed no multimodal
vision model, so all "look at the drawing" work in this project is done by
classical computer vision (src/rngd/vision/*) instead of a vision LLM; Groq is
used purely for fast text reasoning/extraction. Swapping in a vision model
later only means adding a vision_groq.py alongside this file.
"""
from __future__ import annotations

import json
import time
from typing import Any

from groq import Groq

from . import config

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def call_json(
    model: str,
    system: str,
    user: str,
    max_retries: int = 2,
    temperature: float = 0.2,
) -> tuple[dict[str, Any], float]:
    """Calls Groq in JSON mode, retrying with the parse error fed back to the
    model if it returns malformed JSON. Returns (parsed_dict, latency_ms).
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    start = time.perf_counter()
    last_error = None
    for attempt in range(max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        raw = resp.choices[0].message.content
        try:
            parsed = json.loads(raw)
            latency_ms = (time.perf_counter() - start) * 1000
            return parsed, round(latency_ms, 1)
        except json.JSONDecodeError as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That was not valid JSON ({exc}). Reply again with ONLY "
                        "a single valid JSON object, no prose, no markdown fences."
                    ),
                }
            )
    raise ValueError(f"Groq call_json failed after {max_retries + 1} attempts: {last_error}")


def call_text(model: str, system: str, user: str, temperature: float = 0.3) -> tuple[str, float]:
    client = _get_client()
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return resp.choices[0].message.content, round(latency_ms, 1)


def call_chat(model: str, messages: list[dict[str, str]], temperature: float = 0.4) -> tuple[str, float]:
    """Multi-turn variant of call_text: `messages` is a full system+history
    list, used by the chat assistant (rngd/chat.py) to keep conversational
    context across turns instead of one-shot system+user."""
    client = _get_client()
    start = time.perf_counter()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    latency_ms = (time.perf_counter() - start) * 1000
    return resp.choices[0].message.content, round(latency_ms, 1)
