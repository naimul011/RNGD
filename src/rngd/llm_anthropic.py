"""Anthropic wrapper — reserved for the single most complex synthesis step
(the final Arbiter decision) per project directive: Claude is the expensive,
highest-quality reasoning step, called once per pipeline run, not per agent.
"""
from __future__ import annotations

import json
import time
from typing import Any

from anthropic import Anthropic

from . import config

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def call_json(system: str, user: str, max_tokens: int = 2000) -> tuple[dict[str, Any], float]:
    client = _get_client()
    start = time.perf_counter()
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text" and b.text]
    if not text_blocks:
        raise ValueError(f"No text block in Anthropic response (stop_reason={resp.stop_reason})")
    text = text_blocks[-1]
    # Strip markdown fences if the model wraps its JSON despite instructions.
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    return parsed, round(latency_ms, 1)
