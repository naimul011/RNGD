# src/rngd/llm.py
def assert_allowed(model: str) -> str:
    low = model.lower()
    for marker in BLOCKED_MARKERS:
        if marker in low:
            raise BlockedModelError(f"Model '{model}' is from a blocked provider ({marker}).")
    return model

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
