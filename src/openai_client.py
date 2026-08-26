"""Small standard-library OpenAI client used by the lab scripts."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import JUDGE_MODEL, OPENAI_API_KEY


def chat_json(system: str, user: str, model: str = JUDGE_MODEL, timeout: int = 90) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenAI API returned HTTP {error.code}: {detail}") from error
    content = payload["choices"][0]["message"]["content"]
    return json.loads(content)

