"""OpenAI Responses API helpers for structured RoboGenesis agents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str | None
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "OpenAISettings":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        )


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def create_structured_json(
    *,
    schema_path: str | Path,
    system_prompt: str,
    user_payload: dict[str, Any],
    settings: OpenAISettings | None = None,
) -> dict[str, Any]:
    """Call OpenAI with a JSON Schema and return parsed JSON.

    The API key is read from `OPENAI_API_KEY`; the model defaults to
    `OPENAI_MODEL` or `gpt-5.4-mini`.
    """

    settings = settings or OpenAISettings.from_env()
    if not settings.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI

    schema = json.loads(Path(schema_path).read_text())
    client = OpenAI(api_key=settings.api_key)
    response = client.responses.create(
        model=settings.model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema.get("title", "RoboGenesisSchema"),
                "schema": schema,
                "strict": True,
            }
        },
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        output_text = _extract_output_text(response)
    return json.loads(output_text)


def _extract_output_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    if not chunks:
        raise RuntimeError("OpenAI response did not contain output text")
    return "\n".join(chunks)

