from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None


class Backend:
    """Interface mínima: recebe N conversas e devolve `n` amostras para cada uma."""

    name: str = "base"

    def generate(
        self,
        conversations: list[list[dict]],
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> list[list[Completion]]:
        raise NotImplementedError
