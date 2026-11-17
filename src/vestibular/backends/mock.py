"""Baseline aleatório ("chute"). Serve para dois propósitos:
1. linha de base honesta no leaderboard (quanto se acerta chutando?);
2. testar o pipeline inteiro no CI sem API e sem GPU.
"""

from __future__ import annotations

import random
import re

from vestibular.backends.base import Backend, Completion


class RandomBackend(Backend):
    def __init__(self, model: str = "random", seed: int = 0, **_):
        self.name = f"mock:{model}"
        self.rng = random.Random(seed)

    def generate(self, conversations, *, n=1, temperature=0.0, max_tokens=2048):
        out = []
        for conv in conversations:
            content = conv[-1]["content"]
            text = content if isinstance(content, str) else " ".join(p.get("text", "") for p in content)
            letters = re.findall(r"^([A-E])\) ", text, flags=re.M) or list("ABCDE")
            out.append(
                [Completion(text=f"Chutando.\nResposta: {self.rng.choice(letters)}") for _ in range(n)]
            )
        return out
