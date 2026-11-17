"""Formato único de questão, usado por todas as fontes (ENEM, BLUEX, OAB).

Cada dataset original tem um formato diferente. Convertê-los para um único
`Question` faz com que prompts, avaliação e treino não precisem saber de onde
a questão veio.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

LETTERS = "ABCDE"

# Marcador usado no texto para indicar onde entra a imagem i.
IMAGE_TOKEN = "[IMAGEM {i}]"


@dataclass
class Question:
    id: str  # único entre todas as fontes, ex.: "enem-2024-q017"
    source: str  # "enem" | "bluex" | "oab"
    exam: str  # ex.: "ENEM 2024", "USP 2023", "OAB 2016-20"
    year: int
    area: str  # ex.: "matematica", "linguagens", "direito"
    statement: str  # enunciado, com marcadores [IMAGEM i]
    alternatives: list[str]  # sem o prefixo "a)"; podem conter [IMAGEM i]
    answer: str  # letra correta (A-E)
    images: list[str] = field(default_factory=list)  # URL http(s) ou data URL base64
    captions: list[str] = field(default_factory=list)  # descrição textual de cada imagem
    subjects: list[str] = field(default_factory=list)

    @property
    def letters(self) -> str:
        return LETTERS[: len(self.alternatives)]

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Question:
        return cls(**d)


def save_jsonl(questions: Iterable[Question], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> list[Question]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[Question]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Question.from_dict(json.loads(line))
