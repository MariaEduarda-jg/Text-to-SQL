from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

LETTERS = "ABCDE"

IMAGE_TOKEN = "[IMAGEM {i}]"


@dataclass
class Question:
    id: str
    source: str
    exam: str
    year: int
    area: str
    statement: str
    alternatives: list[str]
    answer: str
    images: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
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
