from __future__ import annotations

import re

_PATTERNS = [
    re.compile(
        r"resposta(?:\s+final)?\s*(?:correta)?\s*(?:é|e)?\s*[:：]?\s*\**\s*\(?([A-E])\)?(?![a-zA-Z])", re.I
    ),
    re.compile(r"\\boxed\{\s*\(?([A-E])\)?\s*\}"),
    re.compile(
        r"(?:alternativa|op[çc][ãa]o|letra)\s+(?:correta\s+)?(?:é\s+(?:a\s+)?)?\(?([A-E])\)?(?![a-zA-Z])",
        re.I,
    ),
]

FORMAT_RE = re.compile(r"Resposta:\s*([A-E])\s*$")


def extract_answer(text: str | None, valid: str = "ABCDE") -> str | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for pat in _PATTERNS:
        matches = pat.findall(text)
        for m in reversed(matches):
            letter = m.upper()
            if letter in valid:
                return letter
    stripped = text.strip().strip(".*()[] ").upper()
    if len(stripped) == 1 and stripped in valid:
        return stripped
    return None


def follows_format(text: str | None) -> bool:
    """True se a resposta termina exatamente em 'Resposta: X' (usado como recompensa no GRPO)."""
    return bool(text) and bool(FORMAT_RE.search(text.strip()))
