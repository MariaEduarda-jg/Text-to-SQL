"""Extrai a letra escolhida da resposta livre do modelo.

Modelos não seguem o formato 100% das vezes. Tentamos, em ordem de confiança:
1. "Resposta: X" (o formato pedido) — usamos a ÚLTIMA ocorrência
2. \\boxed{X}
3. frases como "alternativa correta é a X", "letra X"
4. a resposta inteira é só uma letra
Se nada casar, devolvemos None (conta como erro na avaliação).
"""

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
    # Ignora o bloco de "pensamento" de modelos de raciocínio, se vier junto.
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
