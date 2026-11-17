"""Funções de recompensa para o GRPO (formato esperado pelo TRL GRPOTrainer).

A ideia central do GRPO com "recompensa verificável": em múltipla escolha a
resposta certa é conhecida, então não precisamos de um modelo de recompensa
(que poderia ser enganado). Para cada questão o modelo gera um GRUPO de
respostas; as que acertam recebem vantagem positiva, as que erram, negativa.

O TRL chama cada função com:
    completions: lista de textos gerados (prompts em formato texto)
    + cada coluna do dataset como kwarg (aqui: `answer`, `letters`)
e espera uma lista de floats, um por completion.
"""

from __future__ import annotations

from vestibular.parsing import extract_answer, follows_format


def _text(c) -> str:
    # Com prompts conversacionais o TRL passa [{"role": "assistant", "content": ...}]
    if isinstance(c, list):
        return c[-1].get("content", "") if c else ""
    return c or ""


def correctness_reward(completions, answer, letters=None, **kwargs) -> list[float]:
    """+1 se a letra extraída é a correta, 0 caso contrário."""
    letters = letters or ["ABCDE"] * len(completions)
    return [
        1.0 if extract_answer(_text(c), valid) == gold else 0.0
        for c, gold, valid in zip(completions, answer, letters, strict=True)
    ]


def format_reward(completions, **kwargs) -> list[float]:
    """+0.1 se termina exatamente em 'Resposta: X'.

    Peso pequeno de propósito: queremos o formato, mas sem o modelo aprender
    a priorizar formato em vez de acertar.
    """
    return [0.1 if follows_format(_text(c)) else 0.0 for c in completions]


def make_length_penalty(max_chars: int = 3500, weight: float = 0.1):
    """Penaliza respostas longas demais (raciocínio que se arrasta costuma ser cortado e perder a resposta)."""

    def length_penalty(completions, **kwargs) -> list[float]:
        out = []
        for c in completions:
            n = len(_text(c))
            out.append(-weight * min(1.0, (n - max_chars) / max_chars) if n > max_chars else 0.0)
        return out

    return length_penalty
