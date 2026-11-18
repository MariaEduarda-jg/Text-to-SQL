from __future__ import annotations

import random
import re
from dataclasses import replace

from vestibular.schema import IMAGE_TOKEN, LETTERS, Question

MODES = ("caption", "image", "blind")

SYSTEM_COT = (
    "Você é um estudante brasileiro excelente resolvendo uma questão de prova "
    "(ENEM, vestibular ou OAB). Raciocine passo a passo, em português, de forma "
    "objetiva e sem enrolação. No final, escreva a resposta exatamente no formato:\n"
    "Resposta: X\n"
    "onde X é a letra da alternativa correta."
)

SYSTEM_DIRECT = (
    "Você é um estudante brasileiro excelente resolvendo uma questão de prova. "
    "Responda somente com a letra da alternativa correta, no formato:\nResposta: X"
)

_IMG_RE = re.compile(r"\[IMAGEM (\d+)\]")


def _render_images(text: str, q: Question, mode: str, used: set[int]) -> str:
    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        used.add(i)
        if mode == "caption":
            cap = q.captions[i] if i < len(q.captions) else "(sem descrição)"
            return f"[IMAGEM {i}: {cap}]"
        if mode == "blind":
            return "[imagem omitida]"
        return m.group(0)

    return _IMG_RE.sub(repl, text)


def render_question(q: Question, mode: str = "caption") -> str:
    """Texto da questão (enunciado + alternativas) no modo escolhido."""
    if mode not in MODES:
        raise ValueError(f"modo inválido: {mode}")
    used: set[int] = set()
    stmt = _render_images(q.statement, q, mode, used)
    alts = [_render_images(a, q, mode, used) for a in q.alternatives]
    body = stmt + "\n\n" + "\n".join(f"{LETTERS[i]}) {a}" for i, a in enumerate(alts))

    if mode == "caption":
        extra = [f"[IMAGEM {i}: {c}]" for i, c in enumerate(q.captions) if i not in used and c]
        if extra:
            body += "\n\n" + "\n".join(extra)
    return body


def build_messages(q: Question, mode: str = "caption", cot: bool = True) -> list[dict]:
    """Mensagens no formato OpenAI chat (funciona com OpenAI, Gemini, vLLM, HF router...)."""
    text = render_question(q, mode)
    system = SYSTEM_COT if cot else SYSTEM_DIRECT
    if mode == "image" and q.has_images:
        content: list[dict] = [{"type": "text", "text": text}]
        for i, url in enumerate(q.images):
            content.append({"type": "text", "text": IMAGE_TOKEN.format(i=i) + ":"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        user: dict = {"role": "user", "content": content}
    else:
        user = {"role": "user", "content": text}
    return [{"role": "system", "content": system}, user]


def permute_alternatives(q: Question, seed: int) -> Question:
    """Embaralha as alternativas e corrige o gabarito."""
    rng = random.Random(f"{q.id}-{seed}")
    order = list(range(len(q.alternatives)))
    while True:
        rng.shuffle(order)
        if order != list(range(len(q.alternatives))):
            break
    new_alts = [q.alternatives[i] for i in order]
    old_idx = LETTERS.index(q.answer)
    new_answer = LETTERS[order.index(old_idx)]
    return replace(q, id=f"{q.id}#perm{seed}", alternatives=new_alts, answer=new_answer)
