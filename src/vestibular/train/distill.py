"""Gera o dataset de SFT por *rejection sampling* (destilação de raciocínio).

1. Um modelo "professor" grande (via API) resolve cada questão de treino k vezes.
2. Guardamos só os raciocínios que chegaram na resposta CERTA e no formato certo.
3. Entre os corretos, escolhemos o mais curto (raciocínio direto ao ponto).

Resultado: exemplos de "como raciocinar" verificados pelo gabarito — sem
precisar de anotação humana. É a mesma ideia do STaR / rejection sampling
usados no pós-treino de modelos de raciocínio.

Também salvamos a taxa de acerto do professor por questão (`teacher_pass_rate`),
um proxy de dificuldade útil para análise e para filtrar o GRPO.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from vestibular.backends import Backend
from vestibular.parsing import extract_answer, follows_format
from vestibular.prompts import build_messages
from vestibular.schema import Question


def build_sft_dataset(
    questions: list[Question],
    teacher: Backend,
    out_path: str | Path,
    k: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    batch_size: int = 16,
) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = out_path.with_suffix(".stats.jsonl")

    done = set()
    if stats_path.exists():
        done = {json.loads(line)["qid"] for line in stats_path.open(encoding="utf-8")}
    todo = [q for q in questions if q.id not in done]
    print(f"{len(done)} já processadas, {len(todo)} restantes")

    with out_path.open("a", encoding="utf-8") as f_sft, stats_path.open("a", encoding="utf-8") as f_stats:
        for start in tqdm(range(0, len(todo), batch_size), desc="destilando"):
            batch = todo[start : start + batch_size]
            convs = [build_messages(q, mode="caption", cot=True) for q in batch]
            outs = teacher.generate(convs, n=k, temperature=temperature, max_tokens=max_tokens)
            for q, conv, samples in zip(batch, convs, outs, strict=True):
                texts = [s.text for s in samples if not s.error]
                good = [t for t in texts if extract_answer(t, q.letters) == q.answer and follows_format(t)]
                n_correct = sum(extract_answer(t, q.letters) == q.answer for t in texts)
                f_stats.write(
                    json.dumps(
                        {
                            "qid": q.id,
                            "n": len(texts),
                            "correct": n_correct,
                            "teacher_pass_rate": n_correct / max(1, len(texts)),
                        }
                    )
                    + "\n"
                )
                if good:
                    best = min(good, key=len)
                    f_sft.write(
                        json.dumps(
                            {
                                "qid": q.id,
                                "source": q.source,
                                "prompt": conv,
                                "completion": [{"role": "assistant", "content": best.strip()}],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            f_sft.flush()
            f_stats.flush()

    stats = [json.loads(line) for line in stats_path.open(encoding="utf-8")]
    kept = sum(1 for _ in out_path.open(encoding="utf-8"))
    summary = {
        "questions": len(stats),
        "kept_examples": kept,
        "teacher_acc_mean": sum(s["teacher_pass_rate"] for s in stats) / max(1, len(stats)),
    }
    print(summary)
    return summary
