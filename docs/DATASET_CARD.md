---
language: [pt]
license: apache-2.0
task_categories: [multiple-choice, question-answering]
tags: [enem, vestibular, oab, brazil, reasoning, grpo]
pretty_name: Vestibular-BR
---

# Vestibular-BR — dados

Versão unificada e dividida (sem vazamento) de três benchmarks brasileiros, mais um dataset de raciocínios destilados.

## Conteúdo
| Arquivo | Descrição |
|---|---|
| `splits/train.jsonl` | OAB (2010–2018) + USP/Unicamp 2018–2022 |
| `splits/dev.jsonl` | USP/Unicamp 2023 |
| `splits/test.jsonl` | ENEM 2022–2024 + USP/Unicamp 2024–2025 |
| `splits/enem_*.jsonl`, `bluex.jsonl`, `oab.jsonl` | fontes completas no formato unificado |
| `sft/train_sft.jsonl` | raciocínios do modelo professor **verificados pelo gabarito** (rejection sampling) |

## Esquema
`id, source, exam, year, area, statement, alternatives, answer, images, captions, subjects`.
As imagens aparecem no texto como `[IMAGEM i]`, com `captions[i]` trazendo a descrição textual.

## Fontes e licenças
- ENEM: [maritaca-ai/enem](https://huggingface.co/datasets/maritaca-ai/enem) (Apache-2.0)
- BLUEX: [portuguese-benchmark-datasets/BLUEX](https://huggingface.co/datasets/portuguese-benchmark-datasets/BLUEX)
- OAB: [eduagarcia/oab_exams](https://huggingface.co/datasets/eduagarcia/oab_exams)

Cite os trabalhos originais ao usar estes dados.
