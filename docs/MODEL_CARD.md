---
language: [pt]
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
tags: [grpo, reasoning, enem, portuguese, unsloth, trl]
datasets: [SEU_USUARIO/vestibular-br-data]
---

# vestibular-br-2b-{sft|grpo|sft-grpo}

Adaptador LoRA para **Qwen3.5-2B**, treinado para resolver questões de provas brasileiras com raciocínio passo a passo.

## Treino
- **SFT**: raciocínios de um modelo professor filtrados por correção (rejection sampling).
- **GRPO**: recompensa verificável (+1 resposta correta, +0.1 formato), `loss_type=dr_grpo`, `beta=0`, 250 passos, GPU T4.
- Dados de treino: OAB + USP/Unicamp até 2022. **Nenhuma questão do ENEM.**

## Resultados (teste: ENEM 2022–2024 + USP/Unicamp 2024–2025)
| Modelo | ENEM | USP/Unicamp |
|---|---|---|
| base | – | – |
| este | – | – |

## Uso
```python
from vllm import LLM
from vllm.lora.request import LoRARequest
llm = LLM("Qwen/Qwen3.5-2B", enable_lora=True)
# formato do prompt: ver vestibular.prompts.build_messages
```

## Limitações
Modelo pequeno, voltado à pesquisa. Não use para decisões educacionais reais.
