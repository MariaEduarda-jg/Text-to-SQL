# Roteiro do artigo técnico

**Título sugerido:** *"Ensinei um modelo de 2B a fazer o ENEM com aprendizado por reforço: o que funcionou, o que não funcionou e o que os modelos decoraram"*

## 1. Gancho (1 parágrafo)
Um número forte. Exemplo: "o Qwen3.5-2B foi de X% para Y% no ENEM com 4 horas de GPU gratuita".

## 2. Por que provas brasileiras
- Português, contexto cultural, interpretação de texto, figuras.
- Dados abertos: ENEM (Maritaca), BLUEX, OAB.
- O ENEM é um benchmark *verificável*: resposta certa conhecida, então serve para RL.

## 3. Montando uma avaliação confiável
- Divisão por ano e checagem de vazamento.
- Intervalo de confiança e McNemar: "por que 2 pontos de diferença podem não significar nada".
- Baseline aleatório.

## 4. Experimento 1: SFT vs. GRPO vs. SFT+GRPO
- Explicação curta do GRPO com um diagrama de grupo → vantagem.
- Gráfico da curva de recompensa.
- Tabela de resultados. Discussão: em quais áreas o RL ajudou mais? (Matemática? Humanas?)
- Exemplo concreto de questão: raciocínio antes e depois do treino.

## 5. Experimento 2: imagem vs. descrição
- Gráfico de barras blind / caption / image.
- O resultado de 2023 se mantém?

## 6. Experimento 3: contaminação
- Queda de acurácia com alternativas embaralhadas, provas antigas vs. recentes.
- Ressalvas honestas sobre o que o teste mede.

## 7. Engenharia
- Avaliação retomável, async com retry, vLLM + LoRA sem merge, CI com smoke test, demo.

## 8. Limitações e próximos passos

## Links
Repo · modelos · dataset · demo
