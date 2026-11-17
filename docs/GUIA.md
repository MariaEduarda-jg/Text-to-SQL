# Guia de execução (passo a passo)

Ordem sugerida para terminar em ~6 dias. Cada etapa termina com um commit.

## Dia 0: contas (30 min)
1. **GitHub**: crie o repositório `vestibular-br` (público).
2. **Hugging Face**: crie a conta e gere 2 tokens em Settings → Access Tokens: um *read* e um *write*.
3. **Kaggle**: crie a conta e **verifique o telefone**. Sem isso não libera GPU nem internet nos notebooks.
4. **Google AI Studio**: gere uma `GEMINI_API_KEY` (plano gratuito).
5. Copie `.env.example` para `.env` e preencha. **Nunca commite o `.env`.**

```bash
git init && git add . && git commit -m "Estrutura inicial: dados, avaliação, treino"
git branch -M main && git remote add origin https://github.com/SEU_USUARIO/vestibular-br.git
git push -u origin main
```

## Dia 1: baseline com API
```bash
vbr data
vbr eval -b test -m gemini:gemini-2.5-flash --limit 20    # teste rápido
vbr eval -b test -m gemini:gemini-2.5-flash
vbr report
```
⚠️ O plano gratuito do Gemini tem limite de requisições por minuto e por dia. O avaliador **retoma de onde parou**, então se der erro 429 é só rodar de novo mais tarde. Se travar muito, use `--concurrency 2`.
Commit: `results/runs/...` + `results/LEADERBOARD.md`.

## Dia 2: dataset de SFT (destilação)
```bash
vbr distill -m gemini:gemini-2.5-flash --k 4 --limit 50   # teste
vbr distill -m gemini:gemini-2.5-flash --k 4              # completo (~3.100 questões x 4)
vbr publish --repo SEU_USUARIO_HF/vestibular-br-data
```
São ~12 mil chamadas, o que provavelmente leva mais de um dia no plano gratuito. Alternativas:
- `--k 2` (metade das chamadas);
- usar só a OAB + metade do BLUEX (`--limit 1500`);
- usar um modelo barato via OpenRouter (alguns dólares no total).

Commit: `data/sft/train_sft.jsonl` (é pequeno e vira o dataset publicado).

## Dia 2–3: avaliação dos modelos base no Kaggle
Abra `notebooks/01_avaliacao_vllm_kaggle.ipynb` no Kaggle (File → Import Notebook) e rode as seções 1–4.
Baixe `resultados.zip`, extraia `results/` no repositório, rode `vbr report` e commite.

## Dia 3–4: treino
`notebooks/02_treino_sft_grpo_kaggle.ipynb`:
1. SFT (~1h)
2. GRPO a partir do base: comece com `--max-steps 30` para medir quanto tempo leva cada passo.
3. SFT + GRPO

Depois volte ao notebook 01, seção 5, para avaliar os 3 modelos treinados.

## Dia 5: demo + resultados
1. `vbr report`. Os gráficos ficam em `results/figures/`.
2. Preencha a tabela do README com os números de `results/LEADERBOARD.md`.
3. **HF Space**: crie um Space Gradio e suba `app/`, `src/`, `results/`, `data/processed/test.jsonl` e um `requirements.txt` com:
   ```
   gradio
   polars
   huggingface_hub
   openai
   -e .
   ```
   No Space, o arquivo principal precisa se chamar `app.py` na raiz (copie `app/app.py` e ajuste `ROOT`).

## Dia 6: artigo
Use `docs/ARTIGO.md` como roteiro. Publique no Medium ou no dev.to e poste no LinkedIn com o gráfico principal.
Opcional: publique `notebooks/01...` como notebook público no Kaggle.
