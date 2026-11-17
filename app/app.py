"""Demo pública (Hugging Face Space, Gradio).

Abas:
  1. Leaderboard  — tabela de todos os modelos avaliados.
  2. Explorador   — escolha uma questão e compare o raciocínio de cada modelo
                    lado a lado (ótimo para análise de erros).
  3. Pergunte     — cole uma questão e um modelo grande responde ao vivo
                    (só funciona se o Space tiver o secret HF_TOKEN).

Rodar local:  python app/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vestibular.eval.runner import load_predictions  # noqa: E402
from vestibular.parsing import extract_answer  # noqa: E402
from vestibular.prompts import SYSTEM_COT, render_question  # noqa: E402
from vestibular.schema import load_jsonl  # noqa: E402

RUNS = ROOT / "results" / "runs"
LEADERBOARD = ROOT / "results" / "leaderboard.json"
QUESTIONS = ROOT / "data" / "processed" / "test.jsonl"
LIVE_MODEL = os.environ.get("LIVE_MODEL", "Qwen/Qwen3.5-27B")


def _load_questions() -> dict:
    if QUESTIONS.exists():
        return {q.id: q for q in load_jsonl(QUESTIONS)}
    from vestibular.data.loaders import ENEM_YEARS, load_enem

    return {q.id: q for y in ENEM_YEARS for q in load_enem(y)}


QS = _load_questions()


def _load_runs() -> dict[str, dict]:
    runs = {}
    for d in sorted(RUNS.glob("*")):
        cfg = d / "config.json"
        if cfg.exists() and "perm" not in d.name:
            runs[d.name] = {r["qid"]: r for r in load_predictions(d / "predictions.jsonl")}
    return runs


RUN_PREDS = _load_runs()


def leaderboard_rows():
    if not LEADERBOARD.exists():
        return [["—", "rode `vbr report`", "", "", ""]]
    rows = json.loads(LEADERBOARD.read_text(encoding="utf-8"))
    return [
        [r["benchmark"], r["model"], r["mode"], r["n"], f"{r['acc']} ({r['ci95'][0]}–{r['ci95'][1]})"]
        for r in rows
    ]


def question_choices():
    ids = sorted({qid for preds in RUN_PREDS.values() for qid in preds} & QS.keys()) or sorted(QS)
    return [(f"{QS[i].exam} · {QS[i].area} · {i.split('-')[-1]}", i) for i in ids]


def show_question(qid: str):
    if not qid:
        return "", ""
    q = QS[qid]
    md = f"### {q.exam} — {q.area}\n\n" + render_question(q, "caption").replace("\n", "  \n")
    md += f"\n\n**Gabarito: {q.answer}**"
    blocks = []
    for name, preds in RUN_PREDS.items():
        r = preds.get(qid)
        if not r:
            continue
        mark = "✅" if r["correct"] else "❌"
        out = (r["outputs"][0] if r["outputs"] else "").strip() or "(vazio)"
        blocks.append(
            f"#### {mark} `{name}` → {r['pred']}\n\n<details><summary>raciocínio</summary>\n\n{out}\n\n</details>"
        )
    return md, "\n\n".join(blocks) or "_nenhum modelo avaliado nesta questão ainda_"


def ask_live(question_text: str):
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "Este Space não tem HF_TOKEN configurado; modo ao vivo desativado."
    from openai import OpenAI

    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)
    r = client.chat.completions.create(
        model=LIVE_MODEL,
        messages=[{"role": "system", "content": SYSTEM_COT}, {"role": "user", "content": question_text}],
        max_tokens=1500,
        temperature=0.0,
    )
    text = r.choices[0].message.content or ""
    return f"{text}\n\n---\n**Letra extraída:** {extract_answer(text)}"


with gr.Blocks(title="Vestibular-BR") as demo:
    gr.Markdown(
        "# 🎓 Vestibular-BR\nUm modelo pequeno (2B) aprendendo a raciocinar em provas brasileiras com GRPO. "
        "Avaliação em ENEM 2022–2024 e USP/Unicamp 2024–2025."
    )
    with gr.Tab("Leaderboard"):
        gr.Dataframe(
            value=leaderboard_rows(),
            headers=["Benchmark", "Modelo", "Modo", "N", "Acurácia % (IC95%)"],
            interactive=False,
        )
    with gr.Tab("Explorador de questões"):
        sel = gr.Dropdown(choices=question_choices(), label="Questão")
        with gr.Row():
            q_md = gr.Markdown()
            a_md = gr.Markdown()
        sel.change(show_question, sel, [q_md, a_md])
    with gr.Tab("Pergunte ao vivo"):
        inp = gr.Textbox(lines=10, label="Cole o enunciado e as alternativas (A) ... E) ...)")
        btn = gr.Button("Resolver")
        out = gr.Markdown()
        btn.click(ask_live, inp, out)

if __name__ == "__main__":
    demo.launch()
