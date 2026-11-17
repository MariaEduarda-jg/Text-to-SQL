"""Etapa 2 — GRPO: aprendizado por reforço com recompensa verificável.

Para cada questão o modelo gera G respostas (um "grupo"). A vantagem de cada
resposta é (recompensa - média do grupo) / desvio do grupo. Respostas que
acertam são reforçadas, as que erram são desencorajadas — sem modelo de
recompensa e sem modelo "crítico" (é isso que torna o GRPO barato).

Filtro de dificuldade (opcional, mas recomendado): questões que o modelo SEMPRE
acerta ou SEMPRE erra geram vantagem zero para o grupo todo — não ensinam nada
e gastam GPU. Com --difficulty-from passamos um arquivo de predições do modelo
base (n amostras por questão) e treinamos só nas questões "no limite" (0 < acerto < 1).

Rodar na GPU (Kaggle/Colab):
    python -m vestibular.train.grpo --out outputs/grpo                      # a partir do modelo base
    python -m vestibular.train.grpo --init-adapter outputs/sft --out outputs/sft_grpo
"""

from __future__ import annotations

import argparse
import json


def _difficulty_filter(path: str) -> set[str]:
    keep = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        preds = r.get("preds") or []
        k = sum(p == r["gold"] for p in preds)
        if 0 < k < len(preds):
            keep.add(r["qid"])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--init-adapter", default=None, help="adaptador LoRA do SFT para continuar a partir dele")
    ap.add_argument("--split", default="data/processed/train.jsonl")
    ap.add_argument(
        "--difficulty-from", default=None, help="predictions.jsonl do modelo base com n>1 amostras"
    )
    ap.add_argument("--out", default="outputs/grpo")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--max-completion-length", type=int, default=768)
    ap.add_argument("--push-to", default=None)
    args = ap.parse_args()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from vestibular.prompts import build_messages
    from vestibular.schema import load_jsonl
    from vestibular.train.common import bf16_ok, load_model, render_prompt, text_tokenizer
    from vestibular.train.rewards import correctness_reward, format_reward, make_length_penalty

    max_seq = args.max_prompt_length + args.max_completion_length
    model, tokenizer = load_model(args.model, max_seq, args.lora_rank)
    if args.init_adapter:
        # Carrega os pesos LoRA do SFT por cima do adaptador recém-criado.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(f"{args.init_adapter}/adapter_model.safetensors"))
        print(f"adaptador inicial carregado de {args.init_adapter}")
    tok = text_tokenizer(tokenizer)

    questions = load_jsonl(args.split)
    if args.difficulty_from:
        keep = _difficulty_filter(args.difficulty_from)
        questions = [q for q in questions if q.id in keep]
        print(f"filtro de dificuldade: {len(questions)} questões 'no limite'")

    rows = []
    for q in questions:
        prompt = render_prompt(tokenizer, build_messages(q, mode="caption", cot=True))
        if len(tok(prompt).input_ids) <= args.max_prompt_length:
            rows.append({"prompt": prompt, "answer": q.answer, "letters": q.letters, "qid": q.id})
    print(f"{len(rows)} prompts de treino (após limite de tamanho)")
    ds = Dataset.from_list(rows).shuffle(seed=42)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_8bit",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=0.0,  # sem penalidade KL (prática comum hoje; economiza memória do modelo de referência)
        loss_type="dr_grpo",  # remove o viés de tamanho da resposta do GRPO original
        max_steps=args.max_steps,
        save_steps=50,
        logging_steps=1,
        bf16=bf16_ok(),
        fp16=not bf16_ok(),
        report_to="none",
        seed=42,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[correctness_reward, format_reward, make_length_penalty()],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    # Salva o histórico (recompensa média por passo etc.) para os gráficos do relatório.
    with open(f"{args.out}/log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    if args.push_to:
        model.push_to_hub(args.push_to)
        tok.push_to_hub(args.push_to)


if __name__ == "__main__":
    main()
