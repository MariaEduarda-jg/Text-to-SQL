"""Etapa 1 — SFT: ensina o modelo pequeno a imitar os raciocínios corretos do professor.

Rodar na GPU (Kaggle/Colab):
    python -m vestibular.train.sft --data data/sft/train_sft.jsonl --out outputs/sft
"""

from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--data", default="data/sft/train_sft.jsonl")
    ap.add_argument("--out", default="outputs/sft")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=3072)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--push-to", default=None, help="repo no HF Hub, ex.: seu-usuario/vestibular-br-2b-sft")
    args = ap.parse_args()

    from datasets import Dataset  # disponível no Kaggle/Colab
    from trl import SFTConfig, SFTTrainer

    from vestibular.train.common import bf16_ok, load_model, render_prompt, text_tokenizer

    model, tokenizer = load_model(args.model, args.max_seq_length, args.lora_rank)
    tok = text_tokenizer(tokenizer)

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    data = [
        {
            "prompt": render_prompt(tokenizer, r["prompt"]),
            "completion": r["completion"][0]["content"] + tok.eos_token,
        }
        for r in rows
    ]
    # Remove exemplos que não cabem no contexto (seriam truncados e ensinariam errado).
    data = [d for d in data if len(tok(d["prompt"] + d["completion"]).input_ids) <= args.max_seq_length]
    print(f"{len(data)} exemplos de treino")
    ds = Dataset.from_list(data).shuffle(seed=42)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        completion_only_loss=True,  # perda só na resposta, não no enunciado
        bf16=bf16_ok(),
        fp16=not bf16_ok(),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    if args.push_to:
        model.push_to_hub(args.push_to)
        tok.push_to_hub(args.push_to)


if __name__ == "__main__":
    main()
