from __future__ import annotations

DEFAULT_MODEL = "unsloth/Qwen3.5-2B"


def load_model(model_name: str, max_seq_length: int, lora_rank: int, load_in_4bit: bool = True):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def text_tokenizer(tokenizer):
    """Modelos multimodais devolvem um *processor*; o tokenizer de texto fica dentro dele."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def bf16_ok() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
