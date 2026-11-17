"""Backend vLLM offline — para avaliar os modelos treinados na GPU (Kaggle/Colab).

Aceita um adaptador LoRA opcional, então dá para avaliar o modelo base e o
modelo treinado sem precisar fazer merge dos pesos.
"""

from __future__ import annotations

import time

from vestibular.backends.base import Backend, Completion


class VLLMBackend(Backend):
    def __init__(
        self,
        model: str,
        lora: str | None = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.88,
        dtype: str = "auto",
        limit_images: int = 8,
        enable_thinking: bool = False,
    ):
        import os

        import torch
        from vllm import LLM  # import tardio: vLLM só existe no ambiente com GPU

        self.name = f"vllm:{model}" + (f"+{lora}" if lora else "")
        if lora and not os.path.exists(lora):  # repo do HF Hub → baixa para disco
            from huggingface_hub import snapshot_download

            lora = snapshot_download(lora)
        self.lora = lora
        self.enable_thinking = enable_thinking
        if dtype == "auto" and not torch.cuda.is_bf16_supported():
            dtype = "float16"  # GPUs T4/P100 (Kaggle/Colab grátis) não suportam bf16
        # Kaggle "T4 x2": VLLM_TP=2 divide o modelo entre as duas GPUs (cabe o 9B).
        tp = int(os.environ.get("VLLM_TP", "1"))
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enable_lora=lora is not None,
            max_lora_rank=64,
            limit_mm_per_prompt={"image": limit_images},
            trust_remote_code=True,
        )

    def generate(self, conversations, *, n=1, temperature=0.0, max_tokens=2048):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        params = SamplingParams(
            n=n, temperature=temperature, max_tokens=max_tokens, top_p=0.95 if temperature else 1.0
        )
        lora_req = LoRARequest("adapter", 1, self.lora) if self.lora else None
        t0 = time.perf_counter()
        outs = self.llm.chat(
            conversations,
            sampling_params=params,
            lora_request=lora_req,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            use_tqdm=True,
        )
        per_item = (time.perf_counter() - t0) / max(1, len(conversations))
        results = []
        for out in outs:
            p_tok = len(out.prompt_token_ids or [])
            results.append(
                [
                    Completion(
                        text=o.text,
                        prompt_tokens=p_tok,
                        completion_tokens=len(o.token_ids),
                        latency_s=per_item,
                    )
                    for o in out.outputs
                ]
            )
        return results
