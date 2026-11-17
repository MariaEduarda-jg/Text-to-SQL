"""Backends de geração. Todos recebem mensagens no formato OpenAI chat.

Especificação de modelo como string "provedor:modelo", ex.:
    gemini:gemini-2.5-flash
    openai:gpt-4.1-mini
    hf:Qwen/Qwen3.5-27B            (Hugging Face Inference Providers)
    local:meu-modelo                (qualquer servidor OpenAI-compatível em localhost:8000)
    vllm:Qwen/Qwen3.5-2B            (vLLM offline na GPU — Kaggle/Colab)
    mock:random                     (chute aleatório — baseline e testes)
"""

from __future__ import annotations

from vestibular.backends.base import Backend, Completion


def get_backend(spec: str, **kwargs) -> Backend:
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"Use o formato provedor:modelo (recebi {spec!r})")
    if provider == "mock":
        from vestibular.backends.mock import RandomBackend

        return RandomBackend(model)
    if provider == "vllm":
        from vestibular.backends.vllm_local import VLLMBackend

        return VLLMBackend(model, **kwargs)
    from vestibular.backends.openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(provider, model, **kwargs)


__all__ = ["Backend", "Completion", "get_backend"]
