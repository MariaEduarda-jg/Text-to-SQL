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
