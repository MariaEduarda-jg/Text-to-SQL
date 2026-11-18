from __future__ import annotations

import asyncio
import os
import random
import time

from openai import AsyncOpenAI

from vestibular.backends.base import Backend, Completion

PROVIDERS: dict[str, tuple[str | None, str]] = {
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "hf": ("https://router.huggingface.co/v1", "HF_TOKEN"),
    "maritaca": ("https://chat.maritaca.ai/api", "MARITACA_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "local": (os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1"), "LOCAL_API_KEY"),
}


class OpenAICompatBackend(Backend):
    def __init__(
        self, provider: str, model: str, concurrency: int = 8, max_retries: int = 8, rpm: int | None = None
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Provedor desconhecido {provider!r}. Opções: {', '.join(PROVIDERS)}")
        base_url, key_env = PROVIDERS[provider]
        api_key = os.environ.get(key_env) or ("EMPTY" if provider == "local" else None)
        if not api_key:
            raise RuntimeError(f"Defina a variável de ambiente {key_env} (veja .env.example)")
        self.name = f"{provider}:{model}"
        self.model = model
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.min_interval = 60 / rpm if rpm else 0.0
        self._next_slot = 0.0
        self._client_kwargs = {"base_url": base_url, "api_key": api_key, "timeout": 180, "max_retries": 0}

    async def _one(self, client: AsyncOpenAI, sem: asyncio.Semaphore, messages, temperature, max_tokens):
        async with sem:
            for attempt in range(self.max_retries):
                await self._throttle()
                t0 = time.perf_counter()
                try:
                    r = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    usage = r.usage
                    return Completion(
                        text=r.choices[0].message.content or "",
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        latency_s=time.perf_counter() - t0,
                    )
                except Exception as e:  # noqa: BLE001
                    status = getattr(e, "status_code", None)
                    retriable = status is None or status == 429 or status >= 500
                    if not retriable or attempt == self.max_retries - 1:
                        return Completion(text="", error=f"{type(e).__name__}: {e}"[:500])
                    await asyncio.sleep(min(60, 2 ** (attempt + 1) + random.random()))
        return Completion(text="", error="unreachable")

    async def _throttle(self):
        if not self.min_interval:
            return
        now = time.monotonic()
        wait = self._next_slot - now
        self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def _run(self, conversations, n, temperature, max_tokens):
        client = AsyncOpenAI(**self._client_kwargs)
        sem = asyncio.Semaphore(self.concurrency)
        tasks = [
            self._one(client, sem, conv, temperature, max_tokens) for conv in conversations for _ in range(n)
        ]
        flat = await asyncio.gather(*tasks)
        await client.close()
        return [flat[i * n : (i + 1) * n] for i in range(len(conversations))]

    def generate(self, conversations, *, n=1, temperature=0.0, max_tokens=2048):
        return asyncio.run(self._run(conversations, n, temperature, max_tokens))
