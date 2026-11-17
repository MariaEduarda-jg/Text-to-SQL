"""Baixa imagens remotas e converte para data URL base64 (com cache em disco).

Várias APIs não aceitam URL externa de imagem, ou falham quando o GitHub
demora. Mandar base64 é o caminho mais robusto.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import urllib.request
from dataclasses import replace
from pathlib import Path

from vestibular.schema import Question

CACHE = Path("data/images")


def to_data_url(url: str) -> str:
    if url.startswith("data:"):
        return url
    CACHE.mkdir(parents=True, exist_ok=True)
    ext = Path(url.split("?")[0]).suffix or ".png"
    path = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ext)
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "vestibular-br/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            path.write_bytes(r.read())
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def inline_images(q: Question) -> Question:
    if not q.images:
        return q
    return replace(q, images=[to_data_url(u) for u in q.images])
