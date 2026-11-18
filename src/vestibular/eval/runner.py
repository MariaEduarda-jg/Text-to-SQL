from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from vestibular.backends import Backend
from vestibular.images import inline_images
from vestibular.parsing import extract_answer
from vestibular.prompts import build_messages, permute_alternatives
from vestibular.schema import Question

RESULTS_DIR = Path("results/runs")


@dataclass
class RunConfig:
    benchmark: str
    model: str
    mode: str = "caption"
    cot: bool = True
    n_samples: int = 1
    temperature: float = 0.0
    max_tokens: int = 2048
    permutation_seed: int | None = None
    limit: int | None = None
    tag: str = ""

    @property
    def run_name(self) -> str:
        slug = re.sub(r"[^a-zA-Z0-9.-]+", "-", self.model).strip("-")
        parts = [self.benchmark, slug, self.mode]
        if not self.cot:
            parts.append("direct")
        if self.permutation_seed is not None:
            parts.append(f"perm{self.permutation_seed}")
        if self.tag:
            parts.append(self.tag)
        return "__".join(parts)


def run_eval(
    questions: list[Question],
    backend: Backend,
    cfg: RunConfig,
    out_dir: Path = RESULTS_DIR,
    batch_size: int = 32,
) -> Path:
    run_dir = out_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    (run_dir / "config.json").write_text(
        json.dumps(cfg.__dict__, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if cfg.limit:
        questions = questions[: cfg.limit]
    if cfg.permutation_seed is not None:
        questions = [permute_alternatives(q, cfg.permutation_seed) for q in questions]

    done: set[str] = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("error"):
                    done.add(rec["qid"])
    todo = [q for q in questions if q.id not in done]
    print(f"[{cfg.run_name}] {len(done)} já feitas, {len(todo)} restantes")

    with pred_path.open("a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(todo), batch_size), desc=cfg.run_name[:40]):
            batch = todo[start : start + batch_size]
            if cfg.mode == "image":
                batch = [inline_images(q) for q in batch]
            convs = [build_messages(q, cfg.mode, cfg.cot) for q in batch]
            t0 = time.perf_counter()
            outs = backend.generate(
                convs, n=cfg.n_samples, temperature=cfg.temperature, max_tokens=cfg.max_tokens
            )
            for q, samples in zip(batch, outs, strict=True):
                preds = [extract_answer(s.text, q.letters) for s in samples]
                rec = {
                    "qid": q.id,
                    "source": q.source,
                    "exam": q.exam,
                    "year": q.year,
                    "area": q.area,
                    "has_images": q.has_images,
                    "gold": q.answer,
                    "preds": preds,
                    "pred": _majority(preds),
                    "correct": _majority(preds) == q.answer,
                    "outputs": [s.text for s in samples],
                    "prompt_tokens": sum(s.prompt_tokens for s in samples),
                    "completion_tokens": sum(s.completion_tokens for s in samples),
                    "latency_s": round(sum(s.latency_s for s in samples) / len(samples), 3),
                    "error": next((s.error for s in samples if s.error), None),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  lote {start // batch_size + 1}: {time.perf_counter() - t0:.1f}s")
    return pred_path


def _majority(preds: list[str | None]) -> str | None:
    """Voto da maioria entre as amostras (self-consistency). Empate → primeira."""
    valid = [p for p in preds if p]
    if not valid:
        return None
    return max(dict.fromkeys(valid), key=valid.count)


def load_predictions(path: str | Path, include_errors: bool = False) -> list[dict]:
    """Lê predições; se a mesma questão aparece mais de uma vez (retry), fica a última."""
    by_q: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("error") and rec["qid"] in by_q and not by_q[rec["qid"]].get("error"):
                continue
            by_q[rec["qid"]] = rec
    if include_errors:
        return list(by_q.values())
    return [r for r in by_q.values() if not r.get("error")]


def count_errors(path: str | Path) -> int:
    return sum(1 for r in load_predictions(path, include_errors=True) if r.get("error"))
