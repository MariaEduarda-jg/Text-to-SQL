from __future__ import annotations

import math
import random
from collections import defaultdict


def accuracy(records: list[dict]) -> float:
    return sum(r["correct"] for r in records) / len(records) if records else float("nan")


def bootstrap_ci(
    records: list[dict], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if not records:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    xs = [1 if r["correct"] else 0 for r in records]
    n = len(xs)
    stats = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return stats[int(alpha / 2 * n_boot)], stats[int((1 - alpha / 2) * n_boot) - 1]


def by_group(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r[key])].append(r)
    return {k: {"n": len(v), "acc": accuracy(v), "ci95": bootstrap_ci(v)} for k, v in sorted(groups.items())}


def summarize(records: list[dict]) -> dict:
    n = len(records)
    return {
        "n": n,
        "acc": accuracy(records),
        "ci95": bootstrap_ci(records),
        "unparsed_rate": sum(r["pred"] is None for r in records) / n if n else 0.0,
        "error_rate": sum(bool(r.get("error")) for r in records) / n if n else 0.0,
        "avg_completion_tokens": sum(r.get("completion_tokens", 0) for r in records) / n if n else 0.0,
        "by_area": by_group(records, "area"),
        "by_exam": by_group(records, "exam"),
        "by_images": by_group(records, "has_images"),
    }


def mcnemar_exact(a: list[dict], b: list[dict]) -> dict:
    """Compara dois modelos nas mesmas questões."""
    ma = {r["qid"].split("#")[0]: r["correct"] for r in a}
    mb = {r["qid"].split("#")[0]: r["correct"] for r in b}
    common = ma.keys() & mb.keys()
    b01 = sum(1 for q in common if not ma[q] and mb[q])
    b10 = sum(1 for q in common if ma[q] and not mb[q])
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {
        "n_common": len(common),
        "a_only": b10,
        "b_only": b01,
        "acc_a": sum(ma[q] for q in common) / len(common) if common else float("nan"),
        "acc_b": sum(mb[q] for q in common) / len(common) if common else float("nan"),
        "p_value": p,
    }


def permutation_consistency(original: list[dict], permuted: list[list[dict]]) -> dict:
    """Teste de contaminação por permutação das alternativas."""
    orig = {r["qid"]: r["correct"] for r in original}
    perm_maps = [{r["qid"].split("#")[0]: r["correct"] for r in run} for run in permuted]
    common = [q for q in orig if all(q in m for m in perm_maps)]
    if not common:
        return {"n": 0}
    acc_orig = sum(orig[q] for q in common) / len(common)
    acc_perm = sum(sum(m[q] for q in common) / len(common) for m in perm_maps) / len(perm_maps)
    correct_orig = [q for q in common if orig[q]]
    fragile = sum(1 for q in correct_orig if not all(m[q] for m in perm_maps))
    return {
        "n": len(common),
        "acc_original": acc_orig,
        "acc_permuted": acc_perm,
        "drop": acc_orig - acc_perm,
        "fragile_rate": fragile / len(correct_orig) if correct_orig else 0.0,
    }
