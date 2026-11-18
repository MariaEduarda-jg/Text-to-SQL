from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from vestibular.eval.metrics import mcnemar_exact, permutation_consistency, summarize
from vestibular.eval.runner import RESULTS_DIR, load_predictions

OUT = Path("results")


def collect_runs(runs_dir: Path = RESULTS_DIR) -> list[dict]:
    runs = []
    for d in sorted(runs_dir.glob("*")):
        cfg_p, pred_p = d / "config.json", d / "predictions.jsonl"
        if not (cfg_p.exists() and pred_p.exists()):
            continue
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        recs = load_predictions(pred_p)
        runs.append({"name": d.name, "config": cfg, "records": recs, "summary": summarize(recs)})
    return runs


def leaderboard(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        c, s = r["config"], r["summary"]
        if c.get("permutation_seed") is not None:
            continue
        rows.append(
            {
                "benchmark": c["benchmark"],
                "model": c["model"] + (f" [{c['tag']}]" if c.get("tag") else ""),
                "mode": c["mode"],
                "n": s["n"],
                "acc": round(100 * s["acc"], 1),
                "ci95": [round(100 * x, 1) for x in s["ci95"]],
                "unparsed_%": round(100 * s["unparsed_rate"], 1),
                "avg_tokens": round(s["avg_completion_tokens"]),
                **{f"acc_{k}": round(100 * v["acc"], 1) for k, v in s["by_area"].items()},
            }
        )
    return sorted(rows, key=lambda x: (x["benchmark"], -x["acc"]))


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_sem resultados ainda_"
    areas = sorted({k for r in rows for k in r if k.startswith("acc_")})
    head = ["Benchmark", "Modelo", "Modo", "N", "Acurácia (IC95%)", *[a[4:] for a in areas], "Tokens/resp."]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [
            r["benchmark"], r["model"], r["mode"], str(r["n"]),
            f"**{r['acc']}** ({r['ci95'][0]}–{r['ci95'][1]})",
            *[str(r.get(a, "–")) for a in areas],
            str(r["avg_tokens"]),
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def contamination_table(runs: list[dict]) -> list[dict]:
    """Junta cada execução original com suas versões permutadas (mesmo benchmark/modelo/modo)."""
    groups: dict[tuple, dict] = defaultdict(lambda: {"orig": None, "perm": []})
    for r in runs:
        c = r["config"]
        key = (c["benchmark"], c["model"], c["mode"], c.get("tag", ""))
        if c.get("permutation_seed") is None:
            groups[key]["orig"] = r["records"]
        else:
            groups[key]["perm"].append(r["records"])
    out = []
    for (bench, model, mode, _tag), g in groups.items():
        if g["orig"] and g["perm"]:
            by_exam: dict[str, dict] = {}
            for exam in sorted({x["exam"] for x in g["orig"]}):
                o = [x for x in g["orig"] if x["exam"] == exam]
                ps = [[x for x in p if x["exam"] == exam] for p in g["perm"]]
                by_exam[exam] = permutation_consistency(o, ps)
            out.append(
                {
                    "benchmark": bench,
                    "model": model,
                    "mode": mode,
                    "overall": permutation_consistency(g["orig"], g["perm"]),
                    "by_exam": by_exam,
                }
            )
    return out


def pairwise(runs: list[dict], benchmark: str) -> list[dict]:
    base = [
        r
        for r in runs
        if r["config"]["benchmark"] == benchmark and r["config"].get("permutation_seed") is None
    ]
    out = []
    for i, a in enumerate(base):
        for b in base[i + 1 :]:
            out.append({"a": a["name"], "b": b["name"], **mcnemar_exact(a["records"], b["records"])})
    return out


def plots(runs: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in runs:
        c = r["config"]
        if c.get("permutation_seed") is not None or c["mode"] != "caption":
            continue
        by_exam = r["summary"]["by_exam"]
        exams = sorted(by_exam, key=lambda e: (e.split()[0], e.split()[-1]))
        ax.plot(
            exams,
            [100 * by_exam[e]["acc"] for e in exams],
            marker="o",
            label=f"{c['model']} · {c['benchmark']}",
        )
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Acurácia por prova")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "acc_por_prova.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for r in runs:
        c = r["config"]
        if c.get("permutation_seed") is not None:
            continue
        img = r["summary"]["by_images"].get("True")
        if img:
            by_model[f"{c['model']}\n{c['benchmark']}"][c["mode"]] = 100 * img["acc"]
    by_model = {k: v for k, v in by_model.items() if len(v) > 1}
    if by_model:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        modes = ["blind", "caption", "image"]
        w = 0.25
        for j, m in enumerate(modes):
            xs = [i + (j - 1) * w for i in range(len(by_model))]
            ax.bar(xs, [v.get(m, 0) for v in by_model.values()], width=w, label=m)
        ax.set_xticks(range(len(by_model)), list(by_model), fontsize=7)
        ax.set_ylabel("Acurácia nas questões com imagem (%)")
        ax.set_title("Imagem real vs. descrição textual vs. sem imagem")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "imagem_vs_descricao.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    for log in Path("results/training").glob("*/log_history.json"):
        hist = json.loads(log.read_text(encoding="utf-8"))
        steps = [h["step"] for h in hist if "reward" in h]
        rew = [h["reward"] for h in hist if "reward" in h]
        if steps:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(steps, rew, alpha=0.35)
            k = 10
            smooth = [
                sum(rew[max(0, i - k) : i + 1]) / len(rew[max(0, i - k) : i + 1]) for i in range(len(rew))
            ]
            ax.plot(steps, smooth, linewidth=2)
            ax.set_xlabel("passo")
            ax.set_ylabel("recompensa média")
            ax.set_title(f"GRPO — {log.parent.name}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = out_dir / f"grpo_reward_{log.parent.name}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            made.append(p)
    return made


def build_report(out: Path = OUT) -> dict:
    runs = collect_runs()
    lb = leaderboard(runs)
    contam = contamination_table(runs)
    benches = sorted({r["config"]["benchmark"] for r in runs})
    pw = {b: pairwise(runs, b) for b in benches}
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(json.dumps(lb, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "contamination.json").write_text(
        json.dumps(contam, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "pairwise.json").write_text(json.dumps(pw, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "LEADERBOARD.md").write_text(to_markdown(lb) + "\n", encoding="utf-8")
    figs = plots(runs, out / "figures")
    return {"runs": len(runs), "figures": [str(f) for f in figs]}
