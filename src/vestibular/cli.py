from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False, help="Vestibular-BR: avaliação e treino de LLMs em provas brasileiras."
)


def _load_env() -> None:
    """Lê .env simples (CHAVE=valor) sem dependência extra."""
    import os

    p = Path(".env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _backend_kwargs(model: str, concurrency: int, lora: str | None, rpm: int | None = None) -> dict:
    if model.startswith("vllm:"):
        return {"lora": lora}
    if model.startswith("mock:"):
        return {}
    return {"concurrency": concurrency, "rpm": rpm}


@app.callback()
def _main() -> None:
    _load_env()


@app.command()
def data(out_dir: Path = Path("data/processed")):
    """Baixa ENEM, BLUEX e OAB do Hugging Face e gera os splits."""
    from vestibular.data.splits import build_all

    counts = build_all(out_dir)
    for name, n in counts.items():
        typer.echo(f"  {name:12s} {n:5d} questões")


@app.command("eval")
def eval_cmd(
    benchmark: str = typer.Option(
        ..., "-b", help="enem_2022|enem_2023|enem_2024|enem|bluex_test|dev|test|..."
    ),
    model: str = typer.Option(..., "-m", help="provedor:modelo, ex. gemini:gemini-3.5-flash"),
    mode: str = typer.Option("caption", help="caption|image|blind"),
    cot: bool = typer.Option(True, help="raciocínio passo a passo (--no-cot = só a letra)"),
    n: int = typer.Option(1, help="amostras por questão (>1 = self-consistency)"),
    temperature: float = 0.0,
    max_tokens: int = 2048,
    limit: int = typer.Option(None, help="avaliar só as N primeiras (teste rápido)"),
    perm_seed: int = typer.Option(None, help="embaralha as alternativas com essa semente"),
    lora: str = typer.Option(None, help="(vllm) caminho/repo do adaptador LoRA"),
    tag: str = typer.Option("", help="rótulo extra no nome da execução, ex. sft, grpo"),
    concurrency: int = 8,
    rpm: int = typer.Option(None, help="limite de requisições por minuto"),
):
    """Avalia um modelo em um benchmark."""
    from vestibular.backends import get_backend
    from vestibular.data.splits import load_split
    from vestibular.eval.metrics import summarize
    from vestibular.eval.runner import RunConfig, count_errors, load_predictions, run_eval

    backend = get_backend(model, **_backend_kwargs(model, concurrency, lora, rpm))
    cfg = RunConfig(benchmark, model, mode, cot, n, temperature, max_tokens, perm_seed, limit, tag)
    path = run_eval(load_split(benchmark), backend, cfg)
    s = summarize(load_predictions(path))
    typer.echo(
        f"\nAcurácia: {100 * s['acc']:.1f}%  (IC95% {100 * s['ci95'][0]:.1f}–{100 * s['ci95'][1]:.1f})"
        f"  n={s['n']}  sem resposta: {100 * s['unparsed_rate']:.1f}%"
    )
    for area, v in s["by_area"].items():
        typer.echo(f"  {area:12s} {100 * v['acc']:5.1f}%  (n={v['n']})")
    if errors := count_errors(path):
        typer.echo(f"\n{errors} questões com erro de API; rode o mesmo comando de novo para completar.")


@app.command()
def contamination(
    benchmark: str = typer.Option("test", "-b"),
    model: str = typer.Option(..., "-m"),
    seeds: list[int] = typer.Option([1, 2], help="sementes de permutação"),
    concurrency: int = 8,
    rpm: int = typer.Option(None),
):
    """Roda o benchmark original + versões com alternativas embaralhadas."""
    from vestibular.backends import get_backend
    from vestibular.data.splits import load_split
    from vestibular.eval.runner import RunConfig, run_eval

    backend = get_backend(model, **_backend_kwargs(model, concurrency, None, rpm))
    qs = load_split(benchmark)
    for seed in [None, *seeds]:
        run_eval(qs, backend, RunConfig(benchmark, model, permutation_seed=seed))
    typer.echo("Pronto. Rode `vbr report` para ver a tabela de contaminação.")


@app.command()
def distill(
    model: str = typer.Option(..., "-m", help="modelo professor"),
    k: int = typer.Option(4, help="amostras por questão"),
    split: str = "train",
    out: Path = Path("data/sft/train_sft.jsonl"),
    limit: int = typer.Option(None),
    max_tokens: int = 2048,
    batch_size: int = typer.Option(16, help="use valores altos (ex. 512) com vllm"),
    concurrency: int = 8,
    rpm: int = typer.Option(None),
):
    """Gera o dataset de SFT por rejection sampling com um modelo professor."""
    from vestibular.backends import get_backend
    from vestibular.data.splits import load_split
    from vestibular.train.distill import build_sft_dataset

    qs = load_split(split)
    if limit:
        qs = qs[:limit]
    backend = get_backend(model, **_backend_kwargs(model, concurrency, None, rpm))
    build_sft_dataset(qs, backend, out, k=k, max_tokens=max_tokens, batch_size=batch_size)


@app.command()
def publish(
    repo: str = typer.Option(..., help="ex.: seu-usuario/vestibular-br-data"),
    sft: Path = Path("data/sft/train_sft.jsonl"),
):
    """Publica os splits e o dataset de SFT como um dataset no Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    files = {f"splits/{p.name}": p for p in Path("data/processed").glob("*.jsonl")}
    if sft.exists():
        files["sft/train_sft.jsonl"] = sft
    for dest, src in files.items():
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dest, repo_id=repo, repo_type="dataset")
        typer.echo(f"  enviado {dest}")
    card = Path("docs/DATASET_CARD.md")
    if card.exists():
        api.upload_file(
            path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, repo_type="dataset"
        )
    typer.echo(f"https://huggingface.co/datasets/{repo}")


@app.command()
def report():
    """Gera results/LEADERBOARD.md, JSONs de análise e gráficos."""
    from vestibular.report import build_report

    typer.echo(json.dumps(build_report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
