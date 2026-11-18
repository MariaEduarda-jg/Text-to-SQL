from __future__ import annotations

from pathlib import Path

from vestibular.data.loaders import ENEM_YEARS, load_bluex, load_enem, load_oab
from vestibular.schema import Question, load_jsonl, save_jsonl

DEFAULT_DIR = Path("data/processed")

TRAIN_BLUEX_MAX_YEAR = 2022
DEV_BLUEX_YEAR = 2023


def build_all(out_dir: str | Path = DEFAULT_DIR) -> dict[str, int]:
    out_dir = Path(out_dir)
    enem = {y: load_enem(y) for y in ENEM_YEARS}
    bluex = load_bluex()
    oab = load_oab()

    files: dict[str, list[Question]] = {f"enem_{y}": qs for y, qs in enem.items()}
    files["enem"] = [q for y in ENEM_YEARS for q in enem[y]]
    files["bluex"] = bluex
    files["bluex_test"] = [q for q in bluex if q.year > DEV_BLUEX_YEAR]
    files["oab"] = oab
    files["train"] = oab + [q for q in bluex if q.year <= TRAIN_BLUEX_MAX_YEAR]
    files["dev"] = [q for q in bluex if q.year == DEV_BLUEX_YEAR]
    files["test"] = files["enem"] + files["bluex_test"]

    _assert_no_leak(files["train"], files["test"] + files["dev"])
    return {name: save_jsonl(qs, out_dir / f"{name}.jsonl") for name, qs in files.items()}


def _assert_no_leak(train: list[Question], held_out: list[Question]) -> None:
    train_ids = {q.id for q in train}
    leaked = [q.id for q in held_out if q.id in train_ids]
    assert not leaked, f"Vazamento treino/teste: {leaked[:5]}"
    norm = lambda s: " ".join(s.lower().split())[:300]  # noqa: E731
    train_texts = {norm(q.statement) for q in train}
    dup = [q.id for q in held_out if norm(q.statement) in train_texts and len(q.statement) > 80]
    if dup:
        print(f"[aviso] {len(dup)} questões de teste com enunciado idêntico a uma de treino: {dup[:5]}")


def load_split(name: str, data_dir: str | Path = DEFAULT_DIR) -> list[Question]:
    path = Path(data_dir) / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode primeiro: vbr data")
    return load_jsonl(path)
