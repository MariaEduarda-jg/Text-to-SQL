from __future__ import annotations

import json
import re

import polars as pl
from huggingface_hub import hf_hub_download

from vestibular.schema import IMAGE_TOKEN, Question

ENEM_REPO = "maritaca-ai/enem"
BLUEX_REPO = "portuguese-benchmark-datasets/BLUEX"
OAB_REPO = "eduagarcia/oab_exams"

ENEM_YEARS = (2022, 2023, 2024)

ENEM_AREAS = ("linguagens", "humanas", "natureza", "matematica")

SUBJECT_TO_AREA = {
    "mathematics": "matematica",
    "physics": "natureza",
    "chemistry": "natureza",
    "biology": "natureza",
    "history": "humanas",
    "geography": "humanas",
    "philosophy": "humanas",
    "sociology": "humanas",
    "portuguese": "linguagens",
    "english": "linguagens",
}

_ALT_PREFIX = re.compile(r"^\s*\(?[a-eA-E]\)\s*")
_BLUEX_IMG = re.compile(r"\[IMAGE (\d+)\]")


def _download(repo: str, filename: str) -> str:
    return hf_hub_download(repo, filename, repo_type="dataset")


def enem_area(number: int) -> str:
    return ENEM_AREAS[min((number - 1) // 45, 3)]


def _fill_enem_placeholders(statement: str, alternatives: list[str]) -> tuple[str, list[str]]:
    """Troca cada "[[placeholder]]" por "[IMAGEM i]"."""
    counter = 0

    def repl(_: re.Match) -> str:
        nonlocal counter
        token = IMAGE_TOKEN.format(i=counter)
        counter += 1
        return token

    pattern = re.compile(r"\[\[placeholder\]\]")
    statement = pattern.sub(repl, statement)
    alternatives = [pattern.sub(repl, a) for a in alternatives]
    return statement, alternatives


def enem_row_to_question(row: dict, year: int) -> Question | None:
    if row["label"] not in "ABCDE" or len(row["label"]) != 1:
        return None
    number = int(row["id"].split("_")[-1])
    statement, alternatives = _fill_enem_placeholders(row["question"], list(row["alternatives"]))
    return Question(
        id=f"enem-{year}-q{number:03d}",
        source="enem",
        exam=f"ENEM {year}",
        year=year,
        area=enem_area(number),
        statement=statement.strip(),
        alternatives=[a.strip() for a in alternatives],
        answer=row["label"],
        images=list(row.get("figures") or []),
        captions=[d.strip() for d in (row.get("description") or [])],
    )


def load_enem(year: int) -> list[Question]:
    path = _download(ENEM_REPO, f"{year}.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = enem_row_to_question(json.loads(line), year)
            if q:
                out.append(q)
    return out


def bluex_row_to_question(row: dict) -> Question | None:
    answer = (row.get("answer") or "").strip().upper()
    alts = list(row.get("alternatives") or [])
    if answer not in "ABCDE" or len(answer) != 1 or len(alts) not in (4, 5):
        return None
    if "ABCDE".index(answer) >= len(alts):
        return None

    parts = row["id"].split("_")
    university, year = parts[0], int(parts[1])
    subjects = list(row.get("subject") or [])
    area = next((SUBJECT_TO_AREA[s] for s in subjects if s in SUBJECT_TO_AREA), "outros")

    def fix(text: str) -> str:
        return _BLUEX_IMG.sub(lambda m: IMAGE_TOKEN.format(i=m.group(1)), text)

    images = [
        img if img.startswith(("http", "data:")) else f"data:image/png;base64,{img}"
        for img in (row.get("associated_images") or [])
    ]
    captions = list(row.get("context_captions") or row.get("blind_captions") or [])

    return Question(
        id=f"bluex-{row['id'].lower()}",
        source="bluex",
        exam=f"{university} {year}",
        year=year,
        area=area,
        statement=fix(row["question"]).strip(),
        alternatives=[fix(_ALT_PREFIX.sub("", a)).strip() for a in alts],
        answer=answer,
        images=images,
        captions=[c.strip() for c in captions],
        subjects=subjects,
    )


def load_bluex() -> list[Question]:
    df = pl.read_parquet(_download(BLUEX_REPO, "data/questions-00000-of-00001.parquet"))
    out = [q for row in df.iter_rows(named=True) if (q := bluex_row_to_question(row))]
    return sorted(out, key=lambda q: q.id)


def oab_row_to_question(row: dict) -> Question | None:
    if row.get("nullified"):
        return None
    choices = row["choices"]
    answer = (row.get("answerKey") or "").strip().upper()
    if answer not in choices["label"]:
        return None
    qtype = (row.get("question_type") or "").lower().replace("philoshopy", "philosophy")
    return Question(
        id=f"oab-{row['id']}",
        source="oab",
        exam=f"OAB {row['exam_id']}",
        year=int(row["exam_year"]),
        area="direito",
        statement=row["question"].strip(),
        alternatives=[t.strip() for t in choices["text"]],
        answer=answer,
        subjects=[qtype] if qtype else [],
    )


def load_oab() -> list[Question]:
    df = pl.read_parquet(_download(OAB_REPO, "data/train-00000-of-00001.parquet"))
    return [q for row in df.iter_rows(named=True) if (q := oab_row_to_question(row))]
