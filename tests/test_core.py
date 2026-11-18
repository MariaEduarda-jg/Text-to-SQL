from vestibular.data.loaders import (
    bluex_row_to_question,
    enem_area,
    enem_row_to_question,
    oab_row_to_question,
)
from vestibular.eval.metrics import bootstrap_ci, mcnemar_exact, permutation_consistency
from vestibular.eval.runner import _majority
from vestibular.parsing import extract_answer, follows_format
from vestibular.prompts import build_messages, permute_alternatives, render_question
from vestibular.schema import Question
from vestibular.train.rewards import correctness_reward, format_reward, make_length_penalty


def make_q(**kw) -> Question:
    base = dict(
        id="t-1", source="enem", exam="ENEM 2024", year=2024, area="matematica",
        statement="Veja [IMAGEM 0]. Quanto é 2+2?", alternatives=["3", "4", "5", "6", "7"], answer="B",
        images=["https://x/img.png"], captions=["um quadro com 2+2"],
    )  # fmt: skip
    base.update(kw)
    return Question(**base)


def test_extract_answer_formats():
    assert extract_answer("blá blá\nResposta: C", "ABCDE") == "C"
    assert extract_answer("Resposta: A ... na verdade, Resposta: D") == "D"
    assert extract_answer("A alternativa correta é a E.") == "E"
    assert extract_answer("\\boxed{B}") == "B"
    assert extract_answer("**Resposta final:** (A)") == "A"
    assert extract_answer("B") == "B"
    assert extract_answer("não sei") is None
    assert extract_answer("Resposta: E", valid="ABCD") is None
    assert extract_answer("<think>Resposta: A</think> Resposta: C") == "C"


def test_follows_format():
    assert follows_format("raciocínio...\nResposta: B")
    assert not follows_format("Resposta: B, porque...")


def test_enem_area_mapping():
    assert enem_area(1) == "linguagens"
    assert enem_area(45) == "linguagens"
    assert enem_area(46) == "humanas"
    assert enem_area(91) == "natureza"
    assert enem_area(180) == "matematica"


def test_enem_row_placeholders_and_annulled():
    row = {
        "id": "questao_137", "question": "Fig [[placeholder]] texto", "alternatives": ["[[placeholder]]", "b", "c", "d", "e"],
        "label": "C", "figures": ["u0", "u1"], "description": ["d0", "d1"],
    }  # fmt: skip
    q = enem_row_to_question(row, 2023)
    assert q.statement == "Fig [IMAGEM 0] texto"
    assert q.alternatives[0] == "[IMAGEM 1]"
    assert q.area == "matematica" and q.id == "enem-2023-q137"
    assert enem_row_to_question({**row, "label": "Anulado"}, 2023) is None


def test_bluex_row():
    row = {
        "id": "USP_2024_7", "question": "Veja [IMAGE 0]", "alternatives": ["a) um", "b) dois", "c) três", "d) quatro", "e) cinco"],
        "answer": "c", "subject": ["physics"], "associated_images": ["iVBORw0"], "context_captions": ["cap"],
    }  # fmt: skip
    q = bluex_row_to_question(row)
    assert q.answer == "C" and q.year == 2024 and q.area == "natureza"
    assert q.alternatives[0] == "um"
    assert q.statement == "Veja [IMAGEM 0]"
    assert q.images[0].startswith("data:image/png;base64,")
    assert bluex_row_to_question({**row, "answer": None}) is None


def test_oab_row():
    row = {
        "id": "2010-01_1", "exam_id": "2010-01", "exam_year": "2010", "question_type": "ETHICS", "nullified": False,
        "question": "Q?", "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]}, "answerKey": "D",
    }  # fmt: skip
    q = oab_row_to_question(row)
    assert q.letters == "ABCD" and q.answer == "D" and q.area == "direito"
    assert oab_row_to_question({**row, "nullified": True}) is None


def test_render_modes():
    q = make_q()
    assert "um quadro com 2+2" in render_question(q, "caption")
    assert "[imagem omitida]" in render_question(q, "blind")
    assert "[IMAGEM 0]" in render_question(q, "image")
    assert "B) 4" in render_question(q, "caption")


def test_build_messages_image_parts():
    msgs = build_messages(make_q(), mode="image")
    parts = msgs[1]["content"]
    assert any(p["type"] == "image_url" for p in parts)
    assert isinstance(build_messages(make_q(), mode="caption")[1]["content"], str)


def test_permutation_keeps_correct_content():
    q = make_q()
    for seed in range(5):
        p = permute_alternatives(q, seed)
        assert p.alternatives[ord(p.answer) - 65] == "4"
        assert sorted(p.alternatives) == sorted(q.alternatives)
        assert p.alternatives != q.alternatives


def test_majority_vote():
    assert _majority(["A", "B", "B"]) == "B"
    assert _majority([None, None]) is None
    assert _majority(["C", None]) == "C"


def test_bootstrap_ci_bounds():
    recs = [{"correct": i % 2 == 0} for i in range(100)]
    lo, hi = bootstrap_ci(recs)
    assert 0.35 < lo < 0.5 < hi < 0.65


def test_mcnemar():
    a = [{"qid": str(i), "correct": i < 50} for i in range(100)]
    b = [{"qid": str(i), "correct": i < 80} for i in range(100)]
    r = mcnemar_exact(a, b)
    assert r["b_only"] == 30 and r["a_only"] == 0 and r["p_value"] < 1e-6
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_permutation_consistency():
    orig = [{"qid": "q1", "correct": True}, {"qid": "q2", "correct": True}]
    perm = [[{"qid": "q1#perm1", "correct": True}, {"qid": "q2#perm1", "correct": False}]]
    r = permutation_consistency(orig, perm)
    assert r["drop"] == 0.5 and r["fragile_rate"] == 0.5


def test_rewards():
    comps = ["...\nResposta: B", "...\nResposta: A", "sem resposta"]
    assert correctness_reward(comps, answer=["B", "B", "B"]) == [1.0, 0.0, 0.0]
    assert format_reward(comps) == [0.1, 0.1, 0.0]
    conv = [[{"role": "assistant", "content": "Resposta: C"}]]
    assert correctness_reward(conv, answer=["C"]) == [1.0]
    pen = make_length_penalty(max_chars=10, weight=0.1)
    assert pen(["curto", "x" * 30]) == [0.0, -0.1]
