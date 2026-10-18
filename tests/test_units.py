"""Fast unit tests that need no data files: schema, normalization, qualifiers, hierarchy,
context, decoding."""
from collections import Counter

import pandas as pd

from termmap.context import build_context
from termmap.input_schema import detect_schema, to_queries
from termmap.normalize import Vocab, basic_norm, make_variants
from termmap.qualifiers import Relevance, compare, extract
from termmap.specificity import safe_decode
from termmap.terminology import build_icd_kb, norm_icd, norm_snomed


def _toy_icd():
    rows = [("A00-B99", "infectious chapter"), ("E11", "Type 2 diabetes mellitus"),
            ("E11.4", "Type 2 diabetes mellitus with neurological complications"),
            ("E11.40", "Type 2 diabetes mellitus with diabetic neuropathy, unspecified"),
            ("E11.42", "Type 2 diabetes mellitus with diabetic polyneuropathy"),
            ("E11.9", "Type 2 diabetes mellitus without complications"),
            ("I10", "Essential (primary) hypertension"), ("N18.30", "Chronic kidney disease, stage 3 unspecified")]
    return build_icd_kb(pd.DataFrame(rows, columns=["Code", "Description"]))


# ---------------- test_schema
def test_schema_train_style():
    df = pd.DataFrame({"entity": ["htn", "htn"], "chunk_text": ["x", "x"],
                       "coding_system": ["ICD10CM", "SNOMEDCT_US"], "code": ["", ""],
                       "display": ["", ""], "subconcept": ["DiagnosisCode"] * 2})
    s = detect_schema(df)
    assert s.layout == "long" and s.entity_col == "entity" and s.code_col == "code"
    q = to_queries(df, s)
    assert list(q["qid"]) == [0, 1] and list(q["row_id"]) == ["row_0", "row_1"]


def test_schema_statement_style():
    df = pd.DataFrame({"chunk_id": ["c1", "c2"], "clinical_entity": ["HTN", "MS"]})
    s = detect_schema(df)
    assert s.layout == "wide" and s.id_col == "chunk_id" and s.context_col is None
    q = to_queries(df, s)
    assert list(q["row_id"]) == ["c1", "c2"] and (q["chunk_text"] == "").all()


# ---------------- test_normalization
def _vocab():
    c = Counter({"chronic": 50, "kidney": 40, "disease": 90, "stage": 30, "staghorn": 2, "edema": 10,
                 "bilateral": 12, "hypertension": 60, "multiple": 9, "sclerosis": 9, "mitral": 5,
                 "stenosis": 7})
    return Vocab(c, typo_min_len=5, typo_max_distance=2, prefix_min_len=3)


def test_glue_and_truncation_repair():
    vs = make_variants("chronickidney disease stag 3", _vocab(), {})
    assert vs[0].text == "chronickidney disease stag 3" and vs[0].kind == "original"
    assert any(v.text == "chronic kidney disease stage 3" for v in vs)


def test_typo_repair_and_original_preserved():
    vs = make_variants("hypertensoin", _vocab(), {})
    assert vs[0].text == "hypertensoin"
    assert any(v.text == "hypertension" for v in vs)


def test_abbreviation_hypotheses_not_destructive():
    abbr = {"ms": [("multiple sclerosis", 1.0), ("mitral stenosis", 0.8)]}
    vs = make_variants("MS", _vocab(), abbr)
    texts = [v.text for v in vs]
    assert texts[0] == "ms" and "multiple sclerosis" in texts and "mitral stenosis" in texts


def test_basic_norm():
    assert basic_norm("  Café—HTN,  well-controlled ") == "cafe htn well controlled"
    assert basic_norm(None) == "" and basic_norm("nan") == ""


# ---------------- test_qualifiers
def test_laterality_acuity_stage():
    q = extract("acute on chronic left knee pain stage 3")
    assert q.lat == "L" and q.acu == "AC" and q.stage == "3"
    assert extract("bilat le edema").lat == "B"
    assert extract("ckd stage iii").stage == "3"
    f = compare(extract("right knee pain"), extract("pain in left knee"))
    assert f["q_lat_contra"] == 1.0


def test_well_controlled_low_relevance():
    ents = ["htn well controlled", "dm well controlled", "asthma well controlled", "htn"]
    gold = [["essential", "hypertension"], ["diabetes", "mellitus"], ["asthma"], ["essential", "hypertension"]]
    r = Relevance().fit(ents, gold)
    assert r.weight("well") < 0.3 and r.weight("controlled") < 0.3
    assert r.weight("htn") > 0.6  # contraction-style abbreviation of "hypertension"


# ---------------- test_hierarchy
def test_icd_chain_and_flags():
    kb = _toy_icd()
    assert "A00-B99" not in kb.code_set and kb.range_headers
    assert kb.ancestors("E11.40") == ["E11.4", "E11"]
    assert kb.is_leaf["E11.40"] and not kb.is_leaf["E11.4"] and not kb.is_leaf["E11"]
    assert kb.children["E11.4"] == ["E11.40", "E11.42"]


def test_code_normalization_and_malformed():
    assert norm_icd("e1140") == "E11.40" and norm_icd(" I10 ") == "I10"
    assert norm_snomed("59621000.0") == "59621000"
    kb = _toy_icd()
    assert not kb.valid(norm_icd("L6.0"))          # malformed gold stays invalid (reported, not hacked)
    c, rule = safe_decode("E11.49", kb)              # not in toy dict -> nearest valid ancestor
    assert c == "E11.4" and rule == "fallback_ancestor"


# ---------------- test_empty_context
def test_empty_context_and_markup():
    c = build_context("htn", "")
    assert c.found == "empty" and c.window == ""
    c2 = build_context("edema", "header_1: :: header_2: no edema<br>noted:: header_3:")
    assert c2.found == "exact" and c2.negated and "header" not in c2.clean
    c3 = build_context("mi", "patient with h/o mi in 2019")
    assert c3.history
