"""Step 4: train the cross-encoder out-of-fold and write cached score tables for an experiment.

  python scripts/train_cross_encoder.py --experiment val --base-model resources/models/medcpt-cross
  python scripts/train_cross_encoder.py --experiment code_disjoint --base-model ...
  python scripts/train_cross_encoder.py --experiment final_test --base-model ...
  python scripts/train_cross_encoder.py --experiment final --base-model ...   # model for inference

For 'val' / 'code_disjoint' / 'final_test': one cross-encoder per OOF fold (trained on the other
folds' queries) scores that fold's top-N pools -> training-table scores; one trained on the whole
training partition scores the evaluation pools. Tables: models/cache/ce_<experiment>_<hash>.pkl,
merged automatically by scripts/train.py when cross_encoder.enabled is true.
'final' trains on all pairs and saves models/cross_encoder/final for termmap.predict.
Resumable: every fold model and every score table is cached; reruns skip finished pieces.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from termmap.cross_encoder import (doc_text, query_text, top_n_pool,  # noqa: E402
                                   train_cross_encoder, CrossEncoderScorer)
from termmap.data import SYSTEMS, diagnosis_pairs, load_train, system_rows  # noqa: E402
from termmap.pipeline import (_cache_path, _cached, augment_rows, build_eval_features,  # noqa: E402
                              build_oof_features, load_resources)
from termmap.stages import base_qid  # noqa: E402
from termmap.util import ROOT, load_config, read_json  # noqa: E402


def experiment_ids(pairs, sp, name):
    test = set(sp["test_pair_ids"])
    if name == "val":
        return sp["train_pair_ids"], sp["val_pair_ids"], sp["oof_fold"], None
    if name == "code_disjoint":
        cd = sp["code_disjoint_codes"]
        dev = pairs[~pairs["pair_id"].isin(test)]
        is_cd = dev["gold_ICD10CM"].isin(set(cd["ICD10CM"])) | dev["gold_SNOMEDCT_US"].isin(set(cd["SNOMEDCT_US"]))
        return sorted(dev.loc[~is_cd, "pair_id"]), sorted(dev.loc[is_cd, "pair_id"]), sp["oof_fold_dev"], cd
    if name == "final_test":
        return sorted(set(sp["train_pair_ids"]) | set(sp["val_pair_ids"])), sp["test_pair_ids"], sp["oof_fold_dev"], None
    if name == "final":
        return sorted(pairs["pair_id"]), [], sp["oof_fold_all"], None
    raise ValueError(name)


def query_texts(R, pairs_sub):
    """qid -> query text, including the deterministic noise-augmented copies (~aug)."""
    out = {}
    for s in SYSTEMS:
        rows = system_rows(pairs_sub, s)
        rows = pd.concat([rows, augment_rows(rows, R.cfg.get("augment", {}).get("frac", 0.0), R.cfg["seed"])])
        for q, e, c in zip(rows["pair_id"], rows["entity"], rows["chunk_text"]):
            out[q] = query_text(e, c)
    return out


def examples_for(R, trf, qtext, qids, top_n, n_neg):
    ex = []
    for s in SYSTEMS:
        f = trf[s]
        f = f[f["qid"].isin(qids)]
        if not len(f):
            continue
        pool = top_n_pool(f, top_n)
        gold = f[f["label"] == 1][["qid", "code"]]
        for q, g in pool.groupby("qid", sort=True):
            pos = gold[gold["qid"] == q]["code"].tolist()
            negs = [c for c in g["code"] if c not in pos][:n_neg]
            for c in pos:
                ex.append((qtext[q], doc_text(R.kbs[s], s, c), 1.0))
            for c in negs:
                ex.append((qtext[q], doc_text(R.kbs[s], s, c), 0.0))
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--experiment", required=True, choices=["val", "code_disjoint", "final_test", "final"])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--neg", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg.setdefault("cross_encoder", {})["enabled"] = False  # never score with an old CE while training
    R = load_resources(cfg)
    pairs = diagnosis_pairs(load_train(cfg), R.kbs)
    sp = read_json(ROOT / "reports" / "splits.json")
    tr_ids, ev_ids, fold_of, cd = experiment_ids(pairs, sp, a.experiment)
    tr = pairs[pairs["pair_id"].isin(set(tr_ids))]
    trf = build_oof_features(R, tr, fold_of, cfg["rerank"]["code_dropout"])
    qtext = query_texts(R, tr)
    root = ROOT / "models" / "cross_encoder" / a.experiment
    hp = dict(top_n=a.top_n, neg=a.neg, epochs=a.epochs, lr=a.lr, bs=a.batch_size, base=a.base_model)
    folds = sorted({fold_of[p] for p in tr["pair_id"]})
    train_tables = {s: [] for s in SYSTEMS}
    for f in folds:
        qf = {q for s in SYSTEMS for q in trf[s]["qid"] if fold_of.get(base_qid(q)) == f}
        qo = {q for s in SYSTEMS for q in trf[s]["qid"] if fold_of.get(base_qid(q)) != f}
        path = _cache_path(R, f"ce_{a.experiment}_fold{f}", hp, sorted(tr_ids))

        def compute(qf=qf, qo=qo, f=f):
            mdir = train_cross_encoder(examples_for(R, trf, qtext, qo, a.top_n, a.neg), root / f"fold{f}",
                                       a.base_model, a.epochs, a.lr, a.batch_size)
            sc = CrossEncoderScorer(mdir, R.kbs, top_n=a.top_n)
            res = {}
            for s in SYSTEMS:
                sub = trf[s][trf[s]["qid"].isin(qf)]
                qd = pd.DataFrame({"qid": sorted(set(sub["qid"]))})
                pool = top_n_pool(sub, a.top_n)
                pairs_t = [(qtext[q], doc_text(R.kbs[s], s, c)) for q, c in zip(pool["qid"], pool["code"])]
                pool["ce_score"] = sc.score_pairs(pairs_t) if pairs_t else []
                res[s] = pool[["qid", "code", "ce_score"]]
                del qd
            return res

        res = _cached(path, compute)
        for s in SYSTEMS:
            train_tables[s].append(res[s])
        print(f"[ce] fold {f} scored", flush=True)
    if a.experiment == "final":
        all_q = set(trf["ICD10CM"]["qid"]) | set(trf["SNOMEDCT_US"]["qid"])
        p = train_cross_encoder(examples_for(R, trf, qtext, all_q, a.top_n, a.neg), root / "final",
                                a.base_model, a.epochs, a.lr, a.batch_size)
        out = {"train": {s: pd.concat(v, ignore_index=True) for s, v in train_tables.items()}, "eval": None, "hp": hp}
        with open(_cache_path(R, "ce_tables_final", sorted(tr_ids), []), "wb") as fh:
            pickle.dump(out, fh)
        print("final cross-encoder:", p, "-> set cross_encoder.model_path in configs/evaluation.yaml")
        return
    ev = pairs[pairs["pair_id"].isin(set(ev_ids))]
    evf, evrows, _ = build_eval_features(R, tr, ev, cd_codes=cd)
    epath = _cache_path(R, f"ce_{a.experiment}_eval", hp, sorted(tr_ids), sorted(ev_ids))

    def compute_eval():
        all_q = set(trf["ICD10CM"]["qid"]) | set(trf["SNOMEDCT_US"]["qid"])
        mdir = train_cross_encoder(examples_for(R, trf, qtext, all_q, a.top_n, a.neg), root / "all",
                                   a.base_model, a.epochs, a.lr, a.batch_size)
        sc = CrossEncoderScorer(mdir, R.kbs, top_n=a.top_n)
        res = {}
        for s in SYSTEMS:
            q = evrows[s].rename(columns={"pair_id": "qid"})[["qid", "entity", "chunk_text"]]
            res[s] = sc.score_feats(evf[s], q, s) if len(evf[s]) else pd.DataFrame(columns=["qid", "code", "ce_score"])
        return res

    eres = _cached(epath, compute_eval)
    out = {"train": {s: pd.concat(v, ignore_index=True) for s, v in train_tables.items()}, "eval": eres, "hp": hp}
    final_p = _cache_path(R, f"ce_tables_{a.experiment}", sorted(tr_ids), sorted(ev_ids))
    with open(final_p, "wb") as fh:
        pickle.dump(out, fh)
    print("wrote", final_p)


if __name__ == "__main__":
    main()
