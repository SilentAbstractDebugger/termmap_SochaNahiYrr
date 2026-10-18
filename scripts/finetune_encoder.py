"""Step 3: contrastive fine-tuning of a bi-encoder (sentence-transformers, MultipleNegativesRankingLoss
with explicit hard negatives).

  python scripts/finetune_encoder.py --base resources/models/sapbert --out models/encoders/sapbert-ft \
      --experiment code_disjoint            # leakage-safe: excludes CD codes, val rows and test rows

Training pairs (TRAINING partition of the chosen experiment only):
  * entity -> gold description; (entity ; context window) -> gold description
  * noisy copies of entities (same corruption family as the reranker augmentation)
  * dictionary synonym pairs for codes that occur in the training partition (desc_i -> desc_j)
Hard negatives: lexical nearest dictionary neighbours of the gold code's description (char 3-gram
channel) and, for ICD, siblings in the same 3-character block.
STRICT leakage control: rows of validation / FINAL TEST pairs, and every row or synonym pair whose
code is a code-disjoint code (either system), are excluded; the script asserts it.
Early stopping: recall@10 on a held-out 10% slice of the training partition (no CD codes).
Resumable: checkpoints every --save-steps; rerun continues from the last checkpoint.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from termmap.context import build_context  # noqa: E402
from termmap.data import SYSTEMS, diagnosis_pairs, load_train, system_rows  # noqa: E402
from termmap.normalize import basic_norm  # noqa: E402
from termmap.pipeline import _corrupt, load_resources  # noqa: E402
from termmap.terminology import icd_nodot  # noqa: E402
from termmap.util import ROOT, load_config, read_json, stable_rng  # noqa: E402


def build_pairs(R, pairs, train_ids, banned_codes, seed, n_hard=1, max_syn=2):
    rng = random.Random(seed)
    out = []
    tr = pairs[pairs["pair_id"].isin(set(train_ids))]
    for s in SYSTEMS:
        kb, lex = R.kbs[s], R.lex[s]
        rows = system_rows(tr, s)
        rows = rows[rows["gold_valid"] & ~rows["gold"].isin(banned_codes[s])]
        golds = sorted(set(rows["gold"]))
        hard = {}
        descs = [kb.descs_norm[c][0] for c in golds]
        for c, res in zip(golds, lex.search_char(descs, 10)):
            negs = [cc for di, _ in res for cc in lex.doc_codes[di] if cc != c and cc not in banned_codes[s]]
            if s == "ICD10CM":
                negs += [x for x in kb.children.get(kb.parent.get(c, ""), []) if x != c and x not in banned_codes[s]]
            hard[c] = list(dict.fromkeys(negs))
        for r in rows.itertuples():
            pos = kb.descs_norm[r.gold][0]
            negs = hard.get(r.gold, [])
            neg = kb.descs_norm[negs[0]][0] if negs else None
            e = basic_norm(r.entity)
            w = build_context(r.entity, r.chunk_text).window
            anchors = [e] + ([f"{e} ; {w}"] if w else [])
            noisy = basic_norm(_corrupt(r.entity, stable_rng(seed, "ft-noise", r.pair_id)))
            if noisy and noisy != e:
                anchors.append(noisy)
            for a in anchors:
                out.append((a, pos, neg, s, r.gold))
        for c in golds:  # synonym pairs of training codes only
            ds = kb.descs_norm[c]
            for i in range(1, min(len(ds), max_syn + 1)):
                negs = hard.get(c, [])
                out.append((ds[i], ds[0], kb.descs_norm[negs[0]][0] if negs else None, s, c))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiment", default="code_disjoint", choices=["val", "code_disjoint", "final"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-seq-length", type=int, default=64)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    cfg = load_config(a.config)
    cfg["dense"]["enabled"] = False
    cfg["dense"]["models"] = []
    R = load_resources(cfg)
    pairs = diagnosis_pairs(load_train(cfg), R.kbs)
    sp = read_json(ROOT / "reports" / "splits.json")
    test, val = set(sp["test_pair_ids"]), set(sp["val_pair_ids"])
    cd = {s: set(sp["code_disjoint_codes"][s]) for s in SYSTEMS}
    if a.experiment == "final":
        train_ids, banned = sorted(pairs["pair_id"]), {s: set() for s in SYSTEMS}
    else:
        dev = pairs[~pairs["pair_id"].isin(test | (val if a.experiment == "val" else set()))]
        if a.experiment == "code_disjoint":
            dev = dev[~(dev["gold_ICD10CM"].isin(cd["ICD10CM"]) | dev["gold_SNOMEDCT_US"].isin(cd["SNOMEDCT_US"]))]
        train_ids = sorted(set(dev["pair_id"]) - val - test)
        banned = cd if a.experiment == "code_disjoint" else {s: set() for s in SYSTEMS}
    data = build_pairs(R, pairs, train_ids, banned, a.seed)
    # leakage assertions
    used = set(train_ids)
    assert not (used & test), "test rows in fine-tuning data"
    assert not (used & val) or a.experiment == "final", "validation rows in fine-tuning data"
    assert all(c not in banned[s] for _, _, _, s, c in data), "code-disjoint code in fine-tuning data"
    # held-out 10% of training codes for early stopping (IR evaluator over their descriptions)
    codes = sorted({(s, c) for *_, s, c in data})
    rng = random.Random(a.seed)
    dev_codes = set(rng.sample(codes, max(1, len(codes) // 10)))
    tr_rows = [d for d in data if (d[3], d[4]) not in dev_codes]
    dv_rows = [d for d in data if (d[3], d[4]) in dev_codes]
    corpus = {f"{s}|{c}": R.kbs[s].descs_norm[c][0] for s, c in dev_codes}
    distract = rng.sample([(s, c) for s in SYSTEMS for c in R.kbs[s].codes[:: 50]], 5000)
    corpus.update({f"{s}|{c}": R.kbs[s].descs_norm[c][0] for s, c in distract})
    queries = {str(i): d[0] for i, d in enumerate(dv_rows)}
    rel = {str(i): {f"{d[3]}|{d[4]}"} for i, d in enumerate(dv_rows)}
    ev = InformationRetrievalEvaluator(queries, corpus, rel, name="ft-dev", accuracy_at_k=[1, 10],
                                       precision_recall_at_k=[10], show_progress_bar=False)
    model = SentenceTransformer(a.base)
    model.max_seq_length = a.max_seq_length
    trip = [d for d in tr_rows if d[2] is not None]
    ds = Dataset.from_dict({"anchor": [d[0] for d in trip], "positive": [d[1] for d in trip],
                            "negative": [d[2] for d in trip]})
    out = Path(a.out)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr, warmup_ratio=0.1, fp16=torch.cuda.is_available(), seed=a.seed, data_seed=a.seed,
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=2, eval_strategy="steps",
        eval_steps=a.save_steps, load_best_model_at_end=True, metric_for_best_model="ft-dev_cosine_recall@10",
        logging_steps=50, report_to=[], disable_tqdm=True, batch_sampler="no_duplicates")
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                         loss=losses.MultipleNegativesRankingLoss(model), evaluator=ev)
    ck = sorted((out / "checkpoints").glob("checkpoint-*")) if (out / "checkpoints").exists() else []
    base_score = ev(model)
    trainer.train(resume_from_checkpoint=str(ck[-1]) if ck else None)
    final_score = ev(model)
    model.save(str(out))
    (out / "finetune_report.json").write_text(json.dumps({
        "base": a.base, "experiment": a.experiment, "n_train_triplets": len(trip), "n_dev_queries": len(dv_rows),
        "excluded_code_disjoint_codes": {s: len(v) for s, v in banned.items()},
        "dev_before": base_score, "dev_after": final_score, "seed": a.seed}, indent=2, default=str))
    print("saved", out, "dev recall@10 before/after:", base_score.get("ft-dev_cosine_recall@10"),
          final_score.get("ft-dev_cosine_recall@10"))


if __name__ == "__main__":
    main()
