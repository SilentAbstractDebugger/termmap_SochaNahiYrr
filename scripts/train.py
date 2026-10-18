"""Train / evaluate / ablate / refit (iteration 2 protocol).

  python scripts/train.py                      # val + code-disjoint, all variants, seeds, CIs; final refit
  python scripts/train.py --final-test         # ALSO evaluate the frozen config once on the FINAL TEST split
  python scripts/train.py --tag baseline --variants stage1 --no-final   # quick re-measure

Protocol
  * model selection on the group-validation split; code-disjoint = long-tail development metric;
  * FINAL TEST split (reports/splits.json:test_pair_ids) is excluded from every training resource and
    from both development experiments; it is scored once with the frozen configuration;
  * reranker seeds from rerank.seeds -> mean +- std; bootstrap 95% CIs (1000 resamples) on the
    primary seed; differences < 2 points are treated as noise unless CIs separate;
  * final model: refit on train+val+test with the frozen configuration (disclosed).
All long stages (feature tables) are cached on disk with hash keys and resume after interruption.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from termmap.data import SYSTEMS, diagnosis_pairs, load_train  # noqa: E402
from termmap.evaluation import bootstrap_ci, bootstrap_headline, headline, system_metrics  # noqa: E402
from termmap.pipeline import (MODEL_VERSION, _tres_pair, build_eval_features, build_oof_features,  # noqa: E402
                              load_resources, save_model)
from termmap.rerank import heuristic_scores, select_top  # noqa: E402
from termmap.stages import TwoStage  # noqa: E402
from termmap.util import ROOT, load_config, read_json, text_sha256, write_json  # noqa: E402

VARIANTS = {
    # name: (drop_groups, stage2, calibrate, use_aug_rows)
    "stage1": ((), False, False, True),
    "stage1_no_aug_rows": ((), False, False, False),
    "stage1_no_rrf": (("rrf",), False, False, True),
    "stage1_no_priors": (("priors",), False, False, True),
    "stage1_no_context": (("context",), False, False, True),
    "stage1_no_training_neighbours": (("training",), False, False, True),
    "stage1_no_dense": (("dense",), False, False, True),
    "stage1_no_hierarchy": (("hierarchy",), False, False, True),
    "stage1_no_qualifier": (("qualifier",), False, False, True),
    "stage1_no_cross_encoder": (("cross_encoder",), False, False, True),
    "stage2": ((), True, False, True),
    "stage2_calibrated": ((), True, True, True),
    "stage2_icd_only_calibrated": ((), ("ICD10CM",), True, True),
}
HEURISTICS = ("exact_fuzzy", "bm25_only", "char_only", "dense_only", "hybrid_sum", "rrf_only")


def _strip_aug(trf):
    return {s: (d[~d["qid"].astype(str).str.contains("~aug", regex=False)] if len(d) else d) for s, d in trf.items()}


def run_variant(R, name, trf, evf, evrows, tres, fold_of, seeds):
    drop, st2, cal, use_aug = VARIANTS[name]
    from termmap.rerank import FEATURE_GROUPS

    cols = set(trf["ICD10CM"].columns) | set(trf["SNOMEDCT_US"].columns)
    if drop and not any(c.startswith(p) for g in drop for p in FEATURE_GROUPS[g] for c in cols):
        return None  # feature group absent in this run: identical to 'stage1'
    t = trf if use_aug else _strip_aug(trf)
    if not use_aug and all(len(trf[s]) == len(t[s]) for s in SYSTEMS):
        return None  # no augmentation rows in the tables: identical to 'stage1'
    per_seed, primary = [], None
    for sd in seeds:
        ts = TwoStage(R.cfg, R.kbs, R.lex, sd, st2, cal, drop).fit(t, fold_of, R.cfg["splits"]["oof_folds"])
        tops = ts.predict(evf)
        accs = {}
        for s in SYSTEMS:
            m, d = system_metrics(s, R.kbs[s], evrows[s], tops[s], evf[s], tres[s].code_freq, tres[s].most_frequent)
            accs[s] = (m["overall"]["acc"], m["unseen_code"]["acc"])
            if primary is None or sd == seeds[0]:
                pass
        per_seed.append(accs)
        if sd == seeds[0]:
            primary = (ts, tops)
    res = {}
    for s in SYSTEMS:
        ov = [a[s][0] for a in per_seed]
        un = [a[s][1] for a in per_seed if a[s][1] is not None]
        res[s] = {"overall_mean": round(float(np.mean(ov)), 4), "overall_std": round(float(np.std(ov)), 4),
                  "overall_per_seed": ov, "unseen_mean": round(float(np.mean(un)), 4) if un else None}
    res["headline_mean"] = round(min(res["ICD10CM"]["overall_mean"], res["SNOMEDCT_US"]["overall_mean"]), 4)
    return res, primary


def full_metrics(R, tops, evf, evrows, tres, label):
    ms, tables = {}, {}
    for s in SYSTEMS:
        m, d = system_metrics(s, R.kbs[s], evrows[s], tops[s], evf[s], tres[s].code_freq, tres[s].most_frequent,
                              R.vocab)
        if "prob_raw" in tops[s]:
            from termmap.evaluation import ece, predictions_table

            raw = tops[s].rename(columns={"prob": "_p", "prob_raw": "prob"})
            draw = predictions_table(evrows[s], raw, tres[s].most_frequent)
            m["ece_uncalibrated"] = ece(draw["prob"].values, draw["correct"].values)
        m["ci95"] = bootstrap_ci(d["correct"].values)
        ms[s], tables[s] = m, d
    ms["headline"] = headline(ms)
    ms["headline"]["ci95"] = bootstrap_headline(tables["ICD10CM"]["correct"], tables["SNOMEDCT_US"]["correct"])
    print(f"[{label}] ICD={ms['ICD10CM']['ci95']} SNOMED={ms['SNOMEDCT_US']['ci95']} "
          f"headline={ms['headline']['ci95']} pool ICD={ms['ICD10CM']['pool_recall']} SNO={ms['SNOMEDCT_US']['pool_recall']}",
          flush=True)
    return ms, tables


def merge_ce_tables(R, label, train_ids, eval_ids, trf, evf):
    """Merge out-of-fold cross-encoder score tables (scripts/train_cross_encoder.py) if enabled."""
    if not R.cfg.get("cross_encoder", {}).get("enabled"):
        return trf, evf
    from termmap.cross_encoder import merge_scores
    from termmap.pipeline import _cache_path

    p = _cache_path(R, f"ce_tables_{label}", sorted(train_ids), sorted(eval_ids))
    if not p.exists():
        raise FileNotFoundError(f"cross-encoder tables missing for {label}: run scripts/train_cross_encoder.py "
                                f"--experiment {label} first ({p})")
    with open(p, "rb") as f:
        t = pickle.load(f)
    trf = {s: merge_scores(trf[s], t["train"][s]) for s in SYSTEMS}
    if t.get("eval") is not None:
        evf = {s: merge_scores(evf[s], t["eval"][s]) for s in SYSTEMS}
    return trf, evf


def experiment(R, pairs, train_ids, eval_ids, fold_of, variants, seeds, label, cd_codes=None, keep_rows=False):
    t0 = time.time()
    tr = pairs[pairs["pair_id"].isin(set(train_ids))].reset_index(drop=True)
    ev = pairs[pairs["pair_id"].isin(set(eval_ids))].reset_index(drop=True)
    # eval tables first (peak memory of retrieval), then the (cached) OOF training tables
    evf, evrows, tres = build_eval_features(R, tr, ev, cd_codes=cd_codes)
    trf = build_oof_features(R, tr, fold_of, R.cfg["rerank"]["code_dropout"])
    trf, evf = merge_ce_tables(R, label, train_ids, eval_ids, trf, evf)
    print(f"[{label}] features ready in {time.time() - t0:.0f}s", flush=True)
    out = {"variants": {}, "heuristics": {}}
    for h in HEURISTICS:
        acc = {}
        for s in SYSTEMS:
            f = evf[s]
            if h == "rrf_only" and "rrf" not in f:
                continue
            top = select_top(f, heuristic_scores(f, h))
            m, _ = system_metrics(s, R.kbs[s], evrows[s], top, f, tres[s].code_freq, tres[s].most_frequent)
            acc[s] = {"overall": m["overall"]["acc"], "unseen": m["unseen_code"]["acc"]}
        if acc:
            out["heuristics"][h] = acc
    main_name = variants[0]
    main_primary = None
    for name in variants:
        t1 = time.time()
        r = run_variant(R, name, trf, evf, evrows, tres, fold_of, seeds)
        if r is None:
            continue
        res, primary = r
        out["variants"][name] = res
        if name == main_name:
            main_primary = primary
        print(f"  [{label}] {name}: ICD {res['ICD10CM']['overall_mean']}+-{res['ICD10CM']['overall_std']} "
              f"SNOMED {res['SNOMEDCT_US']['overall_mean']}+-{res['SNOMEDCT_US']['overall_std']} "
              f"({time.time() - t1:.0f}s)", flush=True)
    ts, tops = main_primary
    ms, tables = full_metrics(R, tops, evf, evrows, tres, label)
    out["main_variant"] = main_name
    out["seeds"] = list(seeds)
    out["metrics"] = ms
    priv = ROOT / "reports" / "private"
    priv.mkdir(parents=True, exist_ok=True)
    with open(priv / f"{label}_run.pkl", "wb") as f:
        pickle.dump({s: {"rows": tables[s], "feats": evf[s], "top": tops[s]} for s in SYSTEMS}, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--variants", default=None, help="comma list; first = main/frozen variant")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--no-cd", action="store_true")
    ap.add_argument("--tag", default="metrics")
    ap.add_argument("--seeds", default=None, help="comma list overriding rerank.seeds (e.g. 13,14)")
    ap.add_argument("--splits", default="val,code_disjoint",
                    help="which experiments to run in THIS process (results merged into reports/<tag>.json); "
                         "run them in separate processes on small-RAM machines")
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.time()
    R = load_resources(cfg)
    pairs = diagnosis_pairs(load_train(cfg), R.kbs)
    sp = read_json(ROOT / "reports" / "splits.json")
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else cfg["rerank"].get("seeds", [cfg["seed"]])
    rr = cfg["rerank"]
    main_v = rr.get("main_variant") or (("stage2" if rr.get("stage2") else "stage1")
                                        + ("_calibrated" if rr.get("calibrate") and rr.get("stage2") else ""))
    variants = a.variants.split(",") if a.variants else [main_v] + [v for v in VARIANTS if v != main_v]
    rp = ROOT / "reports" / f"{a.tag}.json"
    out = read_json(rp) if rp.exists() else {}
    out.update({"mode": R.mode, "seeds": seeds, "protocol": __doc__.strip().splitlines()[-9:],
                "config_hash": text_sha256(str(sorted(cfg.items())))[:16]})
    test_ids = set(sp["test_pair_ids"])
    todo = set(a.splits.split(",")) if a.splits else set()
    if a.final_test:
        todo.add("final_test")
    if "val" in todo:
        out["val"] = experiment(R, pairs, sp["train_pair_ids"], sp["val_pair_ids"], sp["oof_fold"], variants, seeds, "val")
        write_json(rp, out)
    if "code_disjoint" in todo and not a.no_cd:
        cd = sp["code_disjoint_codes"]
        dev = pairs[~pairs["pair_id"].isin(test_ids)]
        is_cd = dev["gold_ICD10CM"].isin(set(cd["ICD10CM"])) | dev["gold_SNOMEDCT_US"].isin(set(cd["SNOMEDCT_US"]))
        out["code_disjoint"] = experiment(R, pairs, sorted(dev.loc[~is_cd, "pair_id"]), sorted(dev.loc[is_cd, "pair_id"]),
                                          sp["oof_fold_dev"], variants, seeds, "code_disjoint", cd_codes=cd)
        out["code_disjoint"]["protocol"] = "dev pairs only (FINAL TEST excluded); pairs carrying a held-out code in " \
                                           "either system removed from all training resources"
        write_json(rp, out)
    if "final_test" in todo:
        dev_ids = sorted(set(sp["train_pair_ids"]) | set(sp["val_pair_ids"]))
        out["final_test"] = experiment(R, pairs, dev_ids, sp["test_pair_ids"], sp["oof_fold_dev"], [variants[0]],
                                       seeds, "final_test")
        write_json(rp, out)
    if not a.no_final:
        trf = build_oof_features(R, pairs, sp["oof_fold_all"], cfg["rerank"]["code_dropout"])
        trf, _ = merge_ce_tables(R, "final", sorted(pairs["pair_id"]), [], trf, {s: pd.DataFrame() for s in SYSTEMS})
        drop, st2, cal, use_aug = VARIANTS[variants[0]]
        ts = TwoStage(cfg, R.kbs, R.lex, seeds[0], st2, cal, drop).fit(
            trf if use_aug else _strip_aug(trf), sp["oof_fold_all"], cfg["splits"]["oof_folds"])
        model = {"version": MODEL_VERSION, "two_stage": ts, "variant": variants[0], "tres": _tres_pair(R, pairs),
                 "train_hash": text_sha256("\n".join(pairs["pair_id"] + pairs["entity"])),
                 "kb_hashes": {s: R.kbs[s].meta.get("sha256") for s in SYSTEMS}, "config_seed": cfg["seed"]}
        save_model(model, ROOT / "models" / "final.pkl")
        print("saved models/final.pkl (refit on train+val+test, frozen config)")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
