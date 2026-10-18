"""Step 1 benchmark: per-encoder candidate recall@K on validation and code-disjoint rows, the
"plain nearest neighbour" top-1 diagnostic, and union recall with the sparse pool.

  python scripts/dense_benchmark.py --model gte=resources/models/gte-base \
      --model sapbert=resources/models/sapbert --model biolord=resources/models/biolord-2023 \
      --model bgem3=resources/models/bge-m3 [--query-prefix e5="query: "] [--doc-prefix e5="passage: "]

Writes reports/dense_benchmark.json and .md (aggregate numbers only).
Diagnostic: share of rows whose gold code == top-1 of plain entity->description retrieval. The
supplied SNOMED file states it comes from Term Map's gte_base_fp32 index; a high figure for gte-base
would mean the gold labels were (partly) produced by that retriever, which the reranker should then
treat as a strong feature, while a low figure means coders deviate from it substantially.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from termmap.context import build_context  # noqa: E402
from termmap.data import SYSTEMS, diagnosis_pairs, load_train, system_rows  # noqa: E402
from termmap.normalize import basic_norm  # noqa: E402
from termmap.pipeline import build_eval_features, load_resources  # noqa: E402
from termmap.retrieve_dense import DenseIndex  # noqa: E402
from termmap.util import ROOT, load_config, read_json, write_json  # noqa: E402

KS = (1, 5, 10, 20, 50, 100, 200, 300)


def ranks_for(idx, system, texts, golds, k):
    res = idx.search(system, texts, k)
    out = []
    for r, g in zip(res, golds):
        seen, rank = [], None
        for di, _ in r:
            for c in idx.doc_codes[system][di]:
                if c not in seen:
                    seen.append(c)
            if g in seen and rank is None:
                rank = seen.index(g)
        out.append((rank, seen))
    return out


def recall(ranks, n):
    return {f"@{k}": round(float(np.mean([r is not None and r < k for r in ranks])), 4) for k in KS} if n else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--model", action="append", required=True, help="name=path")
    ap.add_argument("--query-prefix", action="append", default=[])
    ap.add_argument("--doc-prefix", action="append", default=[])
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--append", action="store_true", help="merge into an existing reports/dense_benchmark.json")
    a = ap.parse_args()
    qp = dict(x.split("=", 1) for x in a.query_prefix)
    dp = dict(x.split("=", 1) for x in a.doc_prefix)
    cfg = load_config(a.config)
    R = load_resources(cfg)
    pairs = diagnosis_pairs(load_train(cfg), R.kbs)
    sp = read_json(ROOT / "reports" / "splits.json")
    test = set(sp["test_pair_ids"])
    cd = sp["code_disjoint_codes"]
    dev = pairs[~pairs["pair_id"].isin(test)]
    is_cd = dev["gold_ICD10CM"].isin(set(cd["ICD10CM"])) | dev["gold_SNOMEDCT_US"].isin(set(cd["SNOMEDCT_US"]))
    splits = {
        "val": (pairs[pairs["pair_id"].isin(set(sp["train_pair_ids"]))], pairs[pairs["pair_id"].isin(set(sp["val_pair_ids"]))], None),
        "code_disjoint": (dev[~is_cd], dev[is_cd], cd),
    }
    sparse = {}
    for name, (tr, ev, cdc) in splits.items():
        evf, evrows, _ = build_eval_features(R, tr, ev, cd_codes=cdc)
        sparse[name] = (evf, evrows)
    report = report_prev = {"models": {}}
    rp = ROOT / "reports" / "dense_benchmark.json"
    if rp.exists() and a.append:
        report = read_json(rp)
    lists = {}  # (split, system) -> {model: [ranked code lists per row]}
    for spec in a.model:
        mname, path = spec.split("=", 1)
        idx = DenseIndex(cfg, R.kbs, {"name": mname, "model_path": path, "query_prefix": qp.get(mname, ""),
                                      "doc_prefix": dp.get(mname, ""), "enrich_icd_path": True})
        mr = {}
        for name, (evf, evrows) in sparse.items():
            for s in SYSTEMS:
                rows = evrows[s]
                golds = rows["gold"].tolist()
                ent = [basic_norm(e) for e in rows["entity"]]
                ctxq = [f"{e} ; {build_context(e0, c).window}".strip(" ;") for e, e0, c in
                        zip(ent, rows["entity"], rows["chunk_text"])]
                r_ent = ranks_for(idx, s, ent, golds, a.k)
                r_ctx = ranks_for(idx, s, ctxq, golds, a.k)
                lists.setdefault((name, s), {"golds": golds, "qids": rows["pair_id"].tolist()})
                lists[(name, s)][f"{mname}:ent"] = [c for _, c in r_ent]
                lists[(name, s)][f"{mname}:ctx"] = [c for _, c in r_ctx]
                f = evf[s].sort_values(["qid", "s0", "code"], ascending=[True, False, True])
                sp_lists = f.groupby("qid")["code"].apply(list).to_dict()
                union = {}
                for k in (50, 100, 200, 300):
                    hits = []
                    for q, g, (_, de), (_, dc) in zip(rows["pair_id"], golds, r_ent, r_ctx):
                        pool = set(sp_lists.get(q, [])[:k]) | set(de[:k]) | set(dc[:k])
                        hits.append(g in pool)
                    union[f"@{k}"] = round(float(np.mean(hits)), 4)
                sp_only = {}
                for k in (50, 100, 200, 300):
                    sp_only[f"@{k}"] = round(float(np.mean([g in set(sp_lists.get(q, [])[:k])
                                                            for q, g in zip(rows["pair_id"], golds)])), 4)
                mr[f"{name}/{s}"] = {
                    "n": len(rows),
                    "top1_equals_gold_entity_query": recall([r for r, _ in r_ent], len(rows)).get("@1"),
                    "recall_entity": recall([r for r, _ in r_ent], len(rows)),
                    "recall_entity_plus_context": recall([r for r, _ in r_ctx], len(rows)),
                    "sparse_pool_recall": sp_only,
                    "union_sparse_dense_recall": union}
                print(mname, name, s, mr[f"{name}/{s}"]["recall_entity"], "union", union, flush=True)
        report["models"][mname] = {"path": path, "results": mr}
    # union / RRF across sparse + every dense model and view benchmarked in THIS invocation
    fused = {}
    for (name, s), d in lists.items():
        evf, _ = sparse[name]
        f = evf[s].sort_values(["qid", "s0", "code"], ascending=[True, False, True])
        sp_lists = f.groupby("qid")["code"].apply(list).to_dict()
        views = [k for k in d if ":" in k]
        un, rr = {}, {}
        for k in (10, 50, 100, 200, 300):
            hu, hr = [], []
            for i, (q, g) in enumerate(zip(d["qids"], d["golds"])):
                ranked = [sp_lists.get(q, [])] + [d[v][i] for v in views]
                pool = set().union(*[set(r[:k]) for r in ranked])
                hu.append(g in pool)
                sc = {}
                for r in ranked:
                    for j, c in enumerate(r[:300]):
                        sc[c] = sc.get(c, 0.0) + 1.0 / (60 + j)
                top = [c for c, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:k]]
                hr.append(g in top)
            un[f"@{k}"] = round(float(np.mean(hu)), 4)
            rr[f"@{k}"] = round(float(np.mean(hr)), 4)
        fused[f"{name}/{s}"] = {"views": views, "union_recall_per_channel_topk": un, "rrf_recall": rr}
        print("fused", name, s, "union", un, "rrf", rr, flush=True)
    report["fused_this_run"] = fused
    write_json(rp, report)
    L = ["# Dense retrieval benchmark", ""]
    for m, r in report["models"].items():
        L += [f"## {m} ({r['path']})", "", "| split/system | top-1=gold | R@10 | R@100 | R@300 | +ctx R@100 | sparse R@100 | union R@100 |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in r["results"].items():
            L.append(f"| {k} | {v['top1_equals_gold_entity_query']} | {v['recall_entity'].get('@10')} | "
                     f"{v['recall_entity'].get('@100')} | {v['recall_entity'].get('@300')} | "
                     f"{v['recall_entity_plus_context'].get('@100')} | {v['sparse_pool_recall'].get('@100')} | "
                     f"{v['union_sparse_dense_recall'].get('@100')} |")
        L.append("")
    if report.get("fused_this_run"):
        L += ["## Union / RRF across sparse + all dense models/views of the last run", "",
              "| split/system | union R@100 | union R@300 | RRF R@100 | RRF R@300 |", "|---|---|---|---|---|"]
        for k, x in report["fused_this_run"].items():
            u, r = x["union_recall_per_channel_topk"], x["rrf_recall"]
            L.append(f"| {k} | {u['@100']} | {u['@300']} | {r['@100']} | {r['@300']} |")
    (ROOT / "reports" / "dense_benchmark.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
