"""Create leakage-proof splits -> reports/splits.json (ids, seed, data hash)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from termmap.data import SYSTEMS, diagnosis_pairs, load_train  # noqa: E402
from termmap.splits import make_splits  # noqa: E402
from termmap.terminology import load_kbs  # noqa: E402
from termmap.util import ROOT, load_config, write_json  # noqa: E402


def main(config=None):
    cfg = load_config(config)
    kbs = load_kbs(cfg)
    pairs = diagnosis_pairs(load_train(cfg), kbs)
    s = cfg["splits"]
    sp = make_splits(pairs, cfg["seed"], s["val_fraction"], s["code_disjoint_fraction"], s["oof_folds"],
                     s.get("test_fraction", 0.0))
    summary = {"n_pairs": len(pairs), "n_train_pairs": len(sp["train_pair_ids"]),
               "n_val_pairs": len(sp["val_pair_ids"]), "n_test_pairs": len(sp["test_pair_ids"]), "n_groups": sp["n_groups"],
               "largest_group": sp["largest_group"]}
    for sys_ in SYSTEMS:
        cd = set(sp["code_disjoint_codes"][sys_])
        summary[f"cd_codes_{sys_}"] = len(cd)
        summary[f"cd_rows_{sys_}"] = int(pairs[f"gold_{sys_}"].isin(cd).sum())
        tr = set(pairs.loc[pairs["pair_id"].isin(set(sp["train_pair_ids"]) | set(sp["val_pair_ids"])), f"gold_{sys_}"])
        te = pairs[pairs["pair_id"].isin(set(sp["test_pair_ids"])) & pairs[f"has_{sys_}"]]
        summary[f"test_rows_{sys_}"] = int(len(te))
        summary[f"test_rows_code_unseen_in_train_val_{sys_}"] = int((~te[f"gold_{sys_}"].isin(tr)).sum())
    sp["summary"] = summary
    write_json(ROOT / "reports" / "splits.json", sp)
    print(summary)
    return sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
