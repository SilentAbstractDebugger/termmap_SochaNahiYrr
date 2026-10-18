"""TermMap live demo (public / synthetic inputs only).

  python demo.py --entity "MS" --context "Echo: thickened mitral valve leaflets, MS with gradient 10"
  python demo.py --hard-cases          # runs tests/hard_cases.yaml
  python demo.py --interactive         # REPL
  python demo.py --gradio              # optional web UI (pip install gradio)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from termmap.context import build_context  # noqa: E402
from termmap.joint import agreement  # noqa: E402
from termmap.normalize import make_variants  # noqa: E402
from termmap.pipeline import Predictor, load_model, load_resources  # noqa: E402
from termmap.qualifiers import extract  # noqa: E402
from termmap.util import ROOT, load_config  # noqa: E402

_STATE = {}


def _init():
    if not _STATE:
        cfg = load_config("configs/evaluation.yaml")
        R = load_resources(cfg)
        _STATE["R"] = R
        _STATE["P"] = Predictor(R, load_model(ROOT / "models" / "final.pkl"))
    return _STATE["R"], _STATE["P"]


def explain(entity: str, context: str) -> str:
    R, P = _init()
    q = pd.DataFrame({"qid": ["demo"], "entity": [entity], "chunk_text": [context]})
    tops = P.predict(q)
    ctx = build_context(entity, context)
    qual = extract(entity)
    vs = make_variants(entity, R.vocab, P.abbr)
    L = [f"entity : {entity}", f"context: {context[:160]}", ""]
    L.append(f"retrieval views: {[(v.text, v.kind) for v in vs[:8]]}")
    L.append(f"qualifiers: laterality={qual.lat} acuity={qual.acu} severity={qual.sev} stage={qual.stage} "
             f"type={qual.typ} | cues: negated={ctx.negated} history={ctx.history} uncertain={ctx.uncertain}")
    L.append(f"context window: '{ctx.window[:120]}'")
    best = {}
    for s in ("ICD10CM", "SNOMEDCT_US"):
        t = tops[s]
        L.append(f"\n{s}:")
        for r in t.itertuples():
            L.append(f"  #{r.rank + 1} {r.code:<18} p={r.prob:.3f}  {R.kbs[s].preferred.get(r.code, '')[:80]}")
        best[s] = t.iloc[0]["code"] if len(t) else ""
        L.append(f"  low-support: {P.low_support[s].get('demo')}")
    ag = agreement([best["ICD10CM"]], [best["SNOMEDCT_US"]], R.kbs, R.lex)[0]
    L.append(f"\ncross-system text agreement (ICD desc vs SNOMED desc): {ag:.3f}"
             f"{'  <- LOW: flag for review' if ag < 0.15 else ''}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity")
    ap.add_argument("--context", default="")
    ap.add_argument("--hard-cases", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--gradio", action="store_true")
    a = ap.parse_args()
    if a.entity:
        print(explain(a.entity, a.context))
    if a.hard_cases:
        for h in yaml.safe_load((ROOT / "tests" / "hard_cases.yaml").read_text()):
            print("=" * 100, f"\n[{h['source']}] {h['id']}")
            print(explain(h["entity"], h["chunk_text"]))
    if a.interactive:
        while True:
            e = input("entity> ").strip()
            if not e:
                break
            print(explain(e, input("context> ")))
    if a.gradio:
        import gradio as gr  # optional

        gr.Interface(fn=explain, inputs=[gr.Textbox(label="entity"), gr.Textbox(label="chunk context")],
                     outputs=gr.Textbox(label="mapping"), title="TermMap demo").launch()


if __name__ == "__main__":
    main()
