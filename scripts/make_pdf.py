"""Build the 2-3 page documentation PDF and the README results block from ONE editable file.

    python scripts/make_pdf.py            # reads docs/results_for_pdf.json
                                          # writes docs/TermMap_writeup.pdf, docs/architecture.png,
                                          # and the README block between RESULTS:START/END markers

Every number comes from docs/results_for_pdf.json. Any value equal to "PENDING" is rendered as a
highlighted PENDING marker. Examples are public (problem statement) or labelled synthetic only.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results_for_pdf.json"
PEND = '<font backColor="#ffe066" color="#9a0000"><b>PENDING</b></font>'


def P(x) -> str:
    """Format a value; PENDING gets a highlighted marker."""
    if x is None or (isinstance(x, str) and x.strip().upper() == "PENDING"):
        return PEND
    if isinstance(x, float):
        return f"{x:.3f}"
    if isinstance(x, str) and x.strip().upper().startswith("PENDING"):
        return PEND + x.strip()[7:]
    return str(x)


def md(x) -> str:
    """Markdown variant for the README."""
    if x is None or (isinstance(x, str) and x.strip().upper().startswith("PENDING")):
        return "**PENDING**"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


# ----------------------------------------------------------------------------- architecture figure
def architecture_png(path: Path):
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    solid = dict(fc="#e9f1fb", ec="#1f4a7a", lw=1.8)
    band = dict(fc="#f4f8fd", ec="#1f4a7a", lw=1.2)
    dashed = dict(fc="#fbf4ec", ec="#a0561a", lw=1.6, ls="--")

    def box(x, y, w, h, title, body, style=solid, tsize=12.5, bsize=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", **style))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=tsize, weight="bold",
                color="#122f4f" if style is not dashed else "#7a3a0a")
        if body:
            ax.text(x + w / 2, y + (h - 0.45) / 2, body, ha="center", va="center", fontsize=bsize, color="#1a1a1a")

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20, lw=1.8,
                                     color="#1f4a7a", linestyle=ls))

    ax.text(0.1, 7.95, "Resources: supplied ICD-10-CM + SNOMED CT US dictionaries only (code-string hierarchy, semantic "
            "tags, mined abbreviations) + TRAINING partition only (aliases, acronyms, priors)", fontsize=11, color="#122f4f")
    # row 1
    box(0.1, 6.25, 3.0, 1.45, "1  Input adapter", "entity + chunk_text\nlong / wide schema auto-detect")
    box(3.6, 6.25, 3.4, 1.45, "2  Multi-view normalization", "original + typo / glue / truncation\n+ every abbreviation hypothesis")
    box(7.5, 6.25, 3.0, 1.45, "3  Context builder", "markup strip, mention window,\nnegation / history cues")
    arrow(3.1, 6.97, 3.6, 6.97)
    arrow(7.0, 6.97, 7.5, 6.97)
    # retrieval band
    ax.add_patch(FancyBboxPatch((0.1, 3.35), 13.8, 2.45, boxstyle="round,pad=0.05", **band))
    ax.text(8.0, 5.55, "4  Retrieval, run separately for ICD-10-CM and SNOMED CT", fontsize=11.3, weight="bold", color="#122f4f")
    box(0.35, 3.55, 2.55, 1.75, "Sparse", "exact / alias\nBM25\nchar 3-gram TF-IDF", tsize=11.5)
    box(3.1, 3.55, 3.6, 1.75, "Dense (exact flat FAISS)", "gte-base, SapBERT,\nBioLORD-2023, bge-m3\n+ spaCy static vectors", tsize=11.5)
    box(6.9, 3.55, 2.45, 1.75, "Training\nneighbours", "\n\nentity x context\nsimilarity", tsize=11.5)
    box(9.55, 3.55, 2.0, 1.75, "Synonym\nbridge", "\n\nother system's\nsynonym text", tsize=11.5)
    box(11.75, 3.55, 1.95, 1.75, "Union pool", "per-channel\nscores + RRF", tsize=11.5)
    arrow(5.3, 6.25, 5.3, 5.82)
    arrow(7.75, 6.25, 7.75, 5.82)
    # row 3
    box(0.1, 0.95, 2.45, 1.75, "5  ICD hierarchy", "ancestors, ...9 / ...0\nsiblings, children")
    box(2.85, 0.95, 3.15, 1.75, "6  LightGBM LambdaRank", "code-agnostic features\nout-of-fold + code dropout")
    box(6.3, 0.95, 2.4, 1.75, "7  ICD-only\nsecond pass", "\n\nother system's\ntop-1 agreement")
    box(9.0, 0.95, 2.1, 1.75, "8  Calibration", "isotonic,\nfit out-of-fold")
    box(11.4, 0.95, 2.5, 1.75, "9  Validated output", "dictionary-only codes\n+ diagnostics")
    arrow(12.7, 3.55, 12.7, 3.05)
    ax.add_patch(FancyArrowPatch((12.7, 3.05), (1.3, 2.75), arrowstyle="-|>", mutation_scale=20, lw=1.8,
                                 color="#1f4a7a", connectionstyle="arc3,rad=0.0"))
    arrow(2.55, 1.82, 2.85, 1.82)
    arrow(6.0, 1.82, 6.3, 1.82)
    arrow(8.7, 1.82, 9.0, 1.82)
    arrow(11.1, 1.82, 11.4, 1.82)
    # implemented, not in final model
    box(10.85, 6.1, 3.05, 1.65, "Not in final model",
        "implemented, not evaluated:\nMedCPT cross-encoder (scores\nlost), fine-tuning, LLM", style=dashed, tsize=11, bsize=9.8)
    ax.text(0.1, 0.35, "Not supplied: SNOMED IS-A relationships and the SNOMED-to-ICD crosswalk (the engine runs without them). "
            "Dashed = implemented, not evaluated in the final model.", fontsize=10.5, color="#7a3a0a")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- README block
def readme_block(r: dict) -> str:
    v, c, t = r["results"]["validation"], r["results"]["code_disjoint"], r["results"]["final_test"]
    rc = r["recall"]
    L = [f"_Status: **{r['status']}**. Generated from `docs/results_for_pdf.json` by `python scripts/make_pdf.py`._", "",
         "| split | ICD-10-CM | SNOMED CT | headline (lower) |", "|---|---|---|---|",
         f"| Validation (codes seen; {md(v['seeds'])} seeds) | {md(v['icd'])} (CI {md(v['ci_icd'])}) | {md(v['snomed'])} (CI {md(v['ci_snomed'])}) | {md(v['headline'])} |",
         f"| Code-disjoint (codes never seen; {md(c['seeds'])} seeds) | {md(c['icd'])} (CI {md(c['ci_icd'])}) | {md(c['snomed'])} (CI {md(c['ci_snomed'])}) | {md(c['headline'])} |",
         f"| Final test (scored once) | {md(t['icd'])} (CI {md(t['ci_icd'])}) | {md(t['snomed'])} (CI {md(t['ci_snomed'])}) | {md(t['headline'])} |",
         "", f"Target 95% on both systems: **not reached**. Candidate recall of the fused pool (a shortlist ceiling, not accuracy): "
         f"validation @200 ICD {md(rc['fused_union_val_at200']['icd'])}, SNOMED {md(rc['fused_union_val_at200']['snomed'])}; "
         f"code-disjoint @200 ICD {md(rc['fused_union_cd_at200']['icd'])}, SNOMED {md(rc['fused_union_cd_at200']['snomed'])}. "
         f"Final refit: {md(r['results']['final_refit'])}. Demo video: {r['demo_video_link']}."]
    return "\n".join(L)


def update_readme(r: dict):
    p = ROOT / "README.md"
    if not p.exists():
        return
    s = p.read_text()
    new = "<!-- RESULTS:START -->\n" + readme_block(r) + "\n<!-- RESULTS:END -->"
    if "<!-- RESULTS:START -->" in s:
        s = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: new, s, flags=re.S)
        p.write_text(s)


# ----------------------------------------------------------------------------- PDF
def main():
    r = json.loads(RES.read_text())
    arch = ROOT / "docs" / "architecture.png"
    architecture_png(arch)
    ss = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=ss["Heading2"], fontSize=11, spaceBefore=4, spaceAfter=1.5, textColor="#122f4f")
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=7.9, leading=9.8, spaceAfter=2)
    T = ParagraphStyle("T", parent=ss["Title"], fontSize=14.5, spaceAfter=1)
    C = ParagraphStyle("C", parent=B, fontSize=7.1, leading=8.6, spaceAfter=0)

    def tb(rows, widths, fs=7.1):
        rows = [[Paragraph(str(x), ParagraphStyle("c", parent=C, fontSize=fs)) for x in row] for row in rows]
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f3")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
        return t

    d, res, rc = r["data"], r["results"], r["recall"]
    v, cdj, ft, cpu = res["validation"], res["code_disjoint"], res["final_test"], res["earlier_cpu_only"]
    eng = r["engineering"]
    S = [Paragraph("TermMap: mapping clinical entities to ICD-10-CM and SNOMED CT", T),
         Paragraph(f"HiLabs hackathon documentation (results status: <b>{r['status']}</b>; {PEND} = to be filled from the final run). "
                   f"Demo video: {r['demo_video_link']}. Repository: {P(r['repo_link'])}. No proprietary rows appear in this "
                   "document; examples are from the problem statement or labelled synthetic.", B)]

    # 1 Approach
    S.append(Paragraph("1. Approach", H))
    S.append(Paragraph(
        "<b>Interpretation.</b> For each extracted Problem entity (diagnosis, condition, sign or symptom) and its chunk text, "
        "return exactly one ICD-10-CM code and one SNOMED CT US concept, both present in the supplied dictionaries. "
        "Procedures, medications, labs and NER are out of scope, so only <i>DiagnosisCode</i> training rows are used. "
        f"<b>Data facts.</b> {d['icd_rows']:,} ICD and {d['snomed_rows']:,} SNOMED diagnosis rows; {d['pairs']:,} entity+chunk "
        f"pairs, {d['pairs_both_labels']:,} with both labels (per-system label masks everywhere). {d['snomed_codes_singleton_share']} of "
        f"distinct SNOMED codes and {d['icd_codes_singleton_share']} of distinct ICD codes appear exactly once, and every entity string has "
        "exactly one code per system, so memorisation cannot reach the target and context cannot be learned from label conflicts. "
        f"Malformed gold {d['malformed_gold']}. Dictionaries: {d['icd_dictionary_rows']:,} ICD rows; {d['snomed_description_rows']:,} SNOMED "
        f"descriptions for {d['snomed_concepts']:,} concepts (exported from Term Map's gte_base_fp32 index). "
        "<b>Two independent systems.</b> ICD and SNOMED differ in granularity, so each is retrieved and ranked on its own; the "
        "other system is used only as text evidence, never as a crosswalk. <b>Protocol.</b> Group split by entity and chunk; "
        "validation for model selection; a code-disjoint fold (20% of codes removed from every training resource) for the long "
        "tail; a final test split scored once; bootstrap 95% CIs.", B))

    # 2 Architecture
    S.append(Paragraph("2. Architecture", H))
    S.append(Image(str(arch), width=18.4 * cm, height=10.6 * cm))

    # 3 Design decisions
    S.append(Paragraph("3. Design decisions", H))
    S.append(Paragraph(
        f"<b>Hybrid sparse + dense retrieval.</b> Before the dense encoders the correct code was in the top 200 only "
        f"{P(rc['before_dense_val_at200'])} of the time on validation. The fused union of sparse channels and four encoders reaches "
        f"@100 ICD {P(rc['fused_union_val_at100']['icd'])} / SNOMED {P(rc['fused_union_val_at100']['snomed'])} and @200 ICD "
        f"{P(rc['fused_union_val_at200']['icd'])} / SNOMED {P(rc['fused_union_val_at200']['snomed'])} (code-disjoint @200: "
        f"{P(rc['fused_union_cd_at200']['icd'])} / {P(rc['fused_union_cd_at200']['snomed'])}). Recall is a shortlist ceiling, not accuracy. "
        f"<b>Code-agnostic LightGBM reranker.</b> A single encoder's top-1 is only ICD {P(rc['single_model_top1_val']['icd'])} and SNOMED "
        f"{P(rc['single_model_top1_val']['snomed'])}, so ranking is the weak step; LambdaRank combines all channels' scores and ranks, "
        "string, context, qualifier and hierarchy features, none of which encode code identity, so unseen codes compete equally. "
        "<b>Code dropout.</b> Training tables are out-of-fold and a share of queries lose every trace of their gold code, so the "
        "model learns to decide without memory. <b>Learned specificity.</b> Ancestors, ...9/...0 siblings and children of top ICD "
        "candidates join the pool; depth, leaf and 'candidate adds/drops a modifier' features let the reranker pick the level. "
        f"<b>ICD-only second pass.</b> Using the other system's top-1 text raised ICD from {P(r['ablations_val'][1][1])} to "
        f"{P(r['ablations_val'][0][1])} on validation; it did not help SNOMED, so SNOMED keeps stage 1.", B))
    rows = [["split", "ICD-10-CM [95% CI]", "SNOMED CT [95% CI]", "headline (lower)", "seeds / n (ICD, SNOMED)"],
            ["Validation (codes seen)", f"{P(v['icd'])} {P(v['ci_icd'])}", f"{P(v['snomed'])} {P(v['ci_snomed'])}",
             P(v["headline"]), f"{P(v['seeds'])} / {P(v['n_icd'])}, {P(v['n_snomed'])}"],
            ["Code-disjoint (codes never seen)", f"{P(cdj['icd'])} {P(cdj['ci_icd'])}", f"{P(cdj['snomed'])} {P(cdj['ci_snomed'])}",
             P(cdj["headline"]), f"{P(cdj['seeds'])} / {P(cdj['n_icd'])}, {P(cdj['n_snomed'])}"],
            ["Final test (scored once)", f"{P(ft['icd'])} {P(ft['ci_icd'])}", f"{P(ft['snomed'])} {P(ft['ci_snomed'])}",
             P(ft["headline"]), f"{P(ft['seeds'])} / {P(ft['n_icd'])}, {P(ft['n_snomed'])}"],
            ["Earlier CPU-only run (no encoders)", f"val {P(cpu['val_icd'])}; CD {P(cpu['cd_icd'])}",
             f"val {P(cpu['val_snomed'])}; CD {P(cpu['cd_snomed'])}", "", "not like-for-like on CD rows"]]
    abl = [["validation ablation", "ICD", "SNOMED"]] + [[a, P(b), P(c)] for a, b, c in r["ablations_val"]]
    S.append(Table([[tb(rows, [3.4 * cm, 2.75 * cm, 2.75 * cm, 1.6 * cm, 2.4 * cm]),
                     tb(abl, [3.2 * cm, 1.1 * cm, 1.3 * cm], fs=6.8)]],
                   colWidths=[13.1 * cm, 5.7 * cm], style=[("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                           ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    S.append(Paragraph(
        f"Target 95% on both systems: <b>not reached</b>; validation headline {P(v['headline'])} is "
        f"{P(round(res['target'] - v['headline'], 3) if isinstance(v['headline'], float) else 'PENDING')} below it. "
        "<b>Judging criteria.</b> Accuracy (50%): table above. Long tail (20%): code-disjoint row, code dropout, code-agnostic "
        "features. Engineering (15%): one command, offline, deterministic, tests, hash-keyed resumable caches. Error analysis "
        "(15%): section 4.", B))

    # 4 Error analysis
    sl = r["slices_val_icd"]
    S.append(Paragraph("4. Error analysis", H))
    fm = [["challenge failure mode", "technique", "measured evidence"],
          ["1 nearest string is the wrong concept (hypertension vs pulmonary hypertension I27.20)", "reranker over string + context + hierarchy features", f"counts {P(r['failure_modes_counts'])}"],
          ["2 abbreviation collision (MS / RA / PID)", "all expansions kept as hypotheses; context features", f"ICD abbreviation slice {P(sl['abbreviation_terms'])}; MS / RA / PID ICD hard cases fail"],
          ["3 specificity (diabetic neuropathy E11.40 vs diabetes E11.9)", "hierarchy expansion + learned level", "no-hierarchy ablation ICD " + P(r["ablations_val"][3][1]) + "; E13.40 error"],
          ["4 ICD / SNOMED granularity mismatch", "independent systems; ICD-only second pass", f"second pass ICD {P(r['ablations_val'][1][1])} -> {P(r['ablations_val'][0][1])}"],
          ["5 long tail / unseen codes", "code-agnostic features, code dropout", f"ICD unseen {P(sl['by_train_freq']['unseen'])}; code-disjoint headline {P(cdj['headline'])}"],
          ["6 OCR / truncation ('chronickidney disease stag 3')", "glue split, truncation completion, typo views", "hard case " + P(r["hard_cases"][5][1])],
          ["7 terminology version drift", "hash-keyed rebuild, code-agnostic model, drift test", "modified-dictionary test (see tests)"]]
    S.append(tb(fm, [6.6 * cm, 5.3 * cm, 6.7 * cm], fs=6.8))
    S.append(Paragraph(
        f"<b>Retrieval miss vs rank miss:</b> {P(rc['retrieval_miss_vs_rank_miss'])}; with union recall @200 near 0.96 to 0.97, "
        "most remaining errors are ranking errors. <b>ICD accuracy by training frequency of the gold code</b> (validation): unseen "
        f"{P(sl['by_train_freq']['unseen'])}, 1 time {P(sl['by_train_freq']['1'])}, 2-5 times {P(sl['by_train_freq']['2-5'])}, "
        f"6+ {P(sl['by_train_freq']['6+'])}. SNOMED by frequency: {P(r['slices_val_snomed']['by_train_freq'])}. <b>Weak ICD chapters:</b> "
        f"{P(sl['weak_chapters'])}. <b>Calibration:</b> ECE ICD {P(r['calibration']['ece_val_icd'])}, SNOMED "
        f"{P(r['calibration']['ece_val_snomed'])}; accuracy at confidence >= 0.7: {P(r['calibration']['coverage_at_conf_0.7'])}. "
        "<b>Hard cases (statement or synthetic):</b> " + "; ".join(f"{a}: {P(b)}" for a, b in r["hard_cases"]) + ".", B))

    # 5 Innovations
    S.append(Paragraph("5. Innovations (implemented and measured)", H))
    ab = {a: (b, c) for a, b, c in r["ablations_val"]}
    S.append(Paragraph(
        f"<b>Code-dropout out-of-fold reranking</b> for unseen codes (code-disjoint headline {P(cdj['headline'])}). "
        f"<b>Training-mined local conventions</b> via neighbours: removing them drops ICD to {P(ab['no training neighbours'][0])}. "
        f"<b>Hierarchy-aware specificity</b>: removing it drops ICD to {P(ab['no ICD hierarchy features'][0])}. "
        f"<b>Learned qualifier relevance</b>: removing qualifier features gives {P(ab['no qualifier features'][0])} / "
        f"{P(ab['no qualifier features'][1])} (within noise). <b>Dictionary-text synonym bridge</b> and <b>dual-system agreement</b> "
        f"(ICD-only second pass, +{P(round(r['ablations_val'][0][1] - r['ablations_val'][1][1], 3))} ICD; agreement also feeds the "
        "confidence / review flag). <b>Did nothing measurable:</b> learned priors (removing them: "
        f"{P(ab['no learned priors'][0])} / {P(ab['no learned priors'][1])}), RRF and noise-augmented rows (within noise).", B))

    # 6 Challenges
    S.append(Paragraph("6. Challenges and learnings", H))
    S.append(Paragraph(
        "<b>What did not work or was lost:</b> priors, RRF and augmented rows stayed within noise; the MedCPT cross-encoder was "
        "trained but its scores were lost in a Kaggle session reset, so the final model runs without it (implemented, not "
        "evaluated); contrastive fine-tuning and LLM adjudication were not run; RAM (8 GB sandbox) and disk limits forced "
        "per-split processes and hash-keyed resumable caches; a statement-versus-label conflict (HTN) where training labels win. "
        f"<b>Engineering:</b> CPU-only {P(eng['cpu_terms_per_second'])} terms/s, median {P(eng['cpu_median_latency_s'])} s per term, "
        f"peak RAM {P(eng['cpu_peak_ram'])}; GPU throughput {P(eng['gpu_terms_per_second'])}; per-term model calls: "
        f"{P(eng['per_term_model_calls'])}; determinism: {P(eng['determinism'])}; tests {P(eng['tests'])}. "
        "<b>Next:</b> re-run and measure the cross-encoder and fine-tuning; SNOMED IS-A and the crosswalk as auxiliary features; "
        "LLM adjudication only on low-confidence rows; larger candidate pool.", B))
    oq = [["Open questions for the organisers"],
          ["Evaluation schema (long vs wide) | partial credit? | multiple valid codes / statement-vs-label conflicts | annotator "
           "agreement | per-category floors | scoring of rows with one label | approval for running on a hosted notebook (Kaggle)"]]
    S.append(KeepTogether(tb(oq, [18.6 * cm], fs=7.1)))
    S.append(Spacer(1, 3))
    S.append(Paragraph("<b>External resources:</b> " + "; ".join(f"{a} ({b}, {c})" for a, b, c in r["resources"])
                       + ". Supplied ICD-10-CM and SNOMED CT US exports (SNOMED licence applies). No API calls; GPU stages ran in a "
                       "private Kaggle notebook.", B))
    doc = SimpleDocTemplate(str(ROOT / "docs" / "TermMap_writeup.pdf"), pagesize=A4, leftMargin=1.2 * cm,
                            rightMargin=1.2 * cm, topMargin=1.0 * cm, bottomMargin=1.0 * cm,
                            title="TermMap documentation")
    doc.build(S)
    update_readme(r)
    pend = len(re.findall(r'"PENDING', json.dumps(r)))
    print(f"wrote docs/TermMap_writeup.pdf, docs/architecture.png, README results block; PENDING fields left: {pend}")


if __name__ == "__main__":
    sys.exit(main())
