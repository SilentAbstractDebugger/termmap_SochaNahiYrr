# Demo video script (target 2:40, hard limit 3:00)

Inputs are **only** problem-statement examples or labelled synthetic sentences. Never open `data/`, `reports/private/`
or any CSV on screen. `demo.py` prints, for each input:
- the retrieval views and the qualifiers / context cues;
- the top-3 ICD-10-CM and SNOMED CT candidates with probabilities and descriptions;
- a low-support flag and the ICD-vs-SNOMED text-agreement score.

Do not read numbers from this script: **say what the screen shows**, using the conditional lines.

## Before recording

```bash
cd termmap && source .venv/bin/activate
python scripts/make_synthetic_eval.py              # outputs/synthetic_eval.csv from tests/hard_cases.yaml (public/synthetic)
python demo.py --entity "chest pain" > /dev/null    # warm-up: loads the indexes once
clear
```

## 0:00-0:40 Architecture walkthrough

**Screen:** open `docs/architecture.png` full screen and move the cursor across stages 1 to 9.

**Say:**
"TermMap maps one clinical entity and its sentence to one ICD-10-CM code and one SNOMED CT code. Both must exist in
the supplied dictionaries. First, we normalize the term into several views: the original, typo, glue and truncation
repairs, and every abbreviation expansion, kept side by side. We also read the context around the mention. Retrieval
runs separately for each system: exact match, BM25, character n-grams, training neighbours, a synonym bridge, and four
dense encoders with an exact FAISS index. Everything goes into one candidate pool. For ICD we add parent and sibling
codes so the model can choose how specific to be. A LightGBM ranker with code-agnostic features picks the winner. It
was trained with code dropout, so codes it has never seen can still win. ICD gets a second pass that checks agreement
with the SNOMED pick. Then probabilities are calibrated and every code is validated against the dictionary. The
dashed box is implemented but not in the final model: the cross-encoder, fine-tuning and LLM adjudication."

## 0:40-1:50 Live mapping

**0:40 Abbreviation collision (synthetic).**

```bash
python demo.py --entity "MS" --context "Echo: thickened mitral valve leaflets, MS with mean gradient 10 mmHg."
```

**Say:** "MS is ambiguous. Look at the retrieval views: multiple sclerosis and mitral stenosis are both kept as hypotheses."
- *If SNOMED #1 is a mitral stenosis concept:* "SNOMED reads the valve context and picks mitral stenosis."
- *If the ICD #1 description is not mitral stenosis:* "ICD gets this one wrong. The right sense is in the
  candidate list but ranked lower. This is one of our known failures, and the agreement score below flags the disagreement."
- *If ICD #1 is a mitral stenosis code:* "and ICD agrees."

**1:05 OCR-mangled term (problem statement).**

```bash
python demo.py --entity "chronickidney disease stag 3" --context "Problem list: chronickidney disease stag 3"
```

**Say:** "The glued word and the truncated 'stag' are repaired as an extra view, and the original is still searched."
- *If ICD #1 is a stage-3 chronic kidney disease code:* "It lands on chronic kidney disease stage 3."
- *Otherwise:* "Here the repair did not lead to the right code. That is part of the noise slice in the error table."

**1:25 Specificity call (problem statement).**

```bash
python demo.py --entity "diabetic neuropathy" --context "Type 2 DM complicated by diabetic neuropathy of both feet."
python demo.py --entity "diabetes" --context "History of diabetes, on metformin."
```

**Say:** "The statement expects E11.40 for diabetic neuropathy and E11.9 for plain diabetes."
- *If the first shows E13.40:* "Our model picks E13.40, the 'other specified diabetes' family, instead of E11.40. It
  is a known, documented error. The level is right but the family is wrong, even though the context says type 2."
- *If the first shows E11.40:* "It picks E11.40."
- *For the second, if it shows E11.9:* "Plain diabetes stays at E11.9, without complications."
- *Otherwise:* read the code shown and say it differs from the expected E11.9.

**1:40 HTN (problem statement).**

```bash
python demo.py --entity "HTN" --context "Assessment: HTN, continue lisinopril."
```

**Say:**
- *If ICD #1 is I10:* "ICD gives I10, essential hypertension, as expected."
- "On SNOMED, the statement shows 59621000, but the training labels map hypertension to 38341003, and training wins.
  We list this as an open question for the organizers."
- *If SNOMED shows another code:* read it and its description as shown.

## 1:50-2:30 Confidence and error analysis

**Screen:** scroll back to the MS output.

**Say:** "Every prediction carries a calibrated probability, the top-3 alternatives, and an ICD-versus-SNOMED agreement
score. When agreement is low or the margin is small, the row is routed for review in a separate diagnostics file. The
required output format never changes."
- *If the MS agreement line shows "LOW":* "The MS case is exactly such a disagreement."

**Screen:** open `docs/TermMap_writeup.pdf` at section 4 (error-analysis table), or `reports/error_analysis_val.md`.

**Say:** "Here are the errors by failure mode. The candidate pool now contains the right code about 96 to 97 percent
of the time on validation, so most remaining errors are ranking errors. Codes never seen in training are the hardest.
We did not reach 95 percent. The table shows the real gap."

## 2:30-2:50 One-command reproduction and results

```bash
PYTHONPATH=src python -m termmap.predict --input outputs/synthetic_eval.csv --output outputs/demo_pred.csv --config configs/evaluation.yaml
head -5 outputs/demo_pred.csv
```

(`outputs/synthetic_eval.csv` contains only the public / synthetic hard cases.)

**Say:** "One command, offline, no API keys. Same input order, both codes for every row, every code checked against
the dictionary."

**Screen:** show the README results table.

**Say:** "Final numbers are in the README and the PDF."
- Read the validation, code-disjoint and final-test headline values **exactly as shown**.
- *If any cell still says PENDING, do not record yet.*

## Recording checklist

- [ ] 1920x1080, terminal font 16 pt or larger, dark-on-light theme, notifications off.
- [ ] Only public / synthetic inputs are shown. No `data/`, `resources/`, `reports/private/`, Kaggle data panel or
      file names that reveal proprietary data. Close the file explorer and hide shell history.
- [ ] README and PDF have no PENDING cells (run `python scripts/make_pdf.py` after filling the results JSON).
- [ ] Length 2:30-3:00. Export MP4, upload to the internal drive with link access for the judges only.
- [ ] Paste the link into `docs/results_for_pdf.json` (`demo_video_link`) and the README header, then run
      `python scripts/make_pdf.py`.
