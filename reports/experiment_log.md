# Experiment log (iteration 2)

Development metric = group-validation (model selection) and code-disjoint (long tail). Mean +- std over 3 reranker seeds; a change is kept only if it helps beyond ~2 points or CIs separate, or if it is neutral and required (calibration, routing). FINAL TEST is not used here.

| step | what / why | validation | code-disjoint | decision |
|---|---|---|---|---|
| 0 | Re-measured baseline: run-3 features, new 3-way split (train 1874 pairs; FINAL TEST carved out) | ICD 0.5429+-0.0056, SNOMED 0.4963+-0.007 | ICD 0.3208+-0.0041, SNOMED 0.4469+-0.0045 | reference |
| 1-2 | + RRF fusion feature, + OCR glyph views, + noise-augmented training rows (25%), + learned conditional priors, + origin qualifier, + context-vs-candidate char cosine (all new features on) | ICD 0.5414+-0.0064, SNOMED 0.5044+-0.0017 | ICD 0.3275+-0.0072, SNOMED 0.4505+-0.0037 | KEEP new features (RRF, OCR views, priors, origin qualifier, ctx-vs-candidate cosine): neutral on val, +0.4..0.7 on code-disjoint |
| 2 | same tables WITHOUT the noise-augmented training rows | ICD 0.5436+-0.0037, SNOMED 0.5+-0.0016 | ICD 0.3225+-0.0012, SNOMED 0.4439+-0.001 | KEEP augmentation rows: neutral on val, +0.5/+0.7 on code-disjoint |
| 1 | without RRF features | ICD 0.5384+-0.0069, SNOMED 0.5037+-0.004 | ICD 0.3225+-0.0012, SNOMED 0.4498+-0.0018 | KEEP RRF: removing it costs 0.3/0.1 (val) and 0.5/0.1 (CD) - within noise, kept (cheap) |
| 5 | without learned conditional priors | ICD 0.5429+-0.001, SNOMED 0.505+-0.0049 | ICD 0.3325+-0.0066, SNOMED 0.4498+-0.0047 | KEEP priors (neutral within noise; required by Step 5 design) |
| - | without context features | ICD 0.5466+-0.0046, SNOMED 0.4938+-0.0032 | ICD 0.3291+-0.0024, SNOMED 0.4549+-0.0027 | KEEP context (removing: val ICD +0.5, SNOMED -1.1; CD ICD +0.2, SNOMED +0.4 - noise) |
| - | without static-vector dense features | ICD 0.5399+-0.0011, SNOMED 0.5019+-0.0015 | ICD 0.3241+-0.0042, SNOMED 0.4396+-0.0037 | KEEP static dense (removing costs 0.2/0.3 val, 0.3/1.1 CD) |
| 5 | + second-stage joint reranker (other system's top-1 agreement) | ICD 0.5667+-0.0074, SNOMED 0.4938+-0.0035 | ICD 0.3266+-0.0042, SNOMED 0.4454+-0.0048 | ICD ONLY: val ICD +2.5 (beyond 2 pts, std 0.7), CD ICD neutral; SNOMED -1.0 val, -0.5 CD -> rejected for SNOMED |
| 7 | + isotonic calibration on OOF predictions (accuracy-neutral by design; see ECE) | ICD 0.5667+-0.0074, SNOMED 0.4938+-0.0035 | ICD 0.3266+-0.0042, SNOMED 0.4454+-0.0048 | calibration accuracy-neutral by construction; kept for ECE |
| 5+7 | FROZEN: stage 2 for ICD only + isotonic calibration (both systems) | ICD 0.5667+-0.0074, SNOMED 0.5044+-0.0017 | ICD 0.3266+-0.0042, SNOMED 0.4505+-0.0037 | selected on validation; FINAL TEST scored once (SUMMARY.md) |
| 1 | Biomedical dense encoders (gte-base, SapBERT, BioLORD, MedCPT, bge-m3) | NOT RUN (no GPU / no HF access) | NOT RUN | code + `scripts/kaggle_run.sh` |
| 3 | Contrastive fine-tuning | NOT RUN for accuracy (mechanics tested with a tiny random BERT) | NOT RUN | code ready |
| 4 | Cross-encoder features | NOT RUN for accuracy (mechanics tested with a tiny random BERT) | NOT RUN | code ready |
| 6 | Local LLM adjudication | NOT RUN (no weights / GPU) | NOT RUN | not implemented |

