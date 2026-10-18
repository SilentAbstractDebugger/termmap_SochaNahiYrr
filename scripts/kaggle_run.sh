#!/usr/bin/env bash
# TermMap GPU runbook (Kaggle). See docs/KAGGLE_RUNBOOK.md for notebook settings and timings.
#
#   bash scripts/kaggle_run.sh            # all stages a..h in order
#   bash scripts/kaggle_run.sh b c d      # only some stages
#
# Every stage is resumable: embeddings / feature tables / CE fold models / fine-tune checkpoints are
# hash-keyed caches, and a finished stage leaves logs/kaggle/<stage>.done (delete it to force a rerun).
# Progress goes to logs/kaggle/<stage>.log (tail -f it).
# PRIVACY: data stays in private notebook storage; models are downloaded IN; nothing is uploaded.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LOG=logs/kaggle; mkdir -p "$LOG"
CFG=${CFG:-configs/kaggle_dense.yaml}
MODELS_DIR=${MODELS_DIR:-resources/models}
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(a b c d e f g h)

run() {  # run <stage> <command...>
  local st=$1; shift
  if [ -f "$LOG/$st.done" ]; then echo "[$st] already done (rm $LOG/$st.done to rerun)"; return; fi
  echo "[$st] start $(date -u +%FT%TZ): $*" | tee -a "$LOG/$st.log"
  "$@" 2>&1 | tee -a "$LOG/$st.log"
  date -u +%FT%TZ > "$LOG/$st.done"
  echo "[$st] done" | tee -a "$LOG/$st.log"
}

stage_a() {  # GPU + HF access check, deps, model download (commit sha + licence recorded)
  nvidia-smi || echo "WARNING: no GPU visible - stages b-f will run on CPU (very slow)"
  python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  pip install -q -r requirements.txt "sentence-transformers==5.1.1" faiss-cpu "datasets<4.2" "accelerate>=0.26" "huggingface-hub>=0.34,<1.0"
  if curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://huggingface.co/api/models/thenlper/gte-base | grep -q 200; then
    python scripts/setup_resources.py --download-dense --only \
      thenlper/gte-base,cambridgeltl/SapBERT-from-PubMedBERT-fulltext,FremyCompany/BioLORD-2023,BAAI/bge-m3,ncbi/MedCPT-Cross-Encoder
  else
    echo "No Hugging Face access: models must already be in $MODELS_DIR (attach them as a Kaggle dataset, see runbook)"
    for m in gte-base sapbert biolord-2023 bge-m3 medcpt-cross; do test -d "$MODELS_DIR/$m" || { echo "MISSING $MODELS_DIR/$m"; exit 1; }; done
  fi
  python scripts/build_index.py --config configs/evaluation.yaml   # sparse caches (CPU)
}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for st in "${STAGES[@]}"; do
  case $st in
    a) run a stage_a ;;
    b) run b python scripts/dense_benchmark.py --config configs/evaluation.yaml \
           --model gte=$MODELS_DIR/gte-base ;;                       # embeds ICD+SNOMED once (cached), recall@K val + code-disjoint
    c) run c python scripts/dense_benchmark.py --config configs/evaluation.yaml --append \
           --model sapbert=$MODELS_DIR/sapbert --model biolord=$MODELS_DIR/biolord-2023 --model bgem3=$MODELS_DIR/bge-m3 ;;
    d) run d python scripts/dense_benchmark.py --config configs/evaluation.yaml --append \
           --model gte=$MODELS_DIR/gte-base --model sapbert=$MODELS_DIR/sapbert \
           --model biolord=$MODELS_DIR/biolord-2023 --model bgem3=$MODELS_DIR/bge-m3 ;;   # union + RRF recall (embeddings reused from cache)
    e) for EXP in val code_disjoint final_test final; do
         run e_$EXP python scripts/train_cross_encoder.py --config "$CFG" --experiment $EXP \
             --base-model $MODELS_DIR/medcpt-cross --top-n 30 --neg 15 --epochs 1
       done ;;
    f) BEST=${BEST_ENCODER:-sapbert}
       run f python scripts/finetune_encoder.py --base $MODELS_DIR/$BEST --out models/encoders/$BEST-ft \
           --experiment code_disjoint --epochs 1
       run f_bench python scripts/dense_benchmark.py --config configs/evaluation.yaml --append \
           --model ${BEST}_ft=models/encoders/$BEST-ft ;;
    g) sed -i 's/^  enabled: false                 # set true after/  enabled: true                  # set true after/' "$CFG"
       for SPL in val code_disjoint; do
         run g_$SPL python scripts/train.py --config "$CFG" --tag kaggle_metrics --splits $SPL --no-final --seeds 13,14,15
       done
       run g_test python scripts/train.py --config "$CFG" --tag kaggle_metrics --splits "" --final-test --no-final --seeds 13,14,15
       run g_refit python scripts/train.py --config "$CFG" --tag kaggle_refit --splits "" ;;   # models/final.pkl
    h) run h_render cp reports/kaggle_metrics.json reports/metrics.json
       run h_eval python scripts/evaluate.py --render
       for T in val code_disjoint final_test; do
         run h_err_$T python scripts/error_analysis.py --tag $T --run reports/private/${T}_run.pkl
       done
       run h_bench python scripts/benchmark_speed.py --config "$CFG"
       run h_checks python scripts/run_checks.py
       run h_tests bash -c "python -m pytest -q --junitxml=reports/pytest.xml || true"   # hard-case failures are reported, not fatal
       run h_docs bash -c "python scripts/make_experiment_log.py && python scripts/make_summary.py && python scripts/make_pdf.py" ;;
    *) echo "unknown stage $st"; exit 2 ;;
  esac
done
