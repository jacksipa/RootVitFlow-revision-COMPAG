#!/bin/bash
# RootCVT (model97) en TF 2.11.1 (versión del cluster original), CPU: plant_cv + holdout, 3 workers.
cd "$(dirname "$0")"
PY=$HOME/miniconda3/envs/rootcvt211/bin/python
export CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=2 PYTHONHASHSEED=0
export OMP_NUM_THREADS=6 TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2
for w in 1 2 3; do
  ( $PY run_exp.py --model rootcvt --scheme plant_cv --tag _tf211; $PY run_exp.py --model rootcvt --scheme holdout --tag _tf211 ) > logs/tf211_w$w.log 2>&1 &
  sleep 20
done
wait
date > logs/TF211_DONE
