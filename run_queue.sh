#!/bin/bash
# Ejecuta en secuencia una cola de experimentos "modelo:esquema" (reanudable).
# Uso: bash run_queue.sh rootcvt:plant_cv rootcvt:holdout ...
cd "$(dirname "$0")"
source env.sh
for job in "$@"; do
  m="${job%%:*}"; s="${job##*:}"
  echo "=============== $m / $s  $(date) ==============="
  python run_exp.py --model "$m" --scheme "$s" 2>&1 | grep -vE "^W0000|^I0000|absl|while_loop|oneDNN|layout failed|Downloading|ETA"
done
echo "=============== COLA TERMINADA $(date) ==============="
