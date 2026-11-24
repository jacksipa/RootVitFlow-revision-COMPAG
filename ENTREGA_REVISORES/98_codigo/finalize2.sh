#!/bin/bash
# Cierre: espera RootCVT TF 2.11 (6 folds plant_cv rep0 + 3 holdout) y los XAI en curso,
# luego XAI de rootcvt_tf211 → tablas → ENTREGA_REVISORES → ZIP → push. Marca logs/ALL_DONE.
cd "$(dirname "$0")"
n_tf211() { ls results/plant_cv/rootcvt_tf211/rep0_*/info.json results/holdout/rootcvt_tf211/*/info.json 2>/dev/null | wc -l; }
until [ "$(n_tf211)" -ge 9 ] || ! pgrep -f "rootcvt211/bin/python run_exp.py" > /dev/null; do sleep 60; done
echo "[$(date '+%T')] TF 2.11 listo: $(n_tf211)/9"
while pgrep -f "python xai_quant.py" > /dev/null; do sleep 30; done
source env.sh
echo "[$(date '+%T')] XAI rootcvt_tf211"
python xai_quant.py --model rootcvt_tf211 --arch rootcvt > logs/xai_tf211.log 2>&1 || echo "ERROR XAI tf211"
echo "[$(date '+%T')] tablas"
python summarize.py > logs/summarize.log 2>&1 || echo "ERROR summarize"
python organize_deliverables.py > logs/organize.log 2>&1 || echo "ERROR organize"
rm -rf results/plant_cv/rootcvt_tf211/rep1_fold* results/holdout/rootcvt_tf211/rep2_fold0   # carpetas reservadas (no se corrieron en TF 2.11)
rm -rf ENTREGA_REVISORES/99_resultados_completos ENTREGA_REVISORES/98_codigo
cp -r results ENTREGA_REVISORES/99_resultados_completos
find ENTREGA_REVISORES/99_resultados_completos \( -name "*.h5" -o -name "*.pt" -o -name "running.lock" \) -delete
mkdir -p ENTREGA_REVISORES/98_codigo && cp *.py *.sh ENTREGA_REVISORES/98_codigo/
(cd .. && rm -f Prueba3/ENTREGA_REVISORES.zip && zip -qr Prueba3/ENTREGA_REVISORES.zip Prueba3/ENTREGA_REVISORES)
date > logs/ALL_DONE
echo "[$(date '+%T')] listo"
