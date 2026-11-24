#!/bin/bash
# =====================================================================================
# run_all.sh — Plan maestro de la experimentación de revisión (COMPAG-D-26-05203R1)
#
# Idempotente y reanudable: se puede lanzar las veces que haga falta.
#   nohup setsid bash run_all.sh > logs/run_all.log 2>&1 &
#
# Bucle watchdog:
#   1. Clasificación: mantiene 5 colas corriendo hasta completar todas las corridas
#      (run_exp.py omite corridas terminadas y usa bloqueos por corrida).
#   2. Segmentación: lanza U-Net / SwinUNETR / SAM si no tienen summary.json y no están corriendo.
#   3. Limpia bloqueos huérfanos (proceso muerto) para que se reintenten.
#   4. Al terminar todo: XAI cuantitativo → tablas/figuras → carpeta ENTREGA_REVISORES → ZIP.
#   Estado en logs/STATUS.txt; marca final logs/ALL_DONE.
# =====================================================================================
cd "$(dirname "$0")"
mkdir -p logs
PYSEG=$HOME/miniconda3/envs/rootseg/bin/python
QUEUES=(
  "rootcvt:plant_cv rootcvt:holdout rootcvt_7b:plant_cv rootcvt_7b:holdout rootcvt_notransf:plant_cv rootcvt_notransf:holdout"
  "rootcvt:image_cv rootcvt_noaug:image_cv rootcvt_noaug:plant_cv rootcvt_noaug:holdout rootcvt_p4:plant_cv rootcvt_p4:holdout"
  "mobilenetv2:plant_cv inceptionv3:plant_cv resnet50:plant_cv densenet201:plant_cv mobilenetv2:holdout"
  "densenet201:holdout resnet50:holdout inceptionv3:holdout densenet201:plant_cv resnet50:plant_cv inceptionv3:plant_cv"
  "rootcvt_notransf:holdout rootcvt_7b:holdout rootcvt_notransf:plant_cv rootcvt_7b:plant_cv rootcvt_noaug:plant_cv"
)
ALLJOBS=$(printf "%s " "${QUEUES[@]}")
NEED_plant_cv=12; NEED_image_cv=12; NEED_holdout=3

log() { echo "[$(date '+%F %T')] $*"; }

missing_runs() {   # nº de corridas de clasificación que faltan (excluye las marcadas en skip/)
  local tot=0
  for j in $(echo $ALLJOBS | tr ' ' '\n' | sort -u); do
    m=${j%%:*}; s=${j##*:}
    [ -f "skip/${m}_${s}" ] && continue
    n=$(ls results/$s/$m/*/info.json 2>/dev/null | wc -l)
    need=NEED_$s; tot=$((tot + ${!need} - n))
  done
  echo $tot
}

clean_stale_locks() {   # bloqueos sin proceso vivo para ese modelo/esquema
  for lk in results/*/*/*/running.lock; do
    [ -f "$lk" ] || continue
    d=$(dirname "$lk"); s=$(echo "$d" | cut -d/ -f2); m=$(echo "$d" | cut -d/ -f3)
    case "$s" in plant_cv|image_cv|holdout) ;; *) continue ;; esac   # sólo corridas de run_exp.py
    case "$m" in *_tf211) continue ;; esac                           # corridas TF 2.11 (run_tf211.sh) se gestionan aparte
    if [ -f "$d/info.json" ]; then rm -f "$lk"; continue; fi          # corrida terminada: conservar pesos
    if ! pgrep -f "run_exp.py --model $m --scheme $s" > /dev/null; then
      log "bloqueo huérfano eliminado: $d"; rm -f "$lk" "$d/best.weights.h5" "$d/history.csv"
    fi
  done
}

seg_job() {
  m=$1; mins=$2; bs=$3
  [ -f results/segmentation/$m/summary.json ] && return
  pgrep -f "seg_benchmark.py --model $m" > /dev/null && return
  n=$(cat logs/seg_${m}_tries 2>/dev/null || echo 0)
  [ "$n" -ge 3 ] && return
  echo $((n+1)) > logs/seg_${m}_tries
  log "lanzando segmentación $m (intento $((n+1)))"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PYSEG seg_benchmark.py --model $m --max_minutes $mins $bs \
      > logs/seg_${m}.log 2>&1 &
}

# --------------------------------------------------------------------- bucle principal
log "run_all.sh iniciado (PID $$)"
while true; do
  clean_stale_locks
  MISSING=$(missing_runs)
  NQ=$(pgrep -fc "^bash run_queue.sh")
  # Clasificación: si faltan corridas y hay menos de 4 colas vivas (límite por memoria GPU), lanzar
  if [ "$MISSING" -gt 0 ] && [ "$NQ" -lt 4 ]; then
    for i in "${!QUEUES[@]}"; do
      [ "$(pgrep -fc "^bash run_queue.sh")" -ge 4 ] && break
      log "lanzando cola $i"
      nohup bash run_queue.sh ${QUEUES[$i]} > logs/queue_w$i.log 2>&1 &
      sleep 5
    done
  fi
  # Segmentación (SAM espera a que U-Net termine por memoria)
  seg_job unet 35 ""
  seg_job swinunetr 50 ""
  [ -f results/segmentation/unet/summary.json ] && [ -f results/segmentation/swinunetr/summary.json ] && seg_job sam 40 "--bs 2"

  SEGDONE=$(ls results/segmentation/*/summary.json 2>/dev/null | wc -l)
  SEGRUN=$(pgrep -fc "seg_benchmark.py")
  {
    echo "Actualizado: $(date)"
    echo "Clasificación: faltan $MISSING corridas | colas vivas: $(pgrep -fc "^bash run_queue.sh") | procesos: $(pgrep -fc run_exp.py)"
    echo "Segmentación: terminados $SEGDONE/3 | corriendo $SEGRUN"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  } > logs/STATUS.txt

  if [ "$MISSING" -le 0 ] && [ "$(pgrep -fc run_exp.py)" -eq 0 ] && [ "$SEGRUN" -eq 0 ]; then
    break
  fi
  # si ya no se puede avanzar (colas terminaron pero faltan corridas por error repetido) → finalizar igual
  if [ "$MISSING" -gt 0 ] && [ "$(pgrep -fc run_exp.py)" -eq 0 ] && [ "$(cat logs/relaunch_count 2>/dev/null || echo 0)" -ge 4 ]; then
    log "AVISO: quedan $MISSING corridas sin completar tras 4 relanzamientos; se finaliza con lo disponible"
    break
  fi
  if [ "$MISSING" -gt 0 ] && [ "$(pgrep -fc run_exp.py)" -eq 0 ]; then
    echo $(( $(cat logs/relaunch_count 2>/dev/null || echo 0) + 1 )) > logs/relaunch_count
  fi
  sleep 120
done

# --------------------------------------------------------------------- finalización
pkill -f finalize.sh 2>/dev/null
source env.sh
log "XAI cuantitativo"
python xai_quant.py --model rootcvt > logs/xai.log 2>&1 || log "ERROR en XAI (ver logs/xai.log)"
log "tablas y figuras"
python summarize.py > logs/summarize.log 2>&1 || log "ERROR en summarize (ver logs/summarize.log)"
python organize_deliverables.py > logs/organize.log 2>&1 || log "ERROR en organize (ver logs/organize.log)"
log "empaquetando"
cp -r results ENTREGA_REVISORES/99_resultados_completos 2>/dev/null
find ENTREGA_REVISORES/99_resultados_completos -name "*.h5" -delete -o -name "*.pt" -delete -o -name "running.lock" -delete
mkdir -p ENTREGA_REVISORES/98_codigo
cp *.py *.sh ENTREGA_REVISORES/98_codigo/ 2>/dev/null
rm -f ../Prueba3_ENTREGA_REVISORES.zip
(cd .. && zip -qr Prueba3/ENTREGA_REVISORES.zip Prueba3/ENTREGA_REVISORES)
log "TODO TERMINADO"
date > logs/ALL_DONE
