#!/bin/bash
# Espera a que terminen todas las colas de entrenamiento y luego: XAI → tablas → carpeta de entrega.
cd "$(dirname "$0")"
while pgrep -f "run_queue.sh" > /dev/null; do sleep 60; done
source env.sh
echo "[finalize] XAI $(date)"; python xai_quant.py --model rootcvt 2>&1 | grep -E "XAI\]|Traceback|Error"
echo "[finalize] tablas $(date)"; python summarize.py > logs/summarize.log 2>&1
python organize_deliverables.py > logs/organize.log 2>&1
echo "[finalize] listo $(date)"; grep -c "⏳" ENTREGA_REVISORES/README.md
