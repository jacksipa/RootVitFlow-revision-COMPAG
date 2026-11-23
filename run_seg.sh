#!/bin/bash
# Lanza U-Net, SwinUNETR y SAM en paralelo (entorno rootseg, PyTorch cu128 para RTX 5090).
cd "$(dirname "$0")"
PY=$HOME/miniconda3/envs/rootseg/bin/python
until $PY -c "import torch,monai,transformers; assert torch.cuda.is_available()" 2>/dev/null; do sleep 20; done
mkdir -p logs
$PY seg_benchmark.py --model unet      --max_minutes 35 > logs/seg_unet.log 2>&1 &
$PY seg_benchmark.py --model swinunetr --max_minutes 50 > logs/seg_swinunetr.log 2>&1 &
$PY seg_benchmark.py --model sam       --max_minutes 40 > logs/seg_sam.log 2>&1 &
wait
echo "segmentación terminada $(date)"
