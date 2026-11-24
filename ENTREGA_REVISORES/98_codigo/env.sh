#!/bin/bash
# Entorno para correr la experimentación en esta máquina (RTX 5090, TF 2.21 + tf_keras).
# Uso:  source env.sh
CONDA_ENV="$HOME/miniconda3/envs/rootcvt"
export PATH="$CONDA_ENV/bin:$PATH"
SP=$("$CONDA_ENV/bin/python" -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(ls -d "$SP"/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH}"
# Keras 2 (tf_keras) para ejecutar root_cvt_model97.py tal como fue escrito (TF 2.11 / Keras 2)
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=2
# Caché del JIT de PTX (la RTX 5090 / sm_120 no tiene kernels precompilados en TF)
export CUDA_CACHE_MAXSIZE=4294967296
export PYTHONHASHSEED=0
