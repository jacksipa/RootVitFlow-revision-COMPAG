# RootViT-Flow — experimentación para la revisión COMPAG-D-26-05203R1

Repositorio privado con el código, los resultados y la auditoría de la experimentación realizada para
responder a los revisores (Computers and Electronics in Agriculture).

- **`ENTREGA_REVISORES/`** — resultados organizados por comentario de revisor. Empezar por `ENTREGA_REVISORES/README.md`
  (índice: qué archivo responde a qué comentario y en qué sección del manuscrito va).
- **`results/`** — resultados completos por corrida (predicciones, historiales, métricas), tablas y figuras.
- **`splits/assignments.csv`** — asignación imagen → fold → rol para todos los esquemas (archivo de folds del manuscrito).
- Código: `make_splits.py`, `run_exp.py`, `common.py` (importa `Prueba1/root_cvt_model97.py` sin modificarlo),
  `morpho.py`, `xai_quant.py`, `seg_benchmark.py`, `summarize.py`, `organize_deliverables.py`, `run_all.sh`.

Los pesos entrenados y las imágenes del dataset no se incluyen (quedan en la máquina del semillero).
El repositorio se actualiza automáticamente mientras corre la experimentación.
