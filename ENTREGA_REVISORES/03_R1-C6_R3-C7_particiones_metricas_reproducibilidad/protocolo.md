# Protocolo experimental de la revisión (Prueba3)

**Datos**: 200 máscaras binarias RootPainter (`allbin`), 224×224×1, uint8 0–255, cargadas exactamente como
Prueba1/data_roots_transf.py (verificado: idénticas bit a bit a Balanced_partition_2). Etiqueta por prefijo r/s.
Grupos = planta (16). Fechas: 16/07–30/07/2025, una imagen diaria por planta.

**Particiones** (`splits/assignments.csv`):
- Holdout independiente: rb3, rb8, sb13, sb15 (60 imágenes), elegidas al azar (semilla 2026) antes de entrenar.
- Desarrollo (12 plantas, 140 imágenes):
  - `plant_cv`: leave-one-pair-out (1 planta riego + 1 sequía en test), 6 folds × 2 repeticiones;
    validación = otro par de plantas; train = 8 plantas.
  - `image_cv`: CV estratificada por imagen (protocolo original), 6 folds × 2 repeticiones; validación = 25 imágenes.
- `holdout`: train = 10 plantas de desarrollo, validación = 2 plantas, test = holdout; 3 repeticiones.

**Entrenamiento (idéntico para todos los modelos)**: 150 épocas, batch 16, Adam lr 3e-4 constante,
weight_decay 1e-3, clipnorm 1.0, CE con label smoothing 0.08, class_weight {riego: 2, sequía: 1},
data augmentation de root_cvt_model97 (flip H+V, rotación ±0.2·2π, zoom ±20%, contraste ±10%),
cabeza Dense128-BN-ReLU-Dropout0.5-softmax. Checkpoint = mejor val_accuracy en la validación
(nunca en test). CNN baselines: ImageNet, fine-tuning completo, máscara replicada a 3 canales + preprocess_input.

**Semillas**: seed = 1234 + 100·rep + fold (tf.keras.utils.set_random_seed). Holdout: 2026. Particiones: 100+rep / 200+rep.

**Métricas**: accuracy, balanced accuracy, precisión/recall/F1 macro, AUC, recall por clase, accuracy por planta
(voto por media de probabilidad). IC95% por bootstrap por conglomerados (plantas), 2000 réplicas.

**Software**: TensorFlow 2.21 + tf_keras 2.21 (Keras 2), RTX 5090. Referencia de control: TF 2.11.1 (versión original).
