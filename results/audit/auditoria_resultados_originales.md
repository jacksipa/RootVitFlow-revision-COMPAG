# Auditoría de los resultados originales

## 1. El 99.00 ± 1.22% de '5-fold CV' (Prueba2/cross.py)
Carga UN modelo ya entrenado (pesos fijos `saved-model-192-1.0000.hdf5`) y lo evalúa en 5 folds estratificados de las 200 imágenes, que incluyen las imágenes con las que se entrenó. No hay re-entrenamiento por fold: no es una validación cruzada y no estima generalización.

## 2. El 100% en test (Prueba1, logs_2/RootCVT_RiegoSequia_2026-02-20_03-52-06)
- La validación usada para elegir el checkpoint era el MISMO conjunto de test (40 imágenes).
- Arquitectura del checkpoint: 7 bloques CNN (5,450,700 parámetros), LR constante 3e-4 (sin ReduceLROnPlateau).
- Es el máximo de 6 corridas con la misma configuración el 20/02/2026:

| log                               |   epochs |   mejor_val_acc(=test) |   train_acc_final |   reducciones_LR |
|:----------------------------------|---------:|-----------------------:|------------------:|-----------------:|
| pipeline_gpu2_20260220_033634.log |      150 |                  0.725 |            0.675  |                0 |
| pipeline_gpu2_20260220_035833.log |      150 |                  0.9   |            0.9187 |                0 |
| pipeline_gpu2_20260220_041308.log |      150 |                  0.8   |            0.95   |                0 |
| pipeline_gpu2_20260220_042443.log |      150 |                  0.625 |            0.7437 |                4 |
| pipeline_gpu4_20260220_034335.log |      150 |                  0.875 |            0.925  |                0 |
| pipeline_gpu4_20260220_035507.log |      150 |                  0.7   |            0.9187 |                0 |
| pipeline_gpu4_20260220_043035.log |      150 |                  0.775 |            0.7    |                4 |
| pipeline_gpu5_20260220_035200.log |      150 |                  1     |            0.925  |                0 |

Corridas con LR constante: media de la mejor accuracy en test = 83.3 ± 11.4% (n=6).

## 3. Datos
- 200 máscaras de 16 plantas (8 riego rb1–rb10, 8 sequía sb11–sb20), una imagen POR DÍA (16/07–30/07/2025), no cada ~6 días.
- La partición original fue aleatoria por imagen: la misma planta (días consecutivos) en train y test.
