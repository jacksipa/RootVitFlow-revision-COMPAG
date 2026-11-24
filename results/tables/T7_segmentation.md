Test = 60 imágenes de 4 plantas nunca vistas (rb3, rb8, sb13, sb15), resolución completa; media ± DE por imagen.

| Modelo                         | Dice       | IoU        | Precisión   | Recall     | clDice     | Skeleton-F1 (tol 2px)   |   Predicciones vacías |   Parche |   Iteraciones |   Mejor iter (val) |   Minutos |
|:-------------------------------|:-----------|:-----------|:------------|:-----------|:-----------|:------------------------|----------------------:|---------:|--------------:|-------------------:|----------:|
| U-Net (MONAI)                  | 70.4 ± 3.9 | 54.4 ± 4.5 | 65.8 ± 6.7  | 76.2 ± 4.4 | 82.9 ± 5.0 | 61.7 ± 6.5              |                     0 |      512 |          8000 |               6000 |       7   |
| SwinUNETR (MONAI)              | 69.0 ± 4.3 | 52.9 ± 4.9 | 64.8 ± 6.8  | 74.5 ± 4.8 | 81.6 ± 5.4 | 60.5 ± 6.4              |                     0 |      512 |          8000 |               6750 |      27.1 |
| SAM ViT-B (decoder fine-tuned) | 65.8 ± 3.3 | 49.1 ± 3.6 | 65.4 ± 6.1  | 66.9 ± 5.7 | 77.2 ± 5.1 | 56.3 ± 5.7              |                     0 |     1024 |          8000 |               6000 |      37.8 |