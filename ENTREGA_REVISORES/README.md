# Entrega de experimentación — Respuesta a revisores COMPAG-D-26-05203R1
Cada carpeta corresponde a uno o más comentarios. ✅ = listo, ⏳ = pendiente (la experimentación sigue corriendo o falta insumo).
Protocolo completo: `03_.../protocolo.md`. Código: `Prueba3/`.

## 01_R1-C4_R3-C1_fuga_de_datos
**Responde a:** R1 Comentario 4; R3 Comentario 1  
**Va en el manuscrito:** Sec. 3.6 (protocolo de validación), Sec. 4.2.2 (resultados), Sec. 5.5 (limitaciones)
- ✅ `T1_leakage.md` — Tabla: CV por imagen vs CV agrupada por planta (mismo modelo y datos)
- ✅ `T1_leakage.csv` — Igual que arriba, CSV
- ✅ `F_temporal_accuracy.png` — Figura: accuracy por fecha de adquisición
- ✅ `T4_temporal_rootcvt.csv` — Tabla: accuracy y probabilidad media por fecha
- ✅ `auditoria_resultados_originales.md` — Auditoría de las corridas originales (99% y 100%)

## 02_R1-C5_R3-C1_test_independiente
**Responde a:** R1 Comentario 5; R3 Comentario 1  
**Va en el manuscrito:** Sec. 3.6 (holdout), Sec. 4.2.2 (Tabla de resultados principal), Resumen
- ⏳ `T3_holdout.md` — Tabla: test independiente (4 plantas nunca vistas)
- ⏳ `T3_holdout.csv` — Igual que arriba, CSV
- ✅ `holdout_plants.json` — Plantas del holdout y semilla

## 03_R1-C6_R3-C7_particiones_metricas_reproducibilidad
**Responde a:** R1 Comentario 6; R3 Comentario 7  
**Va en el manuscrito:** Sec. 3.6 (completar [SEMILLA]), Sec. 7 (disponibilidad de código: archivos de folds)
- ✅ `assignments.csv` — Asignación imagen→fold→rol para TODOS los esquemas
- ✅ `holdout_plants.json` — Plantas del holdout
- ✅ `protocolo.md` — Protocolo, semillas, versiones y métricas

## 04_R3-C2_baselines_y_ablacion
**Responde a:** R3 Comentario 2  
**Va en el manuscrito:** Sec. 3.6 (baselines equivalentes), Sec. 4.2.1 (Tabla 4), nueva sub-sección de ablación
- ✅ `T2_models_plant_cv.md` — Tabla: todos los modelos, CV agrupada por planta
- ✅ `T2_models_plant_cv.csv` — Igual que arriba, CSV
- ✅ `T2b_paired_tests_vs_rootcvt.csv` — Wilcoxon pareado por fold vs RootCVT
- ✅ `F_models_plant_cv.png` — Figura: accuracy por fold de cada modelo

## 05_R1-C8_R3-C5_morfometria
**Responde a:** R1 Comentario 8; R3 Comentario 5  
**Va en el manuscrito:** Tabla 5 (ML clásico, sin [COMPLETAR]), Tabla 7 (estadística riego vs sequía), Sec. 4.3
- ✅ `T5_classical_ml.md` — Tabla 5 completa: ML clásico con las mismas particiones
- ✅ `T5_classical_ml.csv` — Igual que arriba, CSV
- ✅ `T5b_morpho_stats.md` — Tabla 7: riego vs sequía (Mann-Whitney por planta + modelo mixto)
- ✅ `features_per_image.csv` — Rasgos por imagen (base fenotípica)
- ✅ `temporal_means.csv` — Medias por fecha y tratamiento
- ✅ `F_morpho_temporal.png` — Figura: evolución temporal de rasgos por tratamiento

## 06_R1-C7_R3-C4_XAI_cuantitativo
**Responde a:** R1 Comentario 7; R3 Comentario 4  
**Va en el manuscrito:** Sec. 3.5 (métodos XAI), Sec. 4.4 (resultados), Sec. 5.4 (discusión)
- ⏳ `T6_xai.md` — Tabla: concentración en raíz, deletion/insertion, cobertura de tokens
- ⏳ `T6b_randomization.md` — Tabla: sanity check de randomización en cascada
- ⏳ `xai_per_image_rootcvt.csv` — Métricas XAI por imagen (todas las imágenes de test)
- ⏳ `xai_tests_rootcvt.json` — Pruebas de Wilcoxon atribución vs aleatorio

## 07_R1-C3_R3-C3_segmentacion
**Responde a:** R1 Comentario 3; R3 Comentario 3  
**Va en el manuscrito:** Sec. 3.2, Sec. 4.1
- ⏳ Pendiente: se necesitan los códigos/arquitecturas de segmentación (U-Net, SwinUNETR, SAM).
