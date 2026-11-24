"""
organize_deliverables.py (Prueba3)

Copia los resultados disponibles a ENTREGA_REVISORES/, organizados por comentario de revisor,
y regenera ENTREGA_REVISORES/README.md con el índice (qué archivo responde a qué comentario,
en qué sección del manuscrito va y si ya está listo). Se puede correr las veces que haga falta.
"""
import os
import re
import glob
import shutil
import pandas as pd

import config_transf as cfg

R = cfg.RESULTS_DIR
T = os.path.join(R, "tables")
F = os.path.join(R, "figures")
DEST = os.path.join(cfg.PROJECT_ROOT, "ENTREGA_REVISORES")

# carpeta → (comentarios, sección del manuscrito, [(archivo origen, descripción)])
PLAN = {
    "01_R1-C4_R3-C1_fuga_de_datos": (
        "R1 Comentario 4; R3 Comentario 1",
        "Sec. 3.6 (protocolo de validación), Sec. 4.2.2 (resultados), Sec. 5.5 (limitaciones)",
        [(f"{T}/T1_leakage.md", "Tabla: CV por imagen vs CV agrupada por planta (mismo modelo y datos)"),
         (f"{T}/T1_leakage.csv", "Igual que arriba, CSV"),
         (f"{F}/F_temporal_accuracy.png", "Figura: accuracy por fecha de adquisición"),
         (f"{T}/T4_temporal_rootcvt.csv", "Tabla: accuracy y probabilidad media por fecha"),
         (f"{R}/audit/auditoria_resultados_originales.md", "Auditoría de las corridas originales (99% y 100%)")]),
    "02_R1-C5_R3-C1_test_independiente": (
        "R1 Comentario 5; R3 Comentario 1",
        "Sec. 3.6 (holdout), Sec. 4.2.2 (Tabla de resultados principal), Resumen",
        [(f"{T}/T3_holdout.md", "Tabla: test independiente (4 plantas nunca vistas)"),
         (f"{T}/T3_holdout.csv", "Igual que arriba, CSV"),
         (f"{cfg.SPLITS_DIR}/holdout_plants.json", "Plantas del holdout y semilla")]),
    "03_R1-C6_R3-C7_particiones_metricas_reproducibilidad": (
        "R1 Comentario 6; R3 Comentario 7",
        "Sec. 3.6 (completar [SEMILLA]), Sec. 7 (disponibilidad de código: archivos de folds)",
        [(f"{cfg.SPLITS_DIR}/assignments.csv", "Asignación imagen→fold→rol para TODOS los esquemas"),
         (f"{cfg.SPLITS_DIR}/holdout_plants.json", "Plantas del holdout"),
         (f"{R}/audit/protocolo.md", "Protocolo, semillas, versiones y métricas")]),
    "04_R3-C2_baselines_y_ablacion": (
        "R3 Comentario 2",
        "Sec. 3.6 (baselines equivalentes), Sec. 4.2.1 (Tabla 4), nueva sub-sección de ablación",
        [(f"{T}/T2_models_plant_cv.md", "Tabla: todos los modelos, CV agrupada por planta"),
         (f"{T}/T2_models_plant_cv.csv", "Igual que arriba, CSV"),
         (f"{T}/T2b_paired_tests_vs_rootcvt.csv", "Wilcoxon pareado por fold vs RootCVT"),
         (f"{F}/F_models_plant_cv.png", "Figura: accuracy por fold de cada modelo")]),
    "05_R1-C8_R3-C5_morfometria": (
        "R1 Comentario 8; R3 Comentario 5",
        "Tabla 5 (ML clásico, sin [COMPLETAR]), Tabla 7 (estadística riego vs sequía), Sec. 4.3",
        [(f"{T}/T5_classical_ml.md", "Tabla 5 completa: ML clásico con las mismas particiones"),
         (f"{T}/T5_classical_ml.csv", "Igual que arriba, CSV"),
         (f"{T}/T5b_morpho_stats.md", "Tabla 7: riego vs sequía (Mann-Whitney por planta + modelo mixto)"),
         (f"{R}/morpho/features_per_image.csv", "Rasgos por imagen (base fenotípica)"),
         (f"{R}/morpho/temporal_means.csv", "Medias por fecha y tratamiento"),
         (f"{F}/F_morpho_temporal.png", "Figura: evolución temporal de rasgos por tratamiento")]),
    "06_R1-C7_R3-C4_XAI_cuantitativo": (
        "R1 Comentario 7; R3 Comentario 4",
        "Sec. 3.5 (métodos XAI), Sec. 4.4 (resultados), Sec. 5.4 (discusión)",
        [(f"{T}/T6_xai.md", "Tabla: concentración en raíz, deletion/insertion, cobertura de tokens"),
         (f"{T}/T6b_randomization.md", "Tabla: sanity check de randomización en cascada"),
         (f"{R}/xai/xai_per_image_rootcvt.csv", "Métricas XAI por imagen — RootCVT TF 2.21"),
         (f"{R}/xai/xai_per_image_rootcvt_noaug.csv", "Métricas XAI por imagen — RootCVT sin augmentation"),
         (f"{R}/xai/xai_per_image_rootcvt_tf211.csv", "Métricas XAI por imagen — RootCVT TF 2.11 (original)"),
         (f"{R}/xai/xai_tests_rootcvt.json", "Pruebas de Wilcoxon atribución vs aleatorio")]),
    "07_R1-C3_R3-C3_segmentacion": (
        "R1 Comentario 3; R3 Comentario 3; R1 Comentario 6 (particiones consistentes)",
        "Sec. 3.2 (entrenamiento/test de segmentación), Sec. 4.1 (Tabla de segmentación), Fig. 8–9",
        [(f"{T}/T7_segmentation.md", "Tabla: U-Net / SwinUNETR / SAM, mismo test (4 plantas), Dice, IoU, clDice, skeleton-F1"),
         (f"{T}/T7_segmentation.csv", "Igual que arriba, CSV"),
         (f"{R}/segmentation/unet/test_per_image.csv", "U-Net: métricas por imagen de test"),
         (f"{R}/segmentation/swinunetr/test_per_image.csv", "SwinUNETR: métricas por imagen de test"),
         (f"{R}/segmentation/sam/test_per_image.csv", "SAM: métricas por imagen de test"),
         (f"{R}/audit/auditoria_segmentacion.md", "Problemas encontrados en los notebooks originales")]),
}


def write_seg_table():
    import json as _j
    rows = []
    for m in ["unet", "swinunetr", "sam"]:
        p = os.path.join(R, "segmentation", m, "summary.json")
        if os.path.exists(p):
            s = _j.load(open(p))
            f = lambda k: f"{100*s[k+'_mean']:.1f} ± {100*s[k+'_sd']:.1f}"
            rows.append({"Modelo": {"unet": "U-Net (MONAI)", "swinunetr": "SwinUNETR (MONAI)", "sam": "SAM ViT-B (decoder fine-tuned)"}[m],
                         "Dice": f("dice"), "IoU": f("iou"), "Precisión": f("precision"), "Recall": f("recall"),
                         "clDice": f("cldice"), "Skeleton-F1 (tol 2px)": f("skel_f1"),
                         "Predicciones vacías": int(round(s["empty_pred_mean"] * 60)),
                         "Parche": s["patch"], "Iteraciones": s["iters_done"], "Mejor iter (val)": s["best_iter"],
                         "Minutos": s["train_minutes"]})
    if rows:
        D = pd.DataFrame(rows)
        os.makedirs(T, exist_ok=True)
        D.to_csv(os.path.join(T, "T7_segmentation.csv"), index=False)
        with open(os.path.join(T, "T7_segmentation.md"), "w") as fo:
            fo.write("Test = 60 imágenes de 4 plantas nunca vistas (rb3, rb8, sb13, sb15), resolución completa; "
                     "media ± DE por imagen.\n\n" + D.to_markdown(index=False))
    os.makedirs(os.path.join(R, "audit"), exist_ok=True)
    with open(os.path.join(R, "audit", "auditoria_segmentacion.md"), "w") as fo:
        fo.write("""# Problemas en los notebooks de segmentación originales (unet/Swin/SAM_Segmentor.ipynb)

1. **Validación = test**: el mismo conjunto `valbin` (34 imágenes) se usó para elegir el mejor checkpoint
   (máximo Dice por época, ReduceLROnPlateau sobre ese Dice) y para reportar el resultado.
2. **Resolución**: U-Net y SwinUNETR redimensionaban la imagen COMPLETA (3072×4096) a 512×512 (factor ~6–8);
   SAM a 256×256 (factor ~12–16). Raíces de 1–3 px quedan sub-píxel → el bajo Dice refleja la configuración,
   no la arquitectura (coincide con el Revisor 3).
3. **SAM**: (a) en el bucle de validación se usaba `pred_logits` del último batch de ENTRENAMIENTO
   (`pred_probs = torch.sigmoid(pred_logits)`), no la salida del modelo en validación → el Dice de validación de SAM
   reportado no es válido; (b) el prompt (bounding box) se calculaba a partir de la máscara verdadera (información
   del GT en inferencia).
4. **Splits**: 166/34 por imagen (misma planta en train y val), distinto de la clasificación.
5. **Ground truth**: las máscaras de referencia son las de RootPainter (anotación correctiva). Evaluar RootPainter
   contra sus propias máscaras es circular: el 96.19% de Dice de RootPainter no es una comparación válida y debe
   presentarse como acuerdo con la anotación, no como desempeño comparativo.

**Re-evaluación (seg_benchmark.py)**: mismas particiones que la clasificación (test = 4 plantas nunca vistas),
validación separada (2 plantas), entrenamiento por parches a resolución completa con muestreo positivo/negativo,
inferencia por ventana deslizante en la imagen completa, SAM con caja = parche completo (sin GT),
y métricas tolerantes a la topología (clDice, skeleton-F1 con tolerancia 2 px).
""")


def write_audit():
    """Auditoría estática de las corridas originales (a partir de los logs de Prueba1/Prueba2)."""
    os.makedirs(os.path.join(R, "audit"), exist_ok=True)
    p1 = cfg.PRUEBA1_DIR
    rows = []
    for f in sorted(glob.glob(os.path.join(p1, "pipeline_gpu*_20260220_*.log"))):
        txt = open(f, errors="ignore").read()
        n = len(re.findall(r"^10/10 ", txt, re.M))
        if n < 150:
            continue
        best = re.findall(r"improved from [0-9.inf-]+ to ([0-9.]+)", txt)
        red = len(re.findall(r"reducing learning rate", txt, re.I))
        last = re.findall(r"^10/10.* accuracy: ([0-9.]+)", txt, re.M)
        rows.append({"log": os.path.basename(f), "epochs": n,
                     "mejor_val_acc(=test)": float(best[-1]) if best else None,
                     "train_acc_final": float(last[-1]) if last else None,
                     "reducciones_LR": red})
    D = pd.DataFrame(rows)
    const = D[D.reducciones_LR == 0]
    with open(os.path.join(R, "audit", "auditoria_resultados_originales.md"), "w") as fo:
        fo.write("# Auditoría de los resultados originales\n\n")
        fo.write("## 1. El 99.00 ± 1.22% de '5-fold CV' (Prueba2/cross.py)\n")
        fo.write("Carga UN modelo ya entrenado (pesos fijos `saved-model-192-1.0000.hdf5`) y lo evalúa en 5 folds "
                 "estratificados de las 200 imágenes, que incluyen las imágenes con las que se entrenó. "
                 "No hay re-entrenamiento por fold: no es una validación cruzada y no estima generalización.\n\n")
        fo.write("## 2. El 100% en test (Prueba1, logs_2/RootCVT_RiegoSequia_2026-02-20_03-52-06)\n")
        fo.write("- La validación usada para elegir el checkpoint era el MISMO conjunto de test (40 imágenes).\n")
        fo.write("- Arquitectura del checkpoint: 7 bloques CNN (5,450,700 parámetros), LR constante 3e-4 "
                 "(sin ReduceLROnPlateau).\n")
        fo.write(f"- Es el máximo de {len(const)} corridas con la misma configuración el 20/02/2026:\n\n")
        fo.write(D.to_markdown(index=False))
        fo.write(f"\n\nCorridas con LR constante: media de la mejor accuracy en test = "
                 f"{100*const['mejor_val_acc(=test)'].mean():.1f} ± {100*const['mejor_val_acc(=test)'].std():.1f}% "
                 f"(n={len(const)}).\n\n")
        fo.write("## 3. Datos\n- 200 máscaras de 16 plantas (8 riego rb1–rb10, 8 sequía sb11–sb20), "
                 "una imagen POR DÍA (16/07–30/07/2025), no cada ~6 días.\n"
                 "- La partición original fue aleatoria por imagen: la misma planta (días consecutivos) en train y test.\n")


def write_protocol():
    os.makedirs(os.path.join(R, "audit"), exist_ok=True)
    with open(os.path.join(R, "audit", "protocolo.md"), "w") as fo:
        fo.write("""# Protocolo experimental de la revisión (Prueba3)

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
""")


def main():
    write_audit()
    write_protocol()
    write_seg_table()
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    lines = ["# Entrega de experimentación — Respuesta a revisores COMPAG-D-26-05203R1\n",
             "Cada carpeta corresponde a uno o más comentarios. ✅ = listo, ⏳ = pendiente "
             "(la experimentación sigue corriendo o falta insumo).\n",
             "Protocolo completo: `03_.../protocolo.md`. Código: `Prueba3/`.\n"]
    for folder, (comments, section, files) in PLAN.items():
        d = os.path.join(DEST, folder)
        os.makedirs(d, exist_ok=True)
        lines.append(f"\n## {folder}\n**Responde a:** {comments}  \n**Va en el manuscrito:** {section}\n")
        for src, desc in files:
            ok = os.path.exists(src)
            if ok:
                shutil.copy2(src, d)
            lines.append(f"- {'✅' if ok else '⏳'} `{os.path.basename(src)}` — {desc}\n")
    with open(os.path.join(DEST, "README.md"), "w") as fo:
        fo.writelines(lines)
    print(open(os.path.join(DEST, "README.md")).read())


if __name__ == "__main__":
    main()
