# Problemas en los notebooks de segmentación originales (unet/Swin/SAM_Segmentor.ipynb)

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
