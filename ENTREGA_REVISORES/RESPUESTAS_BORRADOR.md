# Borrador de respuestas — COMPAG-D-26-05203R1 (comentarios marcados "Experimentación")

> **Nota para los autores (leer antes de usar este borrador).**
> La experimentación pedida por los revisores **no sostiene el 99% de RootCVT**. Con plantas que el modelo nunca
> vio (validación agrupada por planta y un test independiente), RootCVT y todas las CNN quedan en el nivel del azar
> (≈40–60%). El 99.00 ± 1.22% del manuscrito provenía de evaluar pesos ya entrenados sobre imágenes que incluían las
> de entrenamiento (Prueba2/cross.py), y el 100% del test era el mejor de 6 corridas elegido sobre el mismo test.
> Lo que SÍ se sostiene y mejora: la segmentación (Dice ≈ 70% en plantas no vistas, frente al ≈ 40–53% reportado) y un
> efecto biológico real en la **dinámica** de crecimiento radicular (interacción tratamiento × día).
> Recomendación: reenfocar la contribución en segmentación + fenotipado morfométrico, y presentar la clasificación como
> resultado negativo/limitación honesta. Decidir esto es de ustedes; el texto de abajo está escrito con ese enfoque.
> RootCVT se reporta con TensorFlow 2.11.1 (la versión original del cluster): 96.1% de accuracy en entrenamiento,
> 45.1 ± 25.5% en CV agrupada por planta y 50.0% en el test independiente.
> Todas las cifras salen de `ENTREGA_REVISORES/` (tablas T1–T7) y son reproducibles con el código de `Prueba3/`.

---

## Reviewer #1

### Comment 4 — Data leakage from temporally proximate images of the same plants

**Response 4:** We thank the reviewer for this important observation, which we have now addressed experimentally.
A re-examination of the acquisition metadata showed that the 200 images correspond to 16 plants (8 irrigated, 8
drought) imaged **daily** between 16 and 30 July 2025 (10–15 images per plant), not at ~6-day intervals as stated in
the previous version. The original image-level random split therefore placed near-duplicate images of the same plant
in both training and evaluation sets. We repeated all classification experiments with **plant-grouped** partitions
(leave-one-irrigated/drought-pair-out cross-validation; validation on a separate pair of plants; 6 folds × 2
repetitions) and compared them with image-level partitions using exactly the same model, data and hyperparameters.
The comparison confirms that leakage inflated performance: the same RootCVT network reached 92.9 ± 5.5% accuracy with
image-level folds but 58.5 ± 27.2% with plant-grouped folds (no data augmentation variant; Table T1), and all models
fall to chance level on plants not seen during training (Comment 5). The manuscript now reports only plant-grouped
results, and the previous figures have been withdrawn.

**Revised text 4:**
- *Section 3.1:* "The dataset comprises 200 binary root masks from 16 tomato plants (8 irrigated, 8 under water
  deficit) acquired daily from 16 to 30 July 2025 (10–15 images per plant)."
- *Section 3.6:* "To prevent leakage between temporally adjacent images of the same plant, all classification
  experiments used plant-grouped partitions: four plants (two per treatment; rb3, rb8, sb13, sb15; 60 images) were
  held out before any experiment as an independent test set, and the remaining 12 plants were evaluated with
  leave-one-pair-out cross-validation (one irrigated and one drought plant per test fold, a different pair for
  validation/checkpoint selection, eight plants for training; two repetitions). Image-to-fold assignment files are
  provided in the code repository."
- *Section 4.2.2:* "When the same network and hyperparameters were evaluated with image-level folds, accuracy reached
  92.9 ± 5.5%, whereas plant-grouped folds yielded 58.5 ± 27.2% (Table T1), showing that image-level validation
  mainly measured the recognition of previously seen plants."

### Comment 5 — Substantiate the 99% with independent test data and an appropriate validation strategy

**Response 5:** We agree, and we were unable to substantiate the 99% figure. An audit of the original pipeline
showed that (i) the reported 99.00 ± 1.22% "5-fold" value was obtained by evaluating a single, already-trained model
on folds that contained its own training images, without retraining per fold, and (ii) the 100% test accuracy was the
best of six identically configured runs, with the checkpoint selected on the same test set (the six runs gave 72.5,
90.0, 80.0, 87.5, 70.0 and 100%). We have therefore re-evaluated RootCVT and all baselines with (a) plant-grouped
cross-validation and (b) an independent test set of four plants never used for training, validation or model
selection. Under these conditions RootCVT (trained in the original TensorFlow 2.11 environment) reached
45.1 ± 25.5% accuracy (AUC 46.5 ± 29.8%) in plant-grouped cross-validation, despite 96.1% training accuracy, and
50.0% (AUC 48.1 ± 11.1%) on the independent test set; the CNN baselines obtained 42.5–49.9% (plant-grouped CV) and
33.3–49.4% (independent test), i.e., none of the classifiers discriminates irrigation treatment above chance on
unseen plants (Tables T2–T3). The 99% claim has been removed from the Abstract, Results and Conclusions, and the
classification results are now reported as a limitation of the present dataset (16 plants).

**Revised text 5:**
- *Abstract:* replace the accuracy sentence with: "Under plant-independent evaluation, neither RootCVT nor CNN
  baselines classified the irrigation treatment above chance level, indicating that 16 plants are insufficient to
  learn treatment-specific root patterns that generalise across individuals."
- *Section 4.2.2 / Table 4 (new):* plant-grouped CV and independent-test results for all models (Tables T2 and T3).
- *Section 5.5:* "The high accuracies previously obtained with image-level partitions reflected plant identity rather
  than treatment; with only 16 plants, plant-to-plant variability (e.g., plants rb6 and sb16 were systematically
  assigned to the opposite treatment by almost all models) dominates the treatment effect."

### Comment 6 — Consistent data-splitting methods and metrics across segmentation and classification

**Response 6:** All stages now use the same, fixed partition and the same reporting. The four held-out plants (rb3,
rb8, sb13, sb15; 60 images) form the test set for both segmentation and classification; validation uses a separate
pair of plants; training uses the remaining plants. Segmentation is reported with Dice, IoU, precision, recall,
clDice and skeleton-F1 (2-px tolerance) as mean ± SD over test images; classification with accuracy, balanced
accuracy, macro-F1 and AUC as mean ± SD over folds, plus plant-cluster bootstrap 95% CIs. The image-to-fold assignment
file (`splits/assignments.csv`) and the random seeds (global seed 1234 + 100·repetition + fold; holdout selection seed
2026) are provided.

**Revised text 6:**
- *Section 3.6 (fill [SEMILLA]):* "All experiments were run with fixed random seeds (seed = 1234 + 100·r + k for
  repetition r and fold k; held-out plants selected with seed 2026)…"
- *Sections 3.2.1/3.2.2:* replace the 166/34 and 79.5/16% splits with the plant-grouped partition above.

### Comment 7 — XAI supported by more concrete findings

**Response 7:** We replaced the qualitative, one-image-per-class analysis with a quantitative evaluation over all 140
development images, each explained by the model of the fold in which its plant was held out. Results (Table T6):
(i) with the black baseline used before, Integrated Gradients concentrates attribution on root pixels by
construction (ratio = 182 = 1/root-pixel fraction), so this result was uninformative; with non-trivial baselines
(uniform grey, noise) the concentration ratio is only 1.4–2.9 (three RootCVT variants); (ii) in deletion/insertion tests, Grad-CAM ordering
was barely better than random ordering (deletion AUC 0.72 vs 0.76 for RootCVT/TF 2.11; 0.68 vs 0.70 under TF 2.21),
whereas IG (grey baseline) was clearly better (0.32 vs 0.76; Wilcoxon p < 10⁻²⁰); (iii) in the cascading
model-parameter randomisation test, Grad-CAM maps decorrelated from the original once ≥ 50% of the layers were
randomised (Spearman 0.31 → ≈ 0), but IG maps remained correlated (≈ 0.49–0.66) even with all layers randomised,
indicating that IG mainly reflects the input mask structure; (iv) 9.9% of root pixels fall outside the 2 × 2 token
grid of the transformer. We have therefore removed the claim that XAI "validates" that the model relies on root
structures, and report these quantitative results instead.

**Revised text 7:** *Section 3.5 / 4.4 / 5.4:* "Explanations were evaluated quantitatively on all test images
(root-concentration ratio with four IG baselines, deletion/insertion AUC against random orderings, and cascading
parameter-randomisation sanity checks). Grad-CAM passed the randomisation check but was not more faithful than random
ordering; IG was faithful in deletion/insertion but failed the randomisation check. Taken together, the XAI analyses
do not establish that classification decisions rely on treatment-relevant root traits."

### Comment 8 — Missing values in the classical machine-learning results (Table 5)

**Response 8:** Table 5 has been completed. Five classifiers (logistic regression, SVM-RBF, random forest, k-NN,
histogram gradient boosting; default hyperparameters, no tuning) were trained on (a) the three morphometric features
of the original manuscript (total length, total area, mean width) and (b) an extended set of 11 features, using the
same partitions as the deep models. With plant-grouped CV, accuracy was 56–61% (three features) and 56–66% (extended);
on the independent test set 25–37% and 27–62%, respectively; with image-level folds 61–65% and 72–79%. No cell of the
table is missing any longer.

**Revised text 8:** *Table 5:* replace with Table T5 (`05_.../T5_classical_ml.md`).

---

## Reviewer #3

### Comment 1 — Data leakage invalidates the headline accuracy

**Response 1:** We agree with the reviewer. See Responses 4–5 to Reviewer #1. In addition: (i) plant identities were
recovered from the file names (16 plants, daily images), allowing plant-grouped splits and a held-out set of four
plants; (ii) RootCVT hyperparameters were fixed a priori (no grid search on the test data); the only selection step
(checkpoint epoch) uses a separate validation pair of plants; (iii) RootPainter masks are used as reference for all
images, which we now state explicitly as a limitation (no independent manual annotation). Under this protocol, the
headline accuracy is not reproduced (45.1 ± 25.5% plant-grouped CV; 50.0% independent test; Tables T2–T3).

### Comment 2 — Performance gap, weak baselines, no ablation, 2 × 2 tokens

**Response 2:** All models were re-trained under an identical protocol: same input (224 × 224 binary masks,
replicated to three channels for ImageNet-pretrained CNNs), same augmentation, classification head, loss
(cross-entropy, label smoothing 0.08), optimiser (Adam, lr 3·10⁻⁴, weight decay 10⁻³), 150 epochs and checkpoint
selection on held-out validation plants. Baselines: MobileNetV2, InceptionV3, ResNet50, DenseNet201 (ImageNet). Ablations:
CNN+SE backbone without transformer, and RootCVT with 49 tokens (patch 4) instead of 4 tokens (patch 11). Results
(Table T2, plant-grouped CV): RootCVT 45.1 ± 25.5% (TF 2.11; 46.9 ± 19.4% under TF 2.21); backbone without transformer 57.1 ± 15.4%; 49-token RootCVT
50.9 ± 22.5%; CNNs 42.5–49.9% (all reach 95–99% training accuracy). The transformer does not improve over the CNN
backbone, and we confirm the reviewer's observation that the original configuration operates on a 2 × 2 token grid
(28 × 28 feature map, 11 × 11 patches; 9.9% of root pixels fall outside the tokenised region). The claims about
"higher-order morphological relationships" and the ViT contribution have been removed.

### Comment 3 — Segmentation comparison circular and not rigorous

**Response 3:** We re-ran the segmentation comparison with a single script and protocol. Issues found in the original
notebooks: validation and test were the same 34 images; U-Net/SwinUNETR down-sampled the whole 3072 × 4096 image to
512 × 512 (SAM to 256 × 256), making 1–3-px roots sub-pixel; SAM's validation loop used the logits of the last
training batch, and its box prompt was derived from the ground-truth mask. The new protocol trains on full-resolution
patches with positive/negative sampling (foreground ≈ 0.13% of pixels), selects checkpoints on two validation plants,
tests on the four held-out plants (60 images) by sliding-window inference at full resolution, prompts SAM with the
whole tile (no ground-truth information), and adds topology-aware metrics:

| Model | Dice | IoU | clDice | Skeleton-F1 (2 px) |
|---|---|---|---|---|
| U-Net (MONAI) | 70.4 ± 3.9 | 54.4 ± 4.5 | 82.9 ± 5.0 | 61.7 ± 6.5 |
| SwinUNETR (MONAI) | 69.0 ± 4.3 | 52.9 ± 4.9 | 81.6 ± 5.4 | 60.5 ± 6.4 |
| SAM ViT-B (decoder fine-tuned) | 65.8 ± 3.3 | 49.1 ± 3.6 | 77.2 ± 5.1 | 56.3 ± 5.7 |

(mean ± SD over the 60 test images, %; Table T7). No model produced empty masks. Because the reference masks are RootPainter outputs, the former
96.19% Dice of RootPainter is circular and has been removed; RootPainter is now described as the annotation tool, and
the absence of independent manual annotation / inter-annotator variability is stated as a limitation.

### Comment 4 — XAI validation overstated

**Response 4:** See Response 7 to Reviewer #1 (quantitative evaluation over all test images, non-black IG baselines,
deletion/insertion, randomisation sanity checks). The misclassified example in former Fig. 14 and the untested
biological explanations have been removed.

### Comment 5 — Morphometric module unvalidated and not linked to stress

**Response 5:** We now report irrigated-vs-drought statistics (Table T5b; plant means, n = 8 vs 8, Mann–Whitney with
Holm correction, and linear mixed models log(trait) ~ treatment × day + (1 | plant)). No trait differed between
treatments in its plant-level mean (all Holm-adjusted p = 1.0). However, the **growth dynamics** differed:
the treatment × day interaction was significant for total root length (−3.0% per day under drought relative to
irrigation, Holm p < 0.001), total root area (−2.5%/day, p = 0.004), rooting depth (p = 0.035) and solidity
(p = 0.042). This links the morphometric module to the water-deficit treatment and suggests that temporal trajectories,
not single images, carry the treatment signal. We also report that masks contain on average 22.4 connected
components per image (fragments < 1 mm: 0.3%), and we state the single global calibration factor (0.0746 mm/px) and the
absence of comparison with WinRHIZO/RhizoVision as limitations.

**Revised text 5:** *Table 7* replaced by Table T5b; *Section 4.3:* add the mixed-model results and Fig.
`F_morpho_temporal.png`.

---

### Additional reproducibility note (for Section 3.6 / Code availability)
RootCVT trained with augmentation learns the training set under TensorFlow 2.11 (the original environment) but not
under TensorFlow 2.21/Keras 2 (tf_keras), on both GPU and CPU; RootCVT results are therefore reported with TF 2.11.1
(45.1 ± 25.5% plant-grouped CV, 50.0% independent test), and CNN baselines with TF 2.21. No paired difference between
RootCVT and any baseline or ablation was significant (Wilcoxon over folds, all p > 0.2; Table T2b). Environment files for both are provided.
