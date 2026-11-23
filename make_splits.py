"""
make_splits.py (Prueba3)

Genera y guarda TODAS las particiones (archivos de asignación imagen→fold, Revisor 3 C7).

Diseño (Revisor 1 C4/C5, Revisor 3 C1):
  1. HOLDOUT independiente: 2 plantas de riego + 2 de sequía, elegidas al azar con semilla fija
     ANTES de cualquier entrenamiento. No se usan para entrenar, validar ni elegir modelos.
  2. DESARROLLO (12 plantas restantes):
     a) plant_cv: CV agrupada por planta, leave-one-pair-out (1 riego + 1 sequía en test),
        6 folds × 2 repeticiones. Validación (selección del epoch) = otro par de plantas.
     b) image_cv: CV estratificada por IMAGEN (protocolo original, con fuga), 6 folds × 2
        repeticiones, validación = 25 imágenes al azar del train. Sirve para cuantificar la fuga.
  3. holdout: entrenar en el desarrollo (10 plantas train + 2 plantas val) y evaluar UNA vez en
     las 4 plantas del holdout; 3 repeticiones (distinto par de validación y semilla).

Salida: splits/assignments.csv  (scheme, rep, fold, fname, plant, label, role)
        splits/holdout_plants.json
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import common as C
import config_transf as cfg

HOLDOUT_SEED = 2026
N_REPS_CV = 2
N_REPS_HOLDOUT = 3


def main():
    _, y, meta = C.load_dataset()
    plants = meta.groupby("plant")["label"].first()
    rb = sorted(plants[plants == 0].index, key=lambda s: int(s[2:]))
    sb = sorted(plants[plants == 1].index, key=lambda s: int(s[2:]))

    rng = np.random.default_rng(HOLDOUT_SEED)
    hold = list(rng.choice(rb, 2, replace=False)) + list(rng.choice(sb, 2, replace=False))
    dev_rb = [p for p in rb if p not in hold]
    dev_sb = [p for p in sb if p not in hold]
    C.save_json({"seed": HOLDOUT_SEED, "holdout_plants": hold, "dev_plants": dev_rb + dev_sb,
                 "n_holdout_images": int(meta.plant.isin(hold).sum())},
                os.path.join(cfg.SPLITS_DIR, "holdout_plants.json"))
    print("[HOLDOUT]", hold, "imgs:", int(meta.plant.isin(hold).sum()))

    dev = meta[~meta.plant.isin(hold)].reset_index(drop=True)
    rows = []

    def emit(scheme, rep, fold, df, role):
        for _, r in df.iterrows():
            rows.append({"scheme": scheme, "rep": rep, "fold": fold, "fname": r.fname,
                         "plant": r.plant, "label": int(r.label), "role": role})

    # ---------- a) CV agrupada por planta (leave-one-pair-out) ----------
    for rep in range(N_REPS_CV):
        r = np.random.default_rng(100 + rep)
        prb, psb = list(r.permutation(dev_rb)), list(r.permutation(dev_sb))
        pairs = [(prb[i], psb[i]) for i in range(len(prb))]
        k = len(pairs)
        for fold in range(k):
            test_p = set(pairs[fold])
            val_p = set(pairs[(fold + 1) % k])
            emit("plant_cv", rep, fold, dev[dev.plant.isin(test_p)], "test")
            emit("plant_cv", rep, fold, dev[dev.plant.isin(val_p)], "val")
            emit("plant_cv", rep, fold, dev[~dev.plant.isin(test_p | val_p)], "train")

    # ---------- b) CV por imagen (protocolo original, con fuga) ----------
    for rep in range(N_REPS_CV):
        skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=100 + rep)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(dev, dev.label)):
            tr = dev.iloc[tr_idx]
            tr_, va_ = train_test_split(tr, test_size=25, stratify=tr.label, random_state=100 + rep)
            emit("image_cv", rep, fold, dev.iloc[te_idx], "test")
            emit("image_cv", rep, fold, va_, "val")
            emit("image_cv", rep, fold, tr_, "train")

    # ---------- c) Holdout independiente ----------
    test_df = meta[meta.plant.isin(hold)]
    for rep in range(N_REPS_HOLDOUT):
        r = np.random.default_rng(200 + rep)
        val_p = {r.choice(dev_rb), r.choice(dev_sb)}
        emit("holdout", rep, 0, test_df, "test")
        emit("holdout", rep, 0, dev[dev.plant.isin(val_p)], "val")
        emit("holdout", rep, 0, dev[~dev.plant.isin(val_p)], "train")

    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(cfg.SPLITS_DIR, "assignments.csv"), index=False)

    # Verificaciones anti-fuga
    for (s, rep, fold), g in A.groupby(["scheme", "rep", "fold"]):
        roles = {ro: set(gg.fname) for ro, gg in g.groupby("role")}
        assert not (roles["train"] & roles["test"]) and not (roles["val"] & roles["test"])
        assert not (roles["train"] & roles["val"])
        if s != "image_cv":
            pr = {ro: set(gg.plant) for ro, gg in g.groupby("role")}
            assert not (pr["train"] & pr["test"]) and not (pr["val"] & pr["test"]) and not (pr["train"] & pr["val"])
    summ = A.groupby(["scheme", "role"]).fname.count() / A.groupby(["scheme", "role"]).apply(
        lambda g: g[["rep", "fold"]].drop_duplicates().shape[0])
    print("[OK] Particiones verificadas (sin solapamiento). Imágenes promedio por rol:")
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
