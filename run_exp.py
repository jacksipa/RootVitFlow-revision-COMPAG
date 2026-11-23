"""
run_exp.py (Prueba3)

Entrena y evalúa un modelo sobre un esquema de partición de splits/assignments.csv.
Reanudable: omite las corridas que ya tienen info.json.

Uso:
  source env.sh
  python run_exp.py --model rootcvt --scheme plant_cv
  python run_exp.py --model mobilenetv2 --scheme holdout
Esquemas: plant_cv | image_cv | holdout
"""
import os
import argparse
import numpy as np
import pandas as pd

import common as C
import config_transf as cfg

BASE_SEED = 1234  # semilla global reportada en el manuscrito: seed = BASE_SEED + 100*rep + fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODEL_BUILDERS))
    ap.add_argument("--scheme", required=True, choices=["plant_cv", "image_cv", "holdout"])
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--reps", type=int, nargs="*", default=None)
    args = ap.parse_args()

    skip = os.path.join(cfg.PROJECT_ROOT, "skip", f"{args.model}_{args.scheme}")
    if os.path.exists(skip):
        print(f"[SKIP] {args.model}/{args.scheme} omitido por prioridad de tiempo ({skip})")
        return
    C.setup_gpu()
    X, y, meta = C.load_dataset()
    pos = {f: i for i, f in enumerate(meta.fname)}
    A = pd.read_csv(os.path.join(cfg.SPLITS_DIR, "assignments.csv"))
    A = A[A.scheme == args.scheme]
    if args.reps is not None:
        A = A[A.rep.isin(args.reps)]

    keep_w = args.model.startswith("rootcvt")  # pesos sólo para RootCVT (XAI)
    for (rep, fold), g in A.groupby(["rep", "fold"]):
        out = os.path.join(cfg.RESULTS_DIR, args.scheme, args.model, f"rep{rep}_fold{fold}")
        if os.path.exists(os.path.join(out, "info.json")):
            print(f"[SKIP] {out}")
            continue
        os.makedirs(out, exist_ok=True)
        try:  # bloqueo por corrida: permite repartir un mismo modelo/esquema entre varias colas
            os.close(os.open(os.path.join(out, "running.lock"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            print(f"[LOCKED] {out} (otra cola la está corriendo)")
            continue
        idx = {ro: np.array([pos[f] for f in gg.fname]) for ro, gg in g.groupby("role")}
        seed = BASE_SEED + 100 * int(rep) + int(fold)
        print(f"[RUN] {args.model} {args.scheme} rep{rep} fold{fold} seed={seed} "
              f"train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}", flush=True)
        p_te, p_va, info = C.train_and_predict(
            args.model, X[idx["train"]], y[idx["train"]], X[idx["val"]], y[idx["val"]],
            X[idx["test"]], seed, out, epochs=args.epochs, save_weights=keep_w)
        te = meta.iloc[idx["test"]]
        pd.DataFrame({"fname": te.fname.values, "plant": te.plant.values, "date": te.date.values,
                      "day_index": te.day_index.values, "y_true": te.label.values,
                      "p_sequia": p_te}).to_csv(os.path.join(out, "preds.csv"), index=False)
        m = C.compute_metrics(te.label.values, p_te, te.plant.values)
        info.update({"scheme": args.scheme, "rep": int(rep), "fold": int(fold),
                     "test_plants": sorted(te.plant.unique().tolist()),
                     "val_plants": sorted(meta.iloc[idx["val"]].plant.unique().tolist())
                     if args.scheme != "image_cv" else None,
                     "val_acc_best_ckpt": float(((p_va >= 0.5).astype(int) == y[idx["val"]]).mean()),
                     "test_metrics": m})
        C.save_json(info, os.path.join(out, "info.json"))
        os.remove(os.path.join(out, "running.lock"))
        print(f"   -> test acc={m['accuracy']:.3f} auc={m['auc']:.3f} best_ep={info['best_epoch']} "
              f"val={info['best_val_acc']:.3f} t={info['train_time_s']}s", flush=True)


if __name__ == "__main__":
    main()
