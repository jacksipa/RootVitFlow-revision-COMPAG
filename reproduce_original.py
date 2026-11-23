"""
reproduce_original.py (Prueba3)

Control de reproducibilidad: re-entrena con el protocolo ORIGINAL de Prueba1
(partición aleatoria por imagen 160/40 de Balanced_partition_2, validación = test, LR constante),
para verificar que el entorno actual (TF 2.21 + tf_keras) reproduce el comportamiento reportado.
Así, cualquier caída en las particiones por planta se debe a la partición y no al entorno.

Uso: python reproduce_original.py --model rootcvt_7b --seeds 0 1 2
"""
import os
import argparse
import numpy as np

import common as C
import config_transf as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rootcvt_7b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="original_protocol")
    args = ap.parse_args()
    C.setup_gpu()
    Xtr = np.load(os.path.join(cfg.ORIG_BALANCED_DIR, "X_Train.npy"))
    ytr = np.load(os.path.join(cfg.ORIG_BALANCED_DIR, "y_Train.npy")).argmax(1)
    Xte = np.load(os.path.join(cfg.ORIG_BALANCED_DIR, "X_Test.npy"))
    yte = np.load(os.path.join(cfg.ORIG_BALANCED_DIR, "y_Test.npy")).argmax(1)
    for s in args.seeds:
        out = os.path.join(cfg.RESULTS_DIR, args.tag, args.model, f"seed{s}")
        if os.path.exists(os.path.join(out, "info.json")):
            continue
        p, _, info = C.train_and_predict(args.model, Xtr, ytr, Xte, yte, Xte, 1234 + s, out,
                                         save_weights=False)
        info["test_metrics"] = C.compute_metrics(yte, p)
        info["note"] = "val = test (protocolo original, optimista por construcción)"
        import tensorflow as tf
        info["tf_version"] = tf.__version__
        C.save_json(info, os.path.join(out, "info.json"))
        print(args.model, s, "test acc (=val, max) =", round(info["test_metrics"]["accuracy"], 4),
              "best_ep", info["best_epoch"], "train acc final", round(info["final_train_acc"], 3), flush=True)


if __name__ == "__main__":
    main()
