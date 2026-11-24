"""
xai_quant.py (Prueba3)

XAI cuantitativo para RootCVT (Revisor 1 C7, Revisor 3 C4).
Cada imagen se explica con el modelo del fold de plant_cv en el que esa PLANTA estuvo en test
(el modelo nunca vio la planta), sobre TODAS las imágenes de desarrollo (rep 0), no un ejemplo por clase.

Métricas:
  1. Concentración en raíz: (fracción de |atribución| sobre píxeles de raíz) / (fracción de área de raíz).
     >1 indica que el mapa se concentra en raíces más que una distribución uniforme.
     Métodos: Grad-CAM (última conv del backbone), IG con baseline negro (referencia, trivial),
     gris, imagen difuminada y ruido uniforme.
  2. Deletion / insertion (Samek et al. 2017; Petsiuk et al. 2018) sobre los píxeles de raíz:
     orden por atribución vs orden aleatorio. AUC de la probabilidad de la clase predicha.
  3. Sanity check de randomización en cascada (Adebayo et al. 2018): correlación de Spearman entre
     el mapa original y el mapa tras re-inicializar progresivamente las capas desde la salida.
  4. Cobertura de tokens: fracción de atribución fuera de la región que cubren los 2x2 patches
     (patch 11 sobre un mapa 28x28 → sólo 22x22 se tokeniza).

Salidas: results/xai/
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

import common as C
import config_transf as cfg

OUT = os.path.join(cfg.RESULTS_DIR, "xai")
os.makedirs(OUT, exist_ok=True)
FRACS = np.linspace(0, 1, 21)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def last_backbone_conv(model):
    import tensorflow as tf
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    patch = [l for l in convs if l.kernel_size[0] > 3]
    assert len(patch) == 1, "se esperaba una sola conv de patch embedding"
    idx = convs.index(patch[0])
    return [l for l in convs[:idx] if l.kernel_size == (3, 3)][-1]


def make_gradcam_fn(model):
    import tensorflow as tf
    layer = last_backbone_conv(model)
    gm = tf.keras.Model(model.inputs, [layer.output, model.output])

    @tf.function
    def f(x, cls):
        with tf.GradientTape() as tape:
            a, p = gm(x, training=False)
            s = tf.reduce_sum(p * tf.one_hot(cls, 2), axis=1)
        g = tape.gradient(s, a)
        w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * a, axis=-1))
        cam = tf.image.resize(cam[..., None], (cfg.IMG_SIZE, cfg.IMG_SIZE), method="bilinear")[..., 0]
        return cam
    return f, layer.name


def make_ig_fn(model, steps=64):
    import tensorflow as tf

    @tf.function
    def grads(xs, cls):
        with tf.GradientTape() as tape:
            tape.watch(xs)
            p = model(xs, training=False)[:, cls]
        return tape.gradient(p, xs)

    def ig(x, base, cls):
        alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1), (-1, 1, 1, 1))
        xs = base[None] + alphas * (x[None] - base[None])
        g = grads(tf.cast(xs, tf.float32), int(cls)).numpy()
        avg = ((g[:-1] + g[1:]) / 2).mean(0)
        return ((x - base) * avg)[..., 0]
    return ig


def baselines(x, rng):
    from scipy.ndimage import gaussian_filter
    return {
        "black": np.zeros_like(x),
        "gray": np.full_like(x, 127.5),
        "blur": gaussian_filter(x, sigma=(8, 8, 0)),
        "noise": rng.uniform(0, 255, size=x.shape).astype(np.float32),
    }


def concentration(attr, root):
    a = np.abs(attr)
    tot = a.sum()
    if tot <= 0 or root.mean() == 0:
        return np.nan
    return (a[root].sum() / tot) / root.mean()


def del_ins_curves(model, x, attr, cls, rng):
    """Deletion/insertion sólo sobre píxeles de raíz (x>0). Devuelve AUC por orden."""
    root_idx = np.flatnonzero(x[..., 0] > 0)
    orders = {"attr": root_idx[np.argsort(-attr.ravel()[root_idx])]}
    for r in range(3):
        orders[f"rand{r}"] = rng.permutation(root_idx)
    res = {}
    for name, order in orders.items():
        dels, inss = [], []
        for fr in FRACS:
            k = int(round(fr * len(order)))
            xd = x.copy().reshape(-1)
            xd[order[:k]] = 0
            xi = np.zeros_like(x).reshape(-1)
            xi[order[:k]] = x.reshape(-1)[order[:k]]
            dels.append(xd.reshape(x.shape))
            inss.append(xi.reshape(x.shape))
        pd_ = model.predict(np.stack(dels), verbose=0, batch_size=32)[:, cls]
        pi_ = model.predict(np.stack(inss), verbose=0, batch_size=32)[:, cls]
        res[name] = (_trapz(pd_, FRACS), _trapz(pi_, FRACS))
    rd = np.mean([res[k][0] for k in res if k.startswith("rand")])
    ri = np.mean([res[k][1] for k in res if k.startswith("rand")])
    return res["attr"][0], res["attr"][1], rd, ri


def randomization_test(model, X, cls, gradcam_fn_maker, ig_maker, rng):
    """Cascada: re-inicializa kernels/bias (Conv, Dense, MHA, Embedding) desde la salida hacia la entrada."""
    import tensorflow as tf
    from scipy.stats import spearmanr
    weighted = [l for l in model.layers if l.trainable_weights and
                not isinstance(l, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization))]
    orig_w = [w.numpy() for w in model.weights]
    gc0, _ = gradcam_fn_maker(model)
    ig0 = ig_maker(model)
    base = np.full_like(X[0], 127.5)
    ref_gc = [gc0(tf.constant(x[None]), tf.constant([c])).numpy()[0] for x, c in zip(X, cls)]
    ref_ig = [np.abs(ig0(x, base, c)) for x, c in zip(X, cls)]
    rows = []
    stages = [0.1, 0.25, 0.5, 0.75, 1.0]
    rev = weighted[::-1]
    done = 0
    for st in stages:
        upto = int(round(st * len(rev)))
        for l in rev[done:upto]:
            for w in l.trainable_weights:
                v = w.numpy()
                sd = v.std() if v.std() > 0 else 0.05
                w.assign(rng.normal(0, sd, size=v.shape).astype(v.dtype))
        done = upto
        gc, _ = gradcam_fn_maker(model)
        igf = ig_maker(model)
        for i, (x, c) in enumerate(zip(X, cls)):
            g = gc(tf.constant(x[None]), tf.constant([c])).numpy()[0]
            a = np.abs(igf(x, base, c))
            rows.append({"stage_frac_layers_randomized": st, "img": i,
                         "gradcam_spearman": spearmanr(ref_gc[i].ravel(), g.ravel())[0],
                         "ig_gray_spearman": spearmanr(ref_ig[i].ravel(), a.ravel())[0]})
    for w, v in zip(model.weights, orig_w):
        w.assign(v)
    return pd.DataFrame(rows)


def main():
    import tensorflow as tf
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rootcvt", help="carpeta de resultados en results/plant_cv/")
    ap.add_argument("--arch", default=None, help="arquitectura (builder) si difiere de --model, p.ej. rootcvt para rootcvt_tf211")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--max_per_fold", type=int, default=999)
    ap.add_argument("--n_random_test", type=int, default=16)
    args = ap.parse_args()

    C.setup_gpu()
    X, y, meta = C.load_dataset()
    pos = {f: i for i, f in enumerate(meta.fname)}
    rng = np.random.default_rng(0)
    rows = []
    rand_imgs = []
    base_dir = os.path.join(cfg.RESULTS_DIR, "plant_cv", args.model)
    folds = sorted(d for d in os.listdir(base_dir) if d.startswith(f"rep{args.rep}_"))
    for fd in folds:
        run = os.path.join(base_dir, fd)
        P = pd.read_csv(os.path.join(run, "preds.csv")).head(args.max_per_fold)
        tf.keras.utils.set_random_seed(0)
        model = C.build_model(args.arch or args.model)
        wpath = os.path.join(run, "best.weights.h5")
        try:
            model.load_weights(wpath)
        except Exception:  # pesos guardados por TF 2.11 (H5 clásico con extensión .weights.h5)
            import shutil, tempfile
            tmp = os.path.join(tempfile.gettempdir(), f"legacy_{args.model}_{fd}.h5")
            shutil.copy(wpath, tmp)
            model.load_weights(tmp)
        gc_fn, gc_layer = make_gradcam_fn(model)
        ig_fn = make_ig_fn(model)
        for _, r in P.iterrows():
            x = X[pos[r.fname]].astype(np.float32)
            pred = int(r.p_sequia >= 0.5)
            root = x[..., 0] > 0
            gc = gc_fn(tf.constant(x[None]), tf.constant([pred])).numpy()[0]
            out = {"fold": fd, "fname": r.fname, "plant": r.plant, "y_true": int(r.y_true), "pred": pred,
                   "correct": int(pred == r.y_true), "p_pred": r.p_sequia if pred else 1 - r.p_sequia,
                   "root_area_frac": root.mean(),
                   "conc_gradcam": concentration(gc, root)}
            igs = {}
            for bn, b in baselines(x, rng).items():
                igs[bn] = ig_fn(x, b, pred)
                out[f"conc_ig_{bn}"] = concentration(igs[bn], root)
            # cobertura de tokens: patch 11 → 2 patches * 11 = 22 de 28 → 176 px de 224
            cov = np.zeros(root.shape, bool)
            cov[:176, :176] = True
            for nm, a in [("gradcam", gc), ("ig_gray", igs["gray"])]:
                aa = np.abs(a)
                out[f"outside_tokens_frac_{nm}"] = aa[~cov].sum() / max(aa.sum(), 1e-12)
            out["outside_tokens_root_frac"] = root[~cov].sum() / max(root.sum(), 1)
            for nm, a in [("gradcam", gc), ("ig_gray", igs["gray"])]:
                d, i, rd, ri = del_ins_curves(model, x, a, pred, rng)
                out.update({f"del_auc_{nm}": d, f"ins_auc_{nm}": i,
                            f"del_auc_random_{nm}": rd, f"ins_auc_random_{nm}": ri})
            rows.append(out)
            if len(rand_imgs) < args.n_random_test and fd == folds[0]:
                rand_imgs.append((x, pred))
        if fd == folds[0]:
            R = randomization_test(model, np.stack([a for a, _ in rand_imgs]), [b for _, b in rand_imgs],
                                   make_gradcam_fn, make_ig_fn, np.random.default_rng(1))
            R.to_csv(os.path.join(OUT, f"randomization_{args.model}.csv"), index=False)
        print(f"[XAI] {fd}: {len(P)} imgs (Grad-CAM layer {gc_layer})", flush=True)
        tf.keras.backend.clear_session()

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, f"xai_per_image_{args.model}.csv"), index=False)
    cols = [c for c in D.columns if c.startswith(("conc_", "del_", "ins_", "outside_"))]
    S = D.groupby("correct")[cols].agg(["mean", "std"]).T
    S["all_mean"] = D[cols].mean()
    S.to_csv(os.path.join(OUT, f"xai_summary_{args.model}.csv"))
    # Pruebas pareadas: atribución vs orden aleatorio
    from scipy.stats import wilcoxon
    tests = {}
    for nm in ["gradcam", "ig_gray"]:
        tests[f"deletion_{nm}_vs_random"] = float(wilcoxon(D[f"del_auc_{nm}"], D[f"del_auc_random_{nm}"]).pvalue)
        tests[f"insertion_{nm}_vs_random"] = float(wilcoxon(D[f"ins_auc_{nm}"], D[f"ins_auc_random_{nm}"]).pvalue)
    C.save_json(tests, os.path.join(OUT, f"xai_tests_{args.model}.json"))
    print(S.round(3).to_string())
    print(tests)


if __name__ == "__main__":
    main()
