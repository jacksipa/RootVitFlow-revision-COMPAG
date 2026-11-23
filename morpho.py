"""
morpho.py (Prueba3)

Revisor 1 C8 (Tabla 5 incompleta) y Revisor 3 C5 (rasgos morfométricos no vinculados al estrés).

1) Extrae rasgos por imagen de las máscaras binarias a resolución completa (0.0746 mm/px):
   - Los 3 rasgos del manuscrito: longitud total, área total y ancho medio (= área/longitud)
   - Rasgos extendidos: nº de componentes, nº de componentes >= 1 mm, fracción de fragmentos
     < 1 mm, profundidad y extensión lateral, área de la envolvente convexa, solidez y
     centroide vertical relativo.
2) Estadística riego vs sequía:
   - Medias por planta (n = 8 vs 8): Mann-Whitney U + Hedges g, con corrección de Holm
   - Modelo lineal mixto: log(rasgo) ~ tratamiento * día + (1 | planta)
3) ML clásico (LogReg, SVM-RBF, RF, kNN, HistGB) con las MISMAS particiones que los modelos
   profundos (plant_cv, image_cv, holdout). Sin ajuste de hiperparámetros (valores por defecto).

Salidas: results/morpho/
"""
import os
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool

import common as C
import config_transf as cfg

warnings.filterwarnings("ignore")
PX_MM = 0.0746
OUT = os.path.join(cfg.RESULTS_DIR, "morpho")
os.makedirs(OUT, exist_ok=True)

PAPER3 = ["length_mm", "area_mm2", "mean_width_mm"]
EXTENDED = PAPER3 + ["n_components", "n_comp_ge1mm", "frac_fragments_lt1mm", "depth_mm",
                     "spread_mm", "hull_area_mm2", "solidity", "centroid_y_rel"]


def features_one(fname):
    import cv2
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    m = cv2.imread(os.path.join(cfg.IMG_DIR, fname), cv2.IMREAD_GRAYSCALE) > 0
    H, W = m.shape
    skel = skeletonize(m)
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    comp_len_px = ndi.sum(skel, lab, index=np.arange(1, n + 1)) if n else np.array([])
    comp_len_mm = comp_len_px * PX_MM
    ys, xs = np.nonzero(m)
    area_mm2 = m.sum() * PX_MM ** 2
    length_mm = skel.sum() * PX_MM
    hull_mm2 = 0.0
    if len(xs) >= 3:
        pts = np.stack([xs, ys], 1).astype(np.int32)
        hull = cv2.convexHull(pts)
        hull_mm2 = cv2.contourArea(hull) * PX_MM ** 2
    return {
        "fname": fname, "img_w": W, "img_h": H,
        "length_mm": length_mm, "area_mm2": area_mm2,
        "mean_width_mm": area_mm2 / length_mm if length_mm > 0 else np.nan,
        "length_main_mm": float(comp_len_mm.max()) if n else 0.0,
        "n_components": int(n),
        "n_comp_ge1mm": int((comp_len_mm >= 1.0).sum()),
        "frac_fragments_lt1mm": float((comp_len_mm < 1.0).mean()) if n else np.nan,
        "depth_mm": float((ys.max() - ys.min()) * PX_MM) if len(ys) else 0.0,
        "spread_mm": float((xs.max() - xs.min()) * PX_MM) if len(xs) else 0.0,
        "hull_area_mm2": hull_mm2,
        "solidity": area_mm2 / hull_mm2 if hull_mm2 > 0 else np.nan,
        "centroid_y_rel": float(ys.mean() / H) if len(ys) else np.nan,
    }


def extract_features():
    path = os.path.join(OUT, "features_per_image.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    _, _, meta = C.load_dataset()
    with Pool(12) as pool:
        rows = pool.map(features_one, list(meta.fname))
    F = meta[["fname", "plant", "date", "day_index", "label"]].merge(pd.DataFrame(rows), on="fname")
    F.to_csv(path, index=False)
    return F


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def group_stats(F):
    from scipy.stats import mannwhitneyu
    import statsmodels.formula.api as smf
    P = F.groupby(["plant", "label"])[EXTENDED].mean().reset_index()
    rows = []
    for f in EXTENDED:
        a = P.loc[P.label == 0, f].dropna().values
        b = P.loc[P.label == 1, f].dropna().values
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        g = d * (1 - 3 / (4 * (len(a) + len(b)) - 9))
        # Modelo mixto con intercepto aleatorio por planta
        D = F[["plant", "label", "day_index", f]].dropna().copy()
        D["yv"] = np.log(D[f] + 1e-6) if (D[f] > 0).all() else D[f]
        D["sequia"] = D["label"]
        try:
            mm = smf.mixedlm("yv ~ sequia * day_index", D, groups=D["plant"]).fit(reml=True)
            b_t, p_t = mm.params["sequia"], mm.pvalues["sequia"]
            b_i, p_i = mm.params["sequia:day_index"], mm.pvalues["sequia:day_index"]
        except Exception:
            b_t = p_t = b_i = p_i = np.nan
        rows.append({"feature": f,
                     "riego_mean": a.mean(), "riego_sd": a.std(ddof=1),
                     "sequia_mean": b.mean(), "sequia_sd": b.std(ddof=1),
                     "hedges_g": g, "mwu_p": p,
                     "lmm_treat_coef": b_t, "lmm_treat_p": p_t,
                     "lmm_treat_x_day_coef": b_i, "lmm_treat_x_day_p": p_i})
    S = pd.DataFrame(rows)
    S["mwu_p_holm"] = holm(S["mwu_p"])
    S["lmm_treat_p_holm"] = holm(S["lmm_treat_p"].fillna(1))
    S["lmm_treat_x_day_p_holm"] = holm(S["lmm_treat_x_day_p"].fillna(1))
    S.to_csv(os.path.join(OUT, "stats_riego_vs_sequia.csv"), index=False)
    # Evolución temporal por tratamiento (para figura)
    T = F.groupby(["label", "date"])[EXTENDED].agg(["mean", "std"]).reset_index()
    T.columns = ["_".join(c).strip("_") for c in T.columns]
    T.to_csv(os.path.join(OUT, "temporal_means.csv"), index=False)
    return S


def classical_ml(F):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    models = {
        "LogReg": lambda s: LogisticRegression(max_iter=5000, random_state=s),
        "SVM-RBF": lambda s: SVC(kernel="rbf", probability=True, random_state=s),
        "RandomForest": lambda s: RandomForestClassifier(n_estimators=500, random_state=s, n_jobs=-1),
        "kNN": lambda s: KNeighborsClassifier(n_neighbors=5),
        "HistGB": lambda s: HistGradientBoostingClassifier(random_state=s),
    }
    A = pd.read_csv(os.path.join(cfg.SPLITS_DIR, "assignments.csv"))
    Fi = F.set_index("fname")
    rows, preds = [], []
    for fset_name, fset in [("paper3", PAPER3), ("extended", EXTENDED)]:
        for scheme in ["plant_cv", "image_cv", "holdout"]:
            for (rep, fold), g in A[A.scheme == scheme].groupby(["rep", "fold"]):
                tr = g[g.role.isin(["train", "val"])].fname  # sin early stopping → train+val
                te = g[g.role == "test"]
                Xtr, ytr = Fi.loc[tr, fset].values, Fi.loc[tr, "label"].values
                Xte, yte = Fi.loc[te.fname, fset].values, Fi.loc[te.fname, "label"].values
                for mname, mk in models.items():
                    seed = 1234 + 100 * int(rep) + int(fold)
                    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), mk(seed))
                    clf.fit(Xtr, ytr)
                    p = clf.predict_proba(Xte)[:, 1]
                    m = C.compute_metrics(yte, p, te.plant.values)
                    rows.append({"features": fset_name, "scheme": scheme, "model": mname,
                                 "rep": rep, "fold": fold, **m})
                    preds.append(pd.DataFrame({"features": fset_name, "scheme": scheme, "model": mname,
                                               "rep": rep, "fold": fold, "fname": te.fname.values,
                                               "plant": te.plant.values, "y_true": yte, "p_sequia": p}))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "classical_ml_per_fold.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(OUT, "classical_ml_preds.csv"), index=False)
    return R


if __name__ == "__main__":
    F = extract_features()
    print(F[EXTENDED].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    S = group_stats(F)
    print(S[["feature", "riego_mean", "sequia_mean", "hedges_g", "mwu_p_holm", "lmm_treat_p_holm",
             "lmm_treat_x_day_p_holm"]].round(4).to_string())
    R = classical_ml(F)
    summ = R.groupby(["features", "scheme", "model"])[["accuracy", "balanced_accuracy", "f1_macro", "auc"]] \
        .agg(["mean", "std"]).round(3)
    summ.to_csv(os.path.join(OUT, "classical_ml_summary.csv"))
    print(summ.to_string())
