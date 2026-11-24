"""
summarize.py (Prueba3)

Consolida todos los resultados en tablas (CSV + Markdown) y figuras para la respuesta a revisores.
  T1 leakage:        CV por imagen vs CV agrupada por planta (mismo modelo, mismos datos)
  T2 comparison:     todos los modelos con CV agrupada por planta (protocolo idéntico) + ablación
  T3 holdout:        test independiente (4 plantas nunca vistas)
  T4 temporal:       accuracy por día de adquisición (CV por planta)
  T5 morpho:         estadística riego vs sequía + ML clásico (de morpho.py)
  T6 xai:            resumen XAI cuantitativo (de xai_quant.py)
IC95%: bootstrap por conglomerados (se remuestrean PLANTAS dentro de cada clase), 2000 réplicas.
"""
import os
import json
import glob
import numpy as np
import pandas as pd

import common as C
import config_transf as cfg

R = cfg.RESULTS_DIR
OUT = os.path.join(R, "tables")
FIG = os.path.join(R, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

NICE = {"rootcvt": "RootCVT (model97, 9 bloques, 4 tokens)",
        "rootcvt_tf211": "RootCVT (model97) entrenado en TF 2.11 (versión original)",
        "rootcvt_7b": "RootCVT checkpoint-100% (7 bloques, 4 tokens)",
        "rootcvt_noaug": "Sensibilidad: RootCVT sin data augmentation",
        "rootcvt_notransf": "Ablación: backbone CNN+SE sin Transformer",
        "rootcvt_p4": "Ablación: RootCVT patch 4 (49 tokens)",
        "rootcvt_p2": "Ablación: RootCVT patch 2 (196 tokens)",
        "mobilenetv2": "MobileNetV2 (ImageNet)", "inceptionv3": "InceptionV3 (ImageNet)",
        "resnet50": "ResNet50 (ImageNet)", "densenet201": "DenseNet201 (ImageNet)",
        "vgg16": "VGG16 (ImageNet)"}
ORDER = list(NICE)


def load_runs(scheme):
    rows, preds = [], []
    for info_p in glob.glob(os.path.join(R, scheme, "*", "*", "info.json")):
        info = json.load(open(info_p))
        m = info.pop("test_metrics")
        rows.append({**{k: v for k, v in info.items() if not isinstance(v, list)}, **m})
        p = pd.read_csv(os.path.join(os.path.dirname(info_p), "preds.csv"))
        p["model"], p["rep"], p["fold"] = info["model"], info["rep"], info["fold"]
        preds.append(p)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def cluster_boot_ci(P, metric="accuracy", n=2000, seed=0):
    """IC95% remuestreando plantas (por clase) sobre predicciones agrupadas."""
    rng = np.random.default_rng(seed)
    by_plant = {pl: g for pl, g in P.groupby("plant")}
    pr = P.groupby("plant").y_true.first()
    p0, p1 = pr[pr == 0].index.values, pr[pr == 1].index.values
    vals = []
    for _ in range(n):
        s = list(rng.choice(p0, len(p0))) + list(rng.choice(p1, len(p1)))
        D = pd.concat([by_plant[s_] for s_ in s])
        vals.append(C.compute_metrics(D.y_true, D.p_sequia)[metric])
    return np.nanpercentile(vals, [2.5, 97.5])


def fmt(m, s):
    return f"{100*m:.1f} ± {100*s:.1f}"


def summary_table(runs, preds, models, scheme):
    out = []
    for mdl in models:
        r = runs[runs.model == mdl]
        if r.empty:
            continue
        p = preds[preds.model == mdl]
        # métricas agrupadas por repetición (todas las imágenes de desarrollo una vez por rep)
        pooled = [C.compute_metrics(g.y_true, g.p_sequia, g.plant) for _, g in p.groupby("rep")]
        pooled_acc = np.mean([x["accuracy"] for x in pooled])
        pooled_auc = np.mean([x["auc"] for x in pooled])
        plant_acc = np.mean([x["plant_accuracy"] for x in pooled])
        lo, hi = cluster_boot_ci(p[p.rep == p.rep.min()])
        out.append({
            "Modelo": NICE.get(mdl, mdl), "key": mdl, "Esquema": scheme, "n_runs": len(r),
            "Accuracy (media ± DE por fold)": fmt(r.accuracy.mean(), r.accuracy.std()),
            "Bal. accuracy": fmt(r.balanced_accuracy.mean(), r.balanced_accuracy.std()),
            "F1 macro": fmt(r.f1_macro.mean(), r.f1_macro.std()),
            "AUC": fmt(r.auc.mean(), r.auc.std()) if r.auc.notna().any() else "—",
            "Accuracy agrupada": f"{100*pooled_acc:.1f}",
            "IC95% (bootstrap por planta)": f"[{100*lo:.1f}, {100*hi:.1f}]",
            "AUC agrupada": f"{100*pooled_auc:.1f}",
            "Accuracy por planta (voto)": f"{100*plant_acc:.1f}",
            "Val acc (selección)": f"{100*r.best_val_acc.mean():.1f}",
            "Train acc final": f"{100*r.final_train_acc.mean():.1f}",
            "Mejor época (media)": f"{r.best_epoch.mean():.0f}",
            "Parámetros": f"{int(r.n_params.iloc[0]):,}",
            "_acc_mean": r.accuracy.mean(),
        })
    return pd.DataFrame(out)


def to_md(df, path, drop=("key", "_acc_mean")):
    d = df.drop(columns=[c for c in drop if c in df.columns])
    with open(path, "w") as f:
        f.write(d.to_markdown(index=False))


def main():
    cv_runs, cv_preds = load_runs("plant_cv")
    im_runs, im_preds = load_runs("image_cv")
    ho_runs, ho_preds = load_runs("holdout")

    # ---------- T1: fuga ----------
    leak_models = [m for m in ORDER if m in set(im_runs.get("model", []))]
    T1 = pd.concat([summary_table(im_runs, im_preds, leak_models, "CV por imagen (con fuga)"),
                    summary_table(cv_runs, cv_preds, leak_models, "CV agrupada por planta")])
    T1.to_csv(os.path.join(OUT, "T1_leakage.csv"), index=False)
    to_md(T1, os.path.join(OUT, "T1_leakage.md"))

    # ---------- T2: comparación con CV por planta ----------
    models = [m for m in ORDER if m in set(cv_runs.get("model", []))]
    T2 = summary_table(cv_runs, cv_preds, models, "CV agrupada por planta")
    T2.to_csv(os.path.join(OUT, "T2_models_plant_cv.csv"), index=False)
    to_md(T2, os.path.join(OUT, "T2_models_plant_cv.md"))
    # Pruebas pareadas por fold vs RootCVT
    if "rootcvt" in models:
        from scipy.stats import wilcoxon
        base = cv_runs[cv_runs.model == "rootcvt"].set_index(["rep", "fold"]).accuracy
        rows = []
        for m in models:
            if m == "rootcvt":
                continue
            o = cv_runs[cv_runs.model == m].set_index(["rep", "fold"]).accuracy
            j = base.to_frame("a").join(o.to_frame("b"), how="inner")
            if len(j) >= 5 and (j.a - j.b).abs().sum() > 0:
                p = wilcoxon(j.a, j.b).pvalue
            else:
                p = np.nan
            rows.append({"modelo": NICE[m], "diff_acc_vs_rootcvt_pp": 100 * (j.b - j.a).mean(),
                         "wilcoxon_p": p, "n_pares": len(j)})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "T2b_paired_tests_vs_rootcvt.csv"), index=False)

    # ---------- T3: holdout independiente ----------
    if not ho_runs.empty:
        hm = [m for m in ORDER if m in set(ho_runs.model)]
        rows = []
        for m in hm:
            r = ho_runs[ho_runs.model == m]
            p = ho_preds[ho_preds.model == m]
            ens = p.groupby(["fname", "plant", "y_true"]).p_sequia.mean().reset_index()
            me = C.compute_metrics(ens.y_true, ens.p_sequia, ens.plant)
            lo, hi = cluster_boot_ci(ens)
            per_plant = ens.groupby("plant").apply(
                lambda g: ((g.p_sequia >= .5).astype(int) == g.y_true).mean()).round(2).to_dict()
            rows.append({"Modelo": NICE[m], "n_reps": len(r),
                         "Accuracy (media ± DE reps)": fmt(r.accuracy.mean(), r.accuracy.std()),
                         "AUC (media ± DE)": fmt(r.auc.mean(), r.auc.std()),
                         "F1 macro (media ± DE)": fmt(r.f1_macro.mean(), r.f1_macro.std()),
                         "Accuracy ensemble": f"{100*me['accuracy']:.1f}",
                         "IC95% planta": f"[{100*lo:.1f}, {100*hi:.1f}]",
                         "AUC ensemble": f"{100*me['auc']:.1f}",
                         "Acc. por planta (ensemble)": json.dumps(per_plant)})
        T3 = pd.DataFrame(rows)
        T3.to_csv(os.path.join(OUT, "T3_holdout.csv"), index=False)
        to_md(T3, os.path.join(OUT, "T3_holdout.md"))

    # ---------- T4: temporal ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "rootcvt" in models:
        P = cv_preds[cv_preds.model == "rootcvt"].copy()
        P["correct"] = ((P.p_sequia >= .5).astype(int) == P.y_true).astype(int)
        T4 = P.groupby("date").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                   p_sequia_riego=("p_sequia", lambda s: s[P.loc[s.index, "y_true"] == 0].mean()),
                                   p_sequia_sequia=("p_sequia", lambda s: s[P.loc[s.index, "y_true"] == 1].mean()))
        T4.to_csv(os.path.join(OUT, "T4_temporal_rootcvt.csv"))
        if not im_preds.empty and "rootcvt" in set(im_preds.model):
            Q = im_preds[im_preds.model == "rootcvt"].copy()
            Q["correct"] = ((Q.p_sequia >= .5).astype(int) == Q.y_true).astype(int)
            T4i = Q.groupby("date").correct.mean()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(pd.to_datetime(T4.index), 100 * T4.accuracy, "o-", label="CV agrupada por planta")
        if not im_preds.empty and "rootcvt" in set(im_preds.model):
            ax.plot(pd.to_datetime(T4i.index), 100 * T4i.values, "s--", label="CV por imagen (con fuga)")
        ax.axhline(50, color="gray", lw=0.8, ls=":")
        ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("Fecha de adquisición"); ax.set_ylim(0, 105)
        ax.legend(frameon=False); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(os.path.join(FIG, "F_temporal_accuracy.png"), dpi=200); plt.close(fig)

    # ---------- Figura fuga + comparación ----------
    if not T2.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        D = T2.sort_values("_acc_mean")
        accs = [cv_runs[cv_runs.model == k].accuracy.values * 100 for k in D.key]
        ax.boxplot(accs, orientation="horizontal", tick_labels=list(D.Modelo), showmeans=True)
        if not im_runs.empty:
            for i, k in enumerate(D.key):
                v = im_runs[im_runs.model == k].accuracy
                if len(v):
                    ax.plot(100 * v.mean(), i + 1, "r*", ms=12, label="CV por imagen (media)" if i == 0 or k == "rootcvt" else None)
        ax.axvline(50, color="gray", ls=":")
        ax.set_xlabel("Accuracy en test por fold (%) — CV agrupada por planta")
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h[:1], l[:1], frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "F_models_plant_cv.png"), dpi=200); plt.close(fig)

    # ---------- T5 morfometría ----------
    mo = os.path.join(R, "morpho")
    if os.path.exists(os.path.join(mo, "classical_ml_per_fold.csv")):
        M = pd.read_csv(os.path.join(mo, "classical_ml_per_fold.csv"))
        T5 = M.groupby(["features", "scheme", "model"]).agg(
            acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
            f1_mean=("f1_macro", "mean"), auc_mean=("auc", "mean")).reset_index()
        T5.to_csv(os.path.join(OUT, "T5_classical_ml.csv"), index=False)
        to_md(T5.round(3), os.path.join(OUT, "T5_classical_ml.md"))
        S = pd.read_csv(os.path.join(mo, "stats_riego_vs_sequia.csv"))
        to_md(S.round(4), os.path.join(OUT, "T5b_morpho_stats.md"))
        F = pd.read_csv(os.path.join(mo, "features_per_image.csv"))
        feats = ["length_mm", "area_mm2", "mean_width_mm", "n_components", "depth_mm", "solidity"]
        fig, axs = plt.subplots(2, 3, figsize=(11, 6))
        for ax, f in zip(axs.ravel(), feats):
            for lab, nm, c in [(0, "Riego", "tab:blue"), (1, "Sequía", "tab:orange")]:
                g = F[F.label == lab].groupby("date")[f]
                mu, sd = g.mean(), g.std()
                x = pd.to_datetime(mu.index)
                ax.plot(x, mu, "o-", color=c, label=nm, ms=3)
                ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=.15)
            ax.set_title(f); ax.tick_params(axis="x", rotation=45, labelsize=7)
        axs[0, 0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "F_morpho_temporal.png"), dpi=200); plt.close(fig)

    # ---------- T6 XAI (todos los modelos analizados) ----------
    parts, rparts = [], []
    for xp in sorted(glob.glob(os.path.join(R, "xai", "xai_per_image_*.csv"))):
        mname = os.path.basename(xp)[len("xai_per_image_"):-4]
        X = pd.read_csv(xp)
        cols = [c for c in X.columns if c.startswith(("conc_", "del_", "ins_", "outside_"))]
        t = X[cols].agg(["mean", "std", "median"]).T
        t.insert(0, "modelo", mname)
        t["n_imgs"] = len(X)
        t["acc_imgs"] = X.correct.mean()
        parts.append(t)
        rp = os.path.join(R, "xai", f"randomization_{mname}.csv")
        if os.path.exists(rp):
            RD = pd.read_csv(rp).groupby("stage_frac_layers_randomized")[["gradcam_spearman", "ig_gray_spearman"]].agg(["mean", "std"])
            RD.columns = ["_".join(c) for c in RD.columns]
            RD.insert(0, "modelo", mname)
            rparts.append(RD.reset_index())
    if parts:
        T6 = pd.concat(parts)
        T6.to_csv(os.path.join(OUT, "T6_xai.csv"))
        to_md(T6.round(3).reset_index().rename(columns={"index": "métrica"}), os.path.join(OUT, "T6_xai.md"), drop=())
    if rparts:
        RD = pd.concat(rparts)
        RD.to_csv(os.path.join(OUT, "T6b_randomization.csv"), index=False)
        to_md(RD.round(3), os.path.join(OUT, "T6b_randomization.md"), drop=())

    print("Tablas en", OUT)
    for f in sorted(glob.glob(os.path.join(OUT, "*.md"))):
        print("\n####", os.path.basename(f))
        print(open(f).read())


if __name__ == "__main__":
    main()
