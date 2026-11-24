"""
common.py (Prueba3)

Utilidades compartidas para la experimentación de revisión:
  - carga de datos idéntica a Prueba1/data_roots_transf.py + metadatos (planta, fecha)
  - construcción de modelos: RootCVT (root_cvt_model97.py SIN modificar), ablaciones y CNN baseline
  - protocolo de entrenamiento único para todos los modelos (mismo que root_cvt_model97.train,
    pero con validación separada del test)
  - métricas

IMPORTANTE: ejecutar con `source env.sh` (TF_USE_LEGACY_KERAS=1 debe estar antes de importar TF).
"""
import os
import re
import sys
import json
import time
import importlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Prueba3 primero → `import config_transf` resuelve al config local
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config_transf as cfg  # noqa: E402

CLASS_NAMES = ["Riego", "Sequia"]
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Fechas mal escritas en el nombre de archivo (19_07_95, 30_07_35) → año 25
DATE_FIX = {"19_07_95": "19_07_25", "30_07_35": "30_07_25"}
NAME_RE = re.compile(r"^(?P<plant>[rs]b\d+)_(?P<date>\d{2}_\d{2}_\d{2})", re.I)


# ======================================================================
# DATOS
# ======================================================================
def parse_name(fname):
    m = NAME_RE.match(fname)
    if m is None:
        raise ValueError(f"Nombre no reconocido: {fname}")
    plant = m.group("plant").lower()
    date = DATE_FIX.get(m.group("date"), m.group("date"))
    d, mth, yy = date.split("_")
    return plant, f"20{yy}-{mth}-{d}"


def load_dataset(cache=True):
    """
    Replica Prueba1/data_roots_transf.load_images_from_single_folder():
    sorted(listdir(allbin)) → cv2.imread GRAYSCALE → cv2.resize(224) → (224,224,1) uint8, r=0 / s=1.
    Devuelve X, y y un DataFrame de metadatos (fname, plant, date, day_index, label).
    """
    cache_x = os.path.join(cfg.BALANCED_DIR, "X_all.npy")
    cache_m = os.path.join(cfg.BALANCED_DIR, "meta_all.csv")
    if cache and os.path.exists(cache_x) and os.path.exists(cache_m):
        X = np.load(cache_x)
        meta = pd.read_csv(cache_m)
        return X, meta["label"].values.astype(int), meta

    import cv2
    X, rows = [], []
    for fname in sorted(os.listdir(cfg.IMG_DIR)):
        if not fname.lower().endswith(VALID_EXTS):
            continue
        img = cv2.imread(os.path.join(cfg.IMG_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        img = np.expand_dims(img, axis=-1)
        label = 0 if fname.lower().startswith("r") else 1 if fname.lower().startswith("s") else None
        if label is None:
            continue
        plant, date = parse_name(fname)
        X.append(img)
        rows.append({"fname": fname, "plant": plant, "date": date, "label": label})

    X = np.array(X)
    meta = pd.DataFrame(rows)
    d0 = pd.to_datetime(meta["date"]).min()
    meta["day_index"] = (pd.to_datetime(meta["date"]) - d0).dt.days
    meta["idx"] = np.arange(len(meta))
    if cache:
        np.save(cache_x, X)
        meta.to_csv(cache_m, index=False)
    return X, meta["label"].values.astype(int), meta


def to_onehot(y):
    return np.eye(cfg.NUM_CLASSES, dtype=np.float32)[np.asarray(y, int)]


# ======================================================================
# MODELOS
# ======================================================================
_MOD97 = None


def get_mod97():
    """Importa Prueba1/root_cvt_model97.py sin modificarlo."""
    global _MOD97
    if _MOD97 is None:
        if cfg.PRUEBA1_DIR not in sys.path:
            sys.path.append(cfg.PRUEBA1_DIR)  # después de Prueba3 → config_transf local
        _MOD97 = importlib.import_module("root_cvt_model97")
        assert os.path.dirname(os.path.abspath(_MOD97.__file__)) == os.path.abspath(cfg.PRUEBA1_DIR)
    return _MOD97


def _compile_like_rootcvt(model):
    """Mismo compile que root_cvt_model97.new_archv()."""
    import tensorflow as tf
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.INITIAL_LR, weight_decay=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return model


def _rootcvt_head(representation, mod):
    """Cabeza de clasificación idéntica a root_cvt_model97.new_archv()."""
    import tensorflow as tf
    z = tf.keras.layers.Dense(128, kernel_initializer="glorot_normal",
                              kernel_regularizer=tf.keras.regularizers.l1_l2(1e-5))(representation)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation("relu")(z)
    z = tf.keras.layers.Dropout(0.5)(z)
    return mod.Dense(cfg.NUM_CLASSES, activation="softmax",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="output_1")(z)


def _rootcvt_backbone(x, mod):
    """Backbone CNN+SE idéntico a root_cvt_model97.new_archv()."""
    z = mod.Block_3(x, 64)
    z = mod.Block_2(z, 64)
    z = mod.Block_1(z, 64)
    z = mod.Block_1(z, 64)
    z = mod.Block_2(z, 64)
    z = mod.Block_3(z, 64)
    z = mod.Block_2(z, 64)
    z = mod.Block_3(z, 128)
    z = mod.Block_4(z, 128)
    return z


def build_rootcvt(patch_size=11):
    """RootCVT principal (patch 11 → 2x2 = 4 tokens). patch_size distinto = ablación de nº de tokens."""
    mod = get_mod97()
    old = mod.PATCH_SIZE
    mod.PATCH_SIZE = patch_size
    try:
        model = mod.new_archv()  # arquitectura + compile originales
    finally:
        mod.PATCH_SIZE = old
    return model


def build_rootcvt_no_augmentation(builder):
    """Variante de sensibilidad: misma arquitectura sin la capa data_augmentation (identidad)."""
    import tensorflow as tf
    mod = get_mod97()
    orig = mod.data_augmentation
    mod.data_augmentation = tf.keras.layers.Activation("linear", name="no_augmentation")
    try:
        return builder()
    finally:
        mod.data_augmentation = orig


def build_rootcvt_7blocks():
    """
    Variante con la lista de bloques del checkpoint de Prueba1 que dio 100% en test
    (logs_2/RootCVT_RiegoSequia_2026-02-20_03-52-06/saved-model-119): B3-B2-B1-B3-B2-B3(128)-B4(128).
    root_cvt_model97.py se editó después de ese entrenamiento (tiene 9 bloques).
    """
    import tensorflow as tf
    mod = get_mod97()
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 1), name="input_1")
    x = mod.data_augmentation(inputs)
    z = mod.Block_3(x, 64)
    z = mod.Block_2(z, 64)
    z = mod.Block_1(z, 64)
    z = mod.Block_3(z, 64)
    z = mod.Block_2(z, 64)
    z = mod.Block_3(z, 128)
    z = mod.Block_4(z, 128)
    cvt = mod.Transformer(z)
    rep = tf.keras.layers.LayerNormalization(epsilon=mod.LAYER_NORM_EPS)(cvt)
    rep = mod.AttentionPooling1D(units=128)(rep)
    out = _rootcvt_head(rep, mod)
    return _compile_like_rootcvt(tf.keras.Model(inputs, out))


def build_rootcvt_no_transformer():
    """Ablación: mismo backbone CNN+SE y misma cabeza, sin bloque Transformer (GAP)."""
    import tensorflow as tf
    mod = get_mod97()
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 1), name="input_1")
    x = mod.data_augmentation(inputs)
    z = _rootcvt_backbone(x, mod)
    rep = tf.keras.layers.GlobalAveragePooling2D()(z)
    out = _rootcvt_head(rep, mod)
    return _compile_like_rootcvt(tf.keras.Model(inputs, out))


CNN_ZOO = {
    "mobilenetv2": ("MobileNetV2", "mobilenet_v2"),
    "inceptionv3": ("InceptionV3", "inception_v3"),
    "resnet50": ("ResNet50", "resnet"),
    "densenet201": ("DenseNet201", "densenet"),
    "vgg16": ("VGG16", "vgg16"),
}


def build_cnn_baseline(name):
    """
    CNN preentrenada en ImageNet con el MISMO protocolo que RootCVT:
    misma entrada (máscara binaria 224x224x1, replicada a 3 canales), mismo data augmentation,
    misma cabeza, misma pérdida/optimizador/LR/épocas. Fine-tuning completo.
    """
    import tensorflow as tf
    mod = get_mod97()
    cls_name, pp_mod = CNN_ZOO[name]
    tf.keras.backend.clear_session()
    app = getattr(tf.keras.applications, cls_name)
    pp = getattr(tf.keras.applications, pp_mod).preprocess_input
    inputs = tf.keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 1), name="input_1")
    x = mod.data_augmentation(inputs)
    x = tf.keras.layers.Concatenate(axis=-1)([x, x, x])
    x = tf.keras.layers.Lambda(pp, name="preprocess")(x)
    base = app(include_top=False, weights="imagenet", input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
    base.trainable = True
    z = base(x)
    rep = tf.keras.layers.GlobalAveragePooling2D()(z)
    out = _rootcvt_head(rep, mod)
    return _compile_like_rootcvt(tf.keras.Model(inputs, out, name=f"{cls_name}_root"))


MODEL_BUILDERS = {
    "rootcvt": lambda: build_rootcvt(11),
    "rootcvt_p4": lambda: build_rootcvt(4),      # 7x7 = 49 tokens
    "rootcvt_p2": lambda: build_rootcvt(2),      # 14x14 = 196 tokens
    "rootcvt_notransf": build_rootcvt_no_transformer,
    "rootcvt_7b": build_rootcvt_7blocks,
    "rootcvt_noaug": lambda: build_rootcvt_no_augmentation(lambda: build_rootcvt(11)),
}
for _k in CNN_ZOO:
    MODEL_BUILDERS[_k] = (lambda k: (lambda: build_cnn_baseline(k)))(_k)


def build_model(name):
    return MODEL_BUILDERS[name]()


# ======================================================================
# ENTRENAMIENTO (protocolo único)
# ======================================================================
def setup_gpu():
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def train_and_predict(model_name, X_tr, y_tr, X_va, y_va, X_te, seed, out_dir,
                      epochs=None, batch_size=None, save_weights=True, reduce_lr=False):
    """
    Protocolo de la corrida que produjo el resultado reportado (Prueba1/logs_2/
    RootCVT_RiegoSequia_2026-02-20_03-52-06): batch 16, 150 épocas, LR CONSTANTE 3e-4
    (esa corrida no registra ninguna reducción de LR ni 'lr' en TensorBoard), class_weight {0:2,1:1},
    checkpoint del mejor val_accuracy.
    reduce_lr=True activa el ReduceLROnPlateau(val_accuracy, 0.2, 10, 1e-6) que tiene hoy
    root_cvt_model97.train() (sólo para análisis de sensibilidad: con validaciones pequeñas
    lleva el LR a 1e-6 en ~50 épocas y el modelo deja de aprender).
    Corrección metodológica: la validación es un conjunto de plantas/imágenes SEPARADO del test;
    el test sólo se usa una vez, con los pesos del mejor epoch en validación.
    """
    import tensorflow as tf
    epochs = epochs or cfg.EPOCHS
    batch_size = batch_size or cfg.BATCH_SIZE
    os.makedirs(out_dir, exist_ok=True)

    tf.keras.utils.set_random_seed(seed)
    model = build_model(model_name)

    ckpt_path = os.path.join(out_dir, "best.weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True,
                                           save_weights_only=True, mode="max", verbose=0),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv")),
    ]
    if reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.2,
                                                              patience=10, min_lr=1e-6, verbose=0))
    t0 = time.time()
    hist = model.fit(X_tr, to_onehot(y_tr), epochs=epochs, batch_size=batch_size,
                     validation_data=(X_va, to_onehot(y_va)), class_weight={0: 2.0, 1: 1.0},
                     callbacks=callbacks, verbose=0)
    train_time = time.time() - t0

    model.load_weights(ckpt_path)
    p_te = model.predict(X_te, batch_size=32, verbose=0)[:, 1]
    p_va = model.predict(X_va, batch_size=32, verbose=0)[:, 1]
    va_acc = hist.history["val_accuracy"]
    info = {
        "model": model_name, "seed": seed, "epochs": epochs,
        "reduce_lr": bool(reduce_lr),
        "best_epoch": int(np.argmax(va_acc) + 1), "best_val_acc": float(np.max(va_acc)),
        "final_train_acc": float(hist.history["accuracy"][-1]),
        "train_time_s": round(train_time, 1), "n_params": int(model.count_params()),
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(X_te)),
    }
    if not save_weights:
        os.remove(ckpt_path)
    tf.keras.backend.clear_session()
    return p_te, p_va, info


# ======================================================================
# MÉTRICAS
# ======================================================================
def compute_metrics(y_true, p, plants=None):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score, confusion_matrix)
    y_true = np.asarray(y_true, int)
    p = np.asarray(p, float)
    y_hat = (p >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    out = {
        "accuracy": accuracy_score(y_true, y_hat),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_hat),
        "precision_macro": precision_score(y_true, y_hat, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_hat, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_hat, average="macro", zero_division=0),
        "recall_riego": cm[0, 0] / max(cm[0].sum(), 1),
        "recall_sequia": cm[1, 1] / max(cm[1].sum(), 1),
        "auc": roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    if plants is not None:
        df = pd.DataFrame({"plant": plants, "y": y_true, "p": p})
        g = df.groupby("plant").agg(y=("y", "first"), p=("p", "mean"))
        out["plant_accuracy"] = float(((g["p"] >= 0.5).astype(int) == g["y"]).mean())
        out["n_plants"] = int(len(g))
    return out


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
