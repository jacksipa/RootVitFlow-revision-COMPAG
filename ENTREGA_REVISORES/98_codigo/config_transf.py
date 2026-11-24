"""
config_transf.py (Prueba3)

Configuración para la experimentación de la revisión COMPAG-D-26-05203R1.
Reproduce EXACTAMENTE los hiperparámetros de Prueba1/config_transf.py (modelo
principal root_cvt_model97.py), pero con rutas locales relativas a esta carpeta.

root_cvt_model97.py hace `import config_transf as cfg`; como Prueba3 va primero
en sys.path, ese import resuelve a este archivo y el modelo original se usa sin
modificarlo.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRANSF_DIR = os.path.dirname(PROJECT_ROOT)                      # .../Transf_PRO_RiegoSequia
PRUEBA1_DIR = os.path.join(TRANSF_DIR, "Prueba1")                # modelo principal (no se modifica)
RESPALDO_ROOT = os.path.abspath(os.path.join(TRANSF_DIR, "..", "..", ".."))  # .../RootVitFlow_Respaldo

# Máscaras binarias usadas para la clasificación (misma carpeta 'allbin' del modelo principal)
IMG_DIR = os.path.join(RESPALDO_ROOT, "allbin")

# Imágenes originales y CSV morfométrico
ORIG_IMG_DIR = os.path.join(RESPALDO_ROOT, "RootVitFlow", "datasets", "Original")
MORPHO_CSV = os.path.join(RESPALDO_ROOT, "RootVitFlow", "Arquitectures", "Size_Root", "root_size_labels.csv")

# Partición original de Prueba1 (para verificar que la carga de datos es idéntica)
ORIG_BALANCED_DIR = os.path.join(PRUEBA1_DIR, "Balanced_partition_2")

# Requeridos por root_cvt_model97 / data_roots_transf al importarse
BALANCED_DIR = os.path.join(PROJECT_ROOT, "cache")
X_TRAIN_NPY = os.path.join(BALANCED_DIR, "X_Train.npy")
Y_TRAIN_NPY = os.path.join(BALANCED_DIR, "y_Train.npy")
X_TEST_NPY = os.path.join(BALANCED_DIR, "X_Test.npy")
Y_TEST_NPY = os.path.join(BALANCED_DIR, "y_Test.npy")

# ========= PARÁMETROS DE ENTRENAMIENTO (idénticos a Prueba1) =========
IMG_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 150
INITIAL_LR = 3e-4
RANDOM_STATE = 42

# Parámetros del Transformer (idénticos a Prueba1)
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4

# ========= SALIDAS =========
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")

for _d in (BALANCED_DIR, LOGS_DIR, RESULTS_DIR, SPLITS_DIR):
    os.makedirs(_d, exist_ok=True)

MODEL_NAME = "RootCVT_RiegoSequia"
