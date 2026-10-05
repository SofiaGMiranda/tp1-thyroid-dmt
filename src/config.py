"""Configuración central del proyecto (rutas, semilla, columnas)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "thyroid_cancer_risk_data.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

# Proporciones de separación: 70 % train, 15 % validation, 15 % test
TEST_SIZE = 0.15
VAL_SIZE = 0.15

TARGET = "Diagnosis"
POSITIVE_CLASS = "Malignant"

ID_COL = "Patient_ID"
# Excluida del modelo principal: es un puntaje de riesgo precalculado a partir
# de los mismos factores y de origen desconocido (riesgo de fuga de información).
EXCLUDED_COLS = [ID_COL, "Thyroid_Cancer_Risk"]

NUMERIC_COLS = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]
BINARY_COLS = [
    "Gender", "Family_History", "Radiation_Exposure", "Iodine_Deficiency",
    "Smoking", "Obesity", "Diabetes",
]
NOMINAL_COLS = ["Country", "Ethnicity"]
CATEGORICAL_COLS = BINARY_COLS + NOMINAL_COLS
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# Rangos válidos (reglas lógicas / clínicas) usados en la validación de calidad
VALID_RANGES = {
    "Age": (0, 120),
    "TSH_Level": (0, 100),   # mIU/L
    "T3_Level": (0, 10),     # nmol/L aprox.
    "T4_Level": (0, 30),     # µg/dL aprox.
    "Nodule_Size": (0, 10),  # cm
}
