"""Carga, validación de calidad y separación de datos."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw(path=C.DATA_RAW) -> pd.DataFrame:
    """Carga el CSV original sin modificarlo."""
    return pd.read_csv(path)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por columna: tipo, faltantes, únicos y valores fuera de rango."""
    rows = []
    for col in df.columns:
        s = df[col]
        out_of_range = None
        if col in C.VALID_RANGES:
            lo, hi = C.VALID_RANGES[col]
            out_of_range = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "columna": col,
            "tipo": str(s.dtype),
            "faltantes": int(s.isna().sum()),
            "unicos": int(s.nunique()),
            "fuera_de_rango": out_of_range,
        })
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza determinista y sin parámetros aprendidos (no genera fuga):
    - elimina espacios sobrantes en categóricas (se verificó que no hay
      variantes de escritura; no se cambia mayúsculas para no alterar 'UK'/'USA'),
    - elimina duplicados exactos (ignorando el ID),
    - descarta filas que violan rangos lógicos.
    Las transformaciones con parámetros (escalado, codificación) van en el Pipeline.
    """
    df = df.copy()
    for col in C.CATEGORICAL_COLS + [C.TARGET, "Thyroid_Cancer_Risk"]:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != C.ID_COL])
    for col, (lo, hi) in C.VALID_RANGES.items():
        df = df[df[col].between(lo, hi)]
    return df.reset_index(drop=True)


def get_xy(df: pd.DataFrame):
    X = df[C.FEATURES].copy()
    y = (df[C.TARGET] == C.POSITIVE_CLASS).astype(int)
    return X, y


def split(X, y, random_state=C.RANDOM_STATE):
    """Separación estratificada 70/15/15. El test no se usa hasta el TF1."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=random_state)
    val_rel = C.VAL_SIZE / (1 - C.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
