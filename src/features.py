"""Preprocesamiento reproducible con ColumnTransformer."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Numéricas: imputación por mediana (+ estandarización opcional).
    Categóricas: imputación por moda + One-Hot (drop='if_binary').
    Todo se ajusta SOLO con train al llamar a .fit() del Pipeline completo.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                 sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), C.NUMERIC_COLS),
            ("cat", Pipeline(cat_steps), C.CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
