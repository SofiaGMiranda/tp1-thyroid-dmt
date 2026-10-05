"""Entrena baseline y modelos preliminares, evalúa en validación y guarda artefactos.

Uso (desde la raíz del repositorio):
    python -m src.train
"""
import unicodedata

import joblib

from . import config as C
from .data import clean, get_xy, load_raw, split
from .evaluate import evaluate, results_table
from .models import make_pipelines


def main():
    df = clean(load_raw())
    X, y = get_xy(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    print(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)} (reservado TF1)")

    results = {}
    for name, pipe in make_pipelines().items():
        pipe.fit(X_train, y_train)
        score = pipe.predict_proba(X_val)[:, 1]
        results[name] = evaluate(y_val, pipe.predict(X_val), score)
        if name != "Baseline (Dummy estratificado)":
            fname = unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode().replace(" ", "_")
            joblib.dump(pipe, C.MODELS_DIR / f"{fname}.joblib")

    table = results_table(results)
    C.REPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(C.REPORTS_DIR / "metricas_validacion.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
