"""Definición del baseline y de los modelos preliminares."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config as C
from .features import build_preprocessor


def make_pipelines() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Baseline (Dummy estratificado)": Pipeline([
            ("prep", build_preprocessor()),
            ("model", DummyClassifier(strategy="stratified", random_state=rs)),
        ]),
        "Regresión Logística": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1, random_state=rs)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=200, class_weight="balanced",
                random_state=rs)),
        ]),
    }
