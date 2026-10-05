# Predicción del diagnóstico de cáncer de tiroides — Proyecto Integrador (TP1)

**Curso:** CC209 – Data Mining Tools (UPC) · **Docente:** Carlos Fernando Montoya Cubas · **NRC:** 17496 · **Grupo 2**
**Integrantes:** Gomez Rubina, Luis David (U20221C621) · Miranda Cárdenas, Sofía (U20191C439)

## Problema
¿Con qué capacidad se puede distinguir un diagnóstico **maligno** de uno **benigno** a partir de datos demográficos,
antecedentes, perfil hormonal (TSH, T3, T4) y tamaño del nódulo, y qué factores aportan esa capacidad?

- **Tipo:** clasificación binaria supervisada (`Diagnosis`: Benign / Malignant), clases desbalanceadas (23.3 % malignos).
- **Uso previsto:** herramienta de priorización (triaje) para estudios confirmatorios, no de diagnóstico.

## Datos
- **Fuente:** Kaggle – *Thyroid Cancer Risk Prediction Dataset* (`thyroid_cancer_risk_data.csv`).
  - URL: `[COMPLETAR con el enlace exacto de Kaggle]`
  - Licencia: `[COMPLETAR según la página del dataset]`
- 212 691 pacientes × 17 variables. Sin faltantes ni duplicados.
- **Limitación principal:** el dataset presenta señales claras de ser sintético (distribuciones uniformes, hormonas sin
  correlación entre sí, misma composición étnica en todos los países). Los resultados no deben extrapolarse a pacientes reales.

## Resultados TP1 (conjunto de validación)
| Modelo | Recall (Malig.) | Precisión | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Dummy estratificado | 0.226 | 0.227 | 0.227 | 0.496 | 0.231 |
| Regla dominio (Risk = High) | 0.453 | 0.701 | 0.550 | 0.701 | 0.465 |
| Regresión Logística | 0.615 | 0.352 | 0.448 | 0.670 | 0.435 |
| Random Forest | 0.453 | 0.700 | 0.550 | 0.697 | 0.499 |
| HistGradientBoosting | 0.453 | 0.700 | 0.550 | 0.700 | 0.500 |

El conjunto de **test (15 %) no se usó en el TP1**: se reserva para la evaluación final del TF1.

## Estructura
```
tp1_thyroid/
├── data/
│   ├── raw/                  # CSV original (no se modifica)
│   └── processed/            # (reservado para TF1)
├── notebooks/
│   ├── 01_TP1_thyroid.ipynb  # Notebook ejecutado del TP1
│   └── build_notebook.py     # Script que genera el notebook
├── src/
│   ├── config.py             # Rutas, semilla, columnas, rangos válidos
│   ├── data.py               # Carga, reporte de calidad, limpieza, split 70/15/15
│   ├── features.py           # ColumnTransformer (numéricas / categóricas)
│   ├── models.py             # Baseline y modelos preliminares (Pipelines)
│   ├── evaluate.py           # Métricas
│   └── train.py              # Entrena, evalúa en validación y guarda modelos
├── models/                   # Pipelines entrenados (.joblib)
├── reports/
│   ├── figures/              # Gráficos generados por el notebook
│   └── metricas_validacion.csv
├── requirements.txt
└── README.md
```

## Cómo ejecutar
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Opción 1: notebook completo
jupyter notebook notebooks/01_TP1_thyroid.ipynb

# Opción 2: solo entrenamiento y métricas (desde la raíz del repositorio)
python -m src.train
```
Todo es reproducible con `RANDOM_STATE = 42` (en `src/config.py`).

## Decisiones clave
- **Objetivo:** `Diagnosis`. `Thyroid_Cancer_Risk` se **excluye** como predictor (puntaje precalculado de origen desconocido,
  redundante y con posible fuga de información) y se usa como **regla baseline del dominio**.
- **Split:** estratificado 70/15/15 + CV estratificada de 5 folds sobre train.
- **Preprocesamiento:** dentro de `Pipeline`/`ColumnTransformer` y ajustado solo con train.
- **Métricas:** recall, precisión y F1 de *Malignant*, ROC-AUC y PR-AUC (no accuracy, por el desbalance).

## Matriz de decisiones de herramientas
| Necesidad | Herramienta elegida | Alternativa considerada | Justificación |
|---|---|---|---|
| EDA | pandas + matplotlib/seaborn + scipy | ydata-profiling | Gráficos orientados a preguntas concretas y pruebas estadísticas (χ², Mann-Whitney, Cramér's V) en lugar de un reporte automático. |
| Preparación | scikit-learn `Pipeline` + `ColumnTransformer` | Transformaciones manuales con pandas | Garantiza que las transformaciones se ajusten solo con train (evita leakage) y que sean reproducibles en inferencia. |
| Modelamiento | Regresión Logística, Random Forest, HistGradientBoosting | XGBoost / LightGBM | HGB ofrece boosting eficiente sin dependencias extra; LR aporta un modelo interpretable de referencia. |
| Experimentación | `StratifiedKFold` (TP1) → Optuna + MLflow (TF1) | GridSearchCV | Optuna explora mejor el espacio de hiperparámetros; MLflow registra configuración y métricas. |
| Interpretabilidad | Coeficientes LR (TP1) → SHAP (TF1) | Permutation importance | SHAP permite explicaciones globales y locales por paciente. |
| Despliegue | Streamlit (TF1) | FastAPI | Interfaz simple para demostrar el uso del modelo fuera del notebook. |

## Uso de IA generativa
Se utilizó un asistente de IA (Claude) como apoyo para organizar el código, la estructura del repositorio y la redacción.
El grupo revisó, ejecutó y validó todos los resultados y es responsable de su interpretación.
