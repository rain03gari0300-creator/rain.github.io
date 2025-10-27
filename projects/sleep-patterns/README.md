# Análisis de Patrones de Sueño (Sintético)

Proyecto end-to-end para clasificar patrones de sueño (Normal, Insomnia, Apnea) con dataset sintético realista. Incluye:
- Generación de datos (`src/generate_synthetic.py`)
- Entrenamiento y evaluación con múltiples modelos (`src/train.py`)
- Métricas: Precision, Recall (Sensitivity), F1, Specificity y Matriz de Confusión
- Artefactos (`artifacts/`): `best_model.pkl`, `metadata.json`, `metrics.csv`, `confusion_matrix.png`, `error_samples.csv`
- UI en Streamlit y Gradio (`app/`)

## 1) Instalar dependencias
```bash
pip install -r projects/sleep-patterns/requirements.txt
```

## 2) Generar datos sintéticos
```bash
python -m projects.sleep-patterns.src.generate_synthetic
# genera projects/sleep-patterns/data/sleep_synthetic.csv
```

## 3) Entrenar modelos
```bash
python -m projects/sleep-patterns.src.train \
  --input projects/sleep-patterns/data/sleep_synthetic.csv \
  --target label \
  --cv 5 \
  --outdir projects/sleep-patterns/artifacts
```
- Modelos evaluados: LogisticRegression, RandomForest, GradientBoosting, SVC, KNN
- Selección por `f1_macro` (puedes cambiar con `--scoring`)

## 4) Ejecutar UI
Streamlit:
```bash
streamlit run projects/sleep-patterns/app/app_streamlit.py
```

Gradio:
```bash
python projects/sleep-patterns/app/app_gradio.py
```

## 5) Análisis de error
Revisa `projects/sleep-patterns/artifacts/error_samples.csv` para entender dónde falla el mejor modelo (falsos positivos/negativos).

## Notas
- Puedes reemplazar el dataset sintético por tu CSV/Excel real y re-entrenar con `--input` y `--sheet` (si es Excel).
- Las categorías disponibles para la UI se infieren y se guardan en `metadata.json`.