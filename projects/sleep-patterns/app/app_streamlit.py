import json
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
MODEL = ART / "best_model.pkl"
META = ART / "metadata.json"

st.set_page_config(page_title="Sleep Pattern Classifier", layout="centered")
st.title("Clasificador de Patrones de Sueño")

if not MODEL.exists() or not META.exists():
    st.error("No encuentro los artefactos del modelo. Primero ejecuta el entrenamiento.")
    st.stop()

pipe = joblib.load(MODEL)
meta = json.loads(META.read_text(encoding="utf-8"))

target = meta["target_column"]
labels = meta["labels"]
num_cols = meta["numeric_features"]
cat_cols = meta["categorical_features"]
choices = meta.get("categorical_choices", {})
feat_cols = num_cols + cat_cols

st.write(f"Modelo: {meta.get('best_model_name','N/A')} — Métrica: {meta.get('scoring','f1_macro')}")

st.sidebar.header("Entrada de datos")
row = {}
for c in num_cols:
    row[c] = st.sidebar.number_input(c, value=0.0)
for c in cat_cols:
    opts = choices.get(c)
    if opts:
        row[c] = st.sidebar.selectbox(c, options=opts, index=0)
    else:
        row[c] = st.sidebar.text_input(c, value="")

if st.sidebar.button("Predecir"):
    X = pd.DataFrame([row], columns=feat_cols)
    pred = pipe.predict(X)[0]
    st.subheader("Predicción")
    st.write(f"{target}: {pred}")
    try:
        proba = pipe.predict_proba(X)[0]
        st.subheader("Probabilidades")
        st.json({str(lbl): float(p) for lbl, p in zip(labels, proba)})
    except Exception:
        st.info("El modelo no expone predict_proba().")