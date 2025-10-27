import json
from pathlib import Path
import gradio as gr
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
MODEL = ART / "best_model.pkl"
META = ART / "metadata.json"

pipe = joblib.load(MODEL)
meta = json.loads(META.read_text(encoding="utf-8"))

target = meta["target_column"]
labels = meta["labels"]
num_cols = meta["numeric_features"]
cat_cols = meta["categorical_features"]
choices = meta.get("categorical_choices", {})
feat_cols = num_cols + cat_cols

def predict_fn(*vals):
    X = pd.DataFrame([dict(zip(feat_cols, vals))], columns=feat_cols)
    pred = pipe.predict(X)[0]
    out = {"prediction": str(pred)}
    try:
        proba = pipe.predict_proba(X)[0]
        out["probabilities"] = {str(lbl): float(p) for lbl, p in zip(labels, proba)}
    except Exception:
        out["probabilities"] = {}
    return out

inputs = [gr.Number(label=c) for c in num_cols] + [
    gr.Dropdown(choices=choices.get(c), label=c) if choices.get(c) else gr.Textbox(label=c)
    for c in cat_cols
]

demo = gr.Interface(
    fn=predict_fn,
    inputs=inputs,
    outputs=gr.JSON(label="Salida"),
    title="Clasificador de Patrones de Sueño",
    description=f"Predice {target} usando el mejor modelo entrenado."
)

if __name__ == "__main__":
    demo.launch()