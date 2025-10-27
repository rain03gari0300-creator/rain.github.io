import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from .preprocess import infer_feature_types, build_preprocessor
from .metrics import compute_metrics, metrics_df, to_table_row

def load_data(input_path: str, sheet: str | None) -> pd.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        import pandas as pd
        return pd.read_excel(p, sheet_name=sheet)
    return pd.read_csv(p)

def build_models(names: List[str]):
    all_models = {
        "lr": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "svc": SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=7),
    }
    return {k: v for k, v in all_models.items() if k in names}

def plot_confusion(cm: np.ndarray, labels: List, out_png: Path):
    plt.figure(figsize=(5,4), dpi=140)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ruta a CSV/Excel")
    ap.add_argument("--sheet", default=None, help="Hoja si es Excel")
    ap.add_argument("--target", required=True, help="Columna objetivo")
    ap.add_argument("--models", nargs="+", default=["lr","rf","gb","svc","knn"])
    ap.add_argument("--scoring", default="f1_macro")
    ap.add_argument("--cv", type=int, default=5, help="n_splits para CV")
    ap.add_argument("--no-cv", action="store_true", help="Evalúa directo en train")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    outdir = Path(__file__).resolve().parents[2] / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input, args.sheet)
    if args.target not in df.columns:
        raise ValueError(f"La columna objetivo {args.target} no existe.")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    numeric_features, categorical_features = infer_feature_types(df, args.target)
    pre = build_preprocessor(numeric_features, categorical_features)
    models = build_models(args.models)

    labels = sorted(pd.Series(y).dropna().unique().tolist())
    metrics_rows = []
    best_name, best_score = None, -1.0
    best_pipe, best_metrics = None, None

    def fit_on_train(pipe, X_tr, y_tr):
        pipe.fit(X_tr, y_tr)
        return pipe

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        try:
            cls_counts = pd.Series(y).value_counts().min()
            if args.no_cv or cls_counts < args.cv:
                pipe = fit_on_train(pipe, X, y)
                y_pred = pipe.predict(X)
            else:
                skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
                oof_pred = np.empty(len(y), dtype=object)
                for tr_idx, va_idx in skf.split(X, y):
                    pipe_cv = Pipeline(steps=[("preprocess", pre), ("model", model)])
                    pipe_cv.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                    oof_pred[va_idx] = pipe_cv.predict(X.iloc[va_idx])
                y_pred = oof_pred
        except Exception:
            pipe = fit_on_train(pipe, X, y)
            y_pred = pipe.predict(X)

        m = compute_metrics(y_true=y, y_pred=y_pred, labels=labels)
        metrics_rows.append(to_table_row(name, m))
        score = m.get(args.scoring, -1.0)
        if score >= best_score:
            best_score, best_name, best_pipe, best_metrics = score, name, pipe, m

    best_pipe.fit(X, y)
    import json as _json
    joblib.dump(best_pipe, outdir / "best_model.pkl")
    meta: Dict = {
        "target_column": args.target,
        "labels": [str(l) for l in labels],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_model_name": best_name,
        "scoring": args.scoring,
        "categorical_choices": {
            c: sorted([str(v) for v in X[c].dropna().unique().tolist()][:24]) for c in categorical_features
        }
    }
    (outdir / "metadata.json").write_text(_json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    dfm = metrics_df(metrics_rows)
    dfm.to_csv(outdir / "metrics.csv", index=False)

    cm = confusion_matrix(y, best_pipe.predict(X), labels=labels)
    plot_confusion(cm, labels, outdir / "confusion_matrix.png")

    y_hat = best_pipe.predict(X)
    err_mask = (pd.Series(y_hat) != pd.Series(y)).values
    err_df = df.loc[err_mask].copy()
    err_df["__y_true__"] = y.values[err_mask]
    err_df["__y_pred__"] = y_hat[err_mask]
    if not err_df.empty:
        err_df.to_csv(outdir / "error_samples.csv", index=False)

    print(f"Mejor modelo: {best_name} ({args.scoring}={best_score:.4f})")
    print(f"Artefactos: {outdir.resolve()}")

if __name__ == "__main__":
    main()