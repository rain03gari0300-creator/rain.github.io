from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

GENDERS = ["Male", "Female"]
SHIFT = ["Day", "Night", "Rotating"]
CHRONO = ["Morning", "Evening", "Neutral"]
YN = ["No", "Yes"]

def _clip(x, lo, hi):
    return np.clip(x, lo, hi)

def generate(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = _clip(rng.integers(18, 75, size=n), 18, 90)
    gender = rng.choice(GENDERS, size=n, p=[0.52, 0.48])
    bmi = _clip(rng.normal(26, 5, size=n), 16, 45)
    resting_hr = _clip(rng.normal(72, 10, size=n), 45, 120)

    sleep_hours = _clip(rng.normal(7.0, 1.5, size=n), 3.0, 11.0)
    sleep_quality = _clip(rng.normal(6.5, 2.0, size=n), 1.0, 10.0)  # 1-10
    stress = _clip(rng.normal(5.0, 2.5, size=n), 1.0, 10.0)         # 1-10

    caffeine_mg = _clip(rng.normal(120, 80, size=n), 0, 600)
    alcohol_week = _clip(rng.normal(2.0, 2.0, size=n), 0.0, 25.0)
    screen_time_h = _clip(rng.normal(5.0, 2.0, size=n), 0.5, 14.0)
    activity_days = _clip(rng.integers(0, 7, size=n), 0, 7)

    snoring = rng.choice(YN, size=n, p=[0.7, 0.3])
    apnea_score = _clip(rng.beta(2.0, 5.0, size=n) * 100, 0, 100)

    shift_work = rng.choice(SHIFT, size=n, p=[0.7, 0.15, 0.15])
    weekend = rng.choice(YN, size=n, p=[0.7, 0.3])
    nap_minutes = _clip(rng.normal(20, 25, size=n), 0, 180)
    chronotype = rng.choice(CHRONO, size=n, p=[0.35, 0.35, 0.30])
    smoker = rng.choice(YN, size=n, p=[0.8, 0.2])

    # Scores latentes para clases
    # Insomnia: poco sueño, baja calidad, alto estrés/cafeína/pantalla, turnos nocturnos
    z_ins = (
        (6.5 - sleep_hours) * 0.9
        + (6.0 - sleep_quality) * 0.7
        + (stress - 5.0) * 0.6
        + (caffeine_mg / 200.0) * 0.4
        + (screen_time_h - 4.0) * 0.4
        + (shift_work == "Night") * 0.7
        + (chronotype == "Evening") * 0.3
    )

    # Apnea: BMI alto, ronquidos, alta puntuación de apnea, edad mayor, algo más frecuente en hombres
    z_apnea = (
        (bmi - 27.0) * 0.6
        + (snoring == "Yes") * 1.2
        + (apnea_score / 100.0) * 1.0
        + ((age - 50) / 30.0) * 0.3
        + (gender == "Male") * 0.2
    )

    # Normal: hábitos saludables (más sueño, mejor calidad, actividad, menos alcohol)
    z_norm = (
        (sleep_hours - 7.0) * 0.4
        + (sleep_quality - 6.5) * 0.6
        + (3.0 - alcohol_week) * 0.2
        + (activity_days / 7.0) * 0.3
        - (stress - 5.0) * 0.2
        - (screen_time_h - 4.0) * 0.2
    )

    # Softmax sobre z_norm/z_ins/z_apnea con sesgo de base (~60/25/15)
    logits = np.vstack([
        z_norm + 0.6,
        z_ins + 0.1,
        z_apnea - 0.1
    ]).T
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)

    labels = np.array(["Normal", "Insomnia", "Apnea"])
    y_idx = [rng.choice(3, p=p) for p in probs]
    y = labels[y_idx]

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "resting_hr": resting_hr,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "stress_level": stress,
        "caffeine_mg": caffeine_mg,
        "alcohol_per_week": alcohol_week,
        "screen_time_hours": screen_time_h,
        "activity_days_week": activity_days,
        "snoring": snoring,
        "apnea_risk_score": apnea_score,
        "shift_work": shift_work,
        "weekend": weekend,
        "nap_minutes": nap_minutes,
        "chronotype": chronotype,
        "smoker": smoker,
        "label": y
    })
    return df

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "projects" / "sleep-patterns" / "data"
    out.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out / "sleep_synthetic.csv", index=False)
    print(f"Generated: {out / 'sleep_synthetic.csv'} (shape={df.shape})")