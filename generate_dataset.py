"""Generate a structured, synthetic allergy dataset (allergy_data.csv).

The data is SIMULATED for demonstration: risk factors drive a latent allergy
risk, which in turn drives symptoms and the final Allergy_Status label.
Replace allergy_data.csv with real clinical data (same columns) when available.

Usage:  python generate_dataset.py
"""
import numpy as np
import pandas as pd

N = 3000
SEED = 42


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(30, 14, n).round(), 3, 80).astype(int)
    gender = rng.choice(["Male", "Female"], n)
    city_type = rng.choice(["Urban", "Semi-Urban", "Rural"], n, p=[0.5, 0.3, 0.2])
    season = rng.choice(["Spring", "Summer", "Monsoon", "Autumn", "Winter"], n)
    family_history = rng.binomial(1, 0.30, n)
    asthma = rng.binomial(1, 0.15, n)
    eczema = rng.binomial(1, 0.12, n)
    smoker = np.where(age < 18, 0, rng.binomial(1, 0.18, n))
    pet_exposure = rng.binomial(1, 0.35, n)

    season_pollen = {"Spring": 2.5, "Summer": 1.0, "Monsoon": 0.5, "Autumn": 1.5, "Winter": 0.0}
    pollen = np.clip(rng.normal(4, 2, n) + np.vectorize(season_pollen.get)(season), 0, 10).round(1)
    dust = np.clip(rng.normal(4.5, 2.2, n) + (city_type == "Urban") * 1.2, 0, 10).round(1)

    base_temp = {"Spring": 28, "Summer": 38, "Monsoon": 30, "Autumn": 29, "Winter": 15}
    temp = (np.vectorize(base_temp.get)(season) + rng.normal(0, 3, n)).round(1)
    humidity = np.clip(
        rng.normal(45, 12, n) + (season == "Monsoon") * 30 - (season == "Summer") * 5, 10, 98
    ).round(0).astype(int)
    aqi = np.clip(
        rng.normal(110, 45, n) + (city_type == "Urban") * 50 + (season == "Winter") * 40, 20, 450
    ).round(0).astype(int)

    # latent allergy risk
    z = (
        -3.6
        + 1.1 * family_history
        + 0.9 * asthma
        + 0.8 * eczema
        + 0.28 * pollen
        + 0.20 * dust
        + 0.5 * pet_exposure
        + 0.4 * smoker
        + 0.006 * (aqi - 100)
        + 0.012 * (humidity - 50)
        - 0.012 * (age - 30)
        + rng.normal(0, 0.8, n)
    )
    p_allergic = 1 / (1 + np.exp(-z))
    allergic = rng.binomial(1, p_allergic)

    def symptom(base, boost):
        return rng.binomial(1, np.clip(base + boost * allergic, 0, 0.97))

    sneezing = symptom(0.10, 0.60)
    itchy_eyes = symptom(0.08, 0.50)
    nasal_congestion = symptom(0.12, 0.50)
    skin_rash = symptom(0.05 + 0.15 * eczema, 0.35)
    wheezing = symptom(0.03 + 0.20 * asthma, 0.30)
    symptom_count = sneezing + itchy_eyes + nasal_congestion + skin_rash + wheezing

    triggers = ["Pollen", "Dust Mites", "Pet Dander", "Food", "Smoke/Pollution", "Mold"]
    trigger_weights = np.column_stack([
        pollen / 10 + 0.2, dust / 10 + 0.2, pet_exposure * 0.8 + 0.1,
        np.full(n, 0.3), aqi / 450 + smoker * 0.3 + 0.1, (humidity / 100) + 0.1,
    ])
    trigger_weights = trigger_weights / trigger_weights.sum(axis=1, keepdims=True)
    primary_trigger = np.array([rng.choice(triggers, p=w) for w in trigger_weights])
    primary_trigger = np.where(allergic == 1, primary_trigger, "None")

    df = pd.DataFrame({
        "Patient_ID": [f"P{str(i + 1).zfill(4)}" for i in range(n)],
        "Age": age,
        "Gender": gender,
        "City_Type": city_type,
        "Season": season,
        "Family_History": family_history,
        "Asthma": asthma,
        "Eczema": eczema,
        "Smoker": smoker,
        "Pet_Exposure": pet_exposure,
        "Pollen_Level": pollen,
        "Dust_Level": dust,
        "Temperature_C": temp,
        "Humidity_Pct": humidity,
        "AQI": aqi,
        "Sneezing": sneezing,
        "Itchy_Eyes": itchy_eyes,
        "Nasal_Congestion": nasal_congestion,
        "Skin_Rash": skin_rash,
        "Wheezing": wheezing,
        "Symptom_Count": symptom_count,
        "Primary_Trigger": primary_trigger,
        "Allergy_Status": np.where(allergic == 1, "Allergic", "Not Allergic"),
    })
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv("allergy_data.csv", index=False)
    print(f"Saved allergy_data.csv: {data.shape[0]} rows, {data.shape[1]} columns")
    print(data["Allergy_Status"].value_counts(normalize=True).round(3))
