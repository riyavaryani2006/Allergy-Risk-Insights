# Allergy Risk Insights

A machine-learning healthcare analytics app that analyzes allergy-associated factors and predicts
allergy-status patterns. Built with Python, Scikit-learn, Streamlit, Pandas, NumPy and Plotly.

## Features
- **Secure sign-in** – salted PBKDF2-hashed passwords (replaces plaintext `users.json` storage)
- **Overview dashboard** – KPIs, allergy split, trigger ranking, seasonal rates
- **Data Explorer** – filters, distributions, environment scatter plots, correlation heatmap
- **Model Performance** – Logistic Regression vs Random Forest vs Gradient Boosting; accuracy, precision,
  recall, F1, ROC-AUC, 5-fold CV, ROC curves, confusion matrix, permutation feature importance
- **Predict Risk** – patient form returning an allergy probability and Low / Moderate / High band
- **Reports** – downloadable PDF report and CSV exports
- **Bring your own data** – upload a CSV with the same columns

## How to run
1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Optionally, regenerate the synthetic dataset:
   ```bash
   python generate_dataset.py
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Project layout
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (Plotly charts, auth, PDF export) |
| `ml_pipeline.py` | Preprocessing, model training/evaluation, prediction |
| `generate_dataset.py` | Builds the 3,000-record dataset |
| `allergy_data.csv` | Dataset |

## Dataset
`allergy_data.csv` is **synthetic** (3,000 patients, 23 columns): demographics, family history, asthma,
eczema, smoking, pet exposure, pollen / dust / AQI / humidity / temperature, five symptoms, primary
trigger and the target `Allergy_Status`. Because it is simulated, model scores (ROC-AUC ≈ 0.97) are
higher than real clinical data would give. Swap in real, consented, anonymized data with the same
columns for genuine use.

`Primary_Trigger` is excluded from model features because it is only recorded for allergic patients
(it would leak the label).

## Disclaimer
Educational analytics project. Not a medical device and not a diagnosis.
