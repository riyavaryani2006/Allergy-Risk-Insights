"""Machine-learning core for Allergy Risk Insights (no UI code here)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Allergy_Status"
POSITIVE = "Allergic"
ID_COL = "Patient_ID"
# Primary_Trigger is only recorded for allergic patients -> would leak the label.
LEAKY = ["Primary_Trigger"]

CATEGORICAL = ["Gender", "City_Type", "Season"]
NUMERIC = [
    "Age", "Family_History", "Asthma", "Eczema", "Smoker", "Pet_Exposure",
    "Pollen_Level", "Dust_Level", "Temperature_C", "Humidity_Pct", "AQI",
    "Sneezing", "Itchy_Eyes", "Nasal_Congestion", "Skin_Rash", "Wheezing",
]
FEATURES = CATEGORICAL + NUMERIC

REQUIRED_COLUMNS = FEATURES + [TARGET]


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with an uploaded dataframe (empty = OK)."""
    problems = [f"Missing column: {c}" for c in REQUIRED_COLUMNS if c not in df.columns]
    if not problems and set(df[TARGET].unique()) - {"Allergic", "Not Allergic"}:
        problems.append("Allergy_Status must contain only 'Allergic' / 'Not Allergic'.")
    return problems


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def model_zoo() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


@dataclass
class TrainResult:
    results: pd.DataFrame                     # one row per model
    pipelines: dict = field(default_factory=dict)
    best_name: str = ""
    X_test: pd.DataFrame | None = None
    y_test: np.ndarray | None = None
    proba: dict = field(default_factory=dict)
    confusion: np.ndarray | None = None
    roc: dict = field(default_factory=dict)
    importance: pd.DataFrame | None = None

    @property
    def best(self) -> Pipeline:
        return self.pipelines[self.best_name]


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> TrainResult:
    X = df[FEATURES]
    y = (df[TARGET] == POSITIVE).astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rows, pipelines, proba, roc = [], {}, {}, {}
    for name, clf in model_zoo().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, p),
            "CV ROC-AUC (5-fold)": cv_auc,
        })
        pipelines[name], proba[name] = pipe, p
        fpr, tpr, _ = roc_curve(y_test, p)
        roc[name] = (fpr, tpr)

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best = pipelines[best_name]
    best_pred = (proba[best_name] >= 0.5).astype(int)

    imp = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=8, random_state=seed, n_jobs=-1
    )
    importance = (
        pd.DataFrame({"Feature": FEATURES, "Importance": imp.importances_mean, "Std": imp.importances_std})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return TrainResult(
        results=results, pipelines=pipelines, best_name=best_name, X_test=X_test,
        y_test=y_test, proba=proba, confusion=confusion_matrix(y_test, best_pred),
        roc=roc, importance=importance,
    )


def risk_band(p: float) -> str:
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def predict_patient(pipe: Pipeline, record: dict) -> tuple[float, str]:
    p = float(pipe.predict_proba(pd.DataFrame([record])[FEATURES])[0, 1])
    return p, risk_band(p)


def trigger_counts(df: pd.DataFrame) -> pd.DataFrame:
    s = df.loc[df["Primary_Trigger"] != "None", "Primary_Trigger"].value_counts()
    return s.rename_axis("Trigger").reset_index(name="Cases")
