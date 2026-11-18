"""Allergy Risk Insights - Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import ml_pipeline as ml
from generate_dataset import generate

BASE = Path(__file__).parent
DATA_FILE = BASE / "allergy_data.csv"
USER_FILE = BASE / "users.json"

st.set_page_config(page_title="Allergy Risk Insights", page_icon="🌿", layout="wide")

# ------------------------------------------------------------------ auth ---

def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex() + "$" + digest.hex()


def _verify(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$")
    candidate = _hash(password, bytes.fromhex(salt_hex)).split("$")[1]
    return hmac.compare_digest(candidate, digest_hex)


def _load_users() -> dict:
    try:
        return json.loads(USER_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def login_screen() -> None:
    st.title("🌿 Allergy Risk Insights")
    st.caption("ML-powered allergy analytics · please sign in")
    tab_in, tab_up = st.tabs(["Sign in", "Create account"])
    with tab_in:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                users = _load_users()
                if u in users and _verify(p, users[u]):
                    st.session_state.user = u
                    st.rerun()
                st.error("Invalid username or password.")
    with tab_up:
        with st.form("register"):
            u = st.text_input("Choose a username")
            p = st.text_input("Choose a password (min 8 chars)", type="password")
            if st.form_submit_button("Register"):
                users = _load_users()
                if not u or len(p) < 8:
                    st.error("Enter a username and a password of at least 8 characters.")
                elif u in users:
                    st.error("That username already exists.")
                else:
                    users[u] = _hash(p)
                    USER_FILE.write_text(json.dumps(users))
                    st.success("Account created. You can sign in now.")


# ------------------------------------------------------------------ data ---

@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    if not DATA_FILE.exists():
        generate().to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE)


@st.cache_resource(show_spinner="Training models…")
def train(df_json: str):
    return ml.train_and_evaluate(pd.read_json(io.StringIO(df_json)))


def pdf_report(df: pd.DataFrame, res: ml.TrainResult) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Allergy Risk Insights Report")
    s = getSampleStyleSheet()
    rate = (df[ml.TARGET] == ml.POSITIVE).mean()

    def table(rows, widths=None):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f7d5b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f7f4")]),
        ]))
        return t

    story = [
        Paragraph("Allergy Risk Insights – Analysis Report", s["Title"]),
        Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M}", s["Normal"]),
        Spacer(1, 12),
        Paragraph("Dataset summary", s["Heading2"]),
        Paragraph(
            f"{len(df):,} patient records · {rate:.1%} allergic · "
            f"mean age {df['Age'].mean():.1f} · mean symptom count "
            f"{df[ml.NUMERIC[-5:]].sum(axis=1).mean():.2f}", s["Normal"]),
        Spacer(1, 10),
        Paragraph("Model comparison (hold-out test set)", s["Heading2"]),
    ]
    m = res.results.round(3)
    story.append(table([list(m.columns)] + m.astype(str).values.tolist()))
    story += [Spacer(1, 10), Paragraph(f"Selected model: <b>{res.best_name}</b>", s["Normal"]),
              Spacer(1, 10), Paragraph("Top risk drivers (permutation importance)", s["Heading2"])]
    imp = res.importance.head(10).round(4)
    story.append(table([["Feature", "Importance", "Std"]] + imp.astype(str).values.tolist()))
    tn, fp, fn, tp = res.confusion.ravel()
    story += [Spacer(1, 10), Paragraph("Confusion matrix", s["Heading2"]),
              Paragraph(f"TN {tn} · FP {fp} · FN {fn} · TP {tp}", s["Normal"]), Spacer(1, 14),
              Paragraph("<i>Educational analytics tool - not a medical diagnosis. "
                        "Consult a qualified clinician for any health decisions.</i>", s["Normal"])]
    doc.build(story)
    return buf.getvalue()


# ----------------------------------------------------------------- pages ---

GREEN, RED, GREY = "#2f7d5b", "#d1495b", "#8d99ae"
COLORS = {"Allergic": RED, "Not Allergic": GREEN}


def page_overview(df: pd.DataFrame, res: ml.TrainResult) -> None:
    st.header("Overview")
    allergic = df[ml.TARGET] == ml.POSITIVE
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(df):,}")
    c2.metric("Allergic", f"{allergic.mean():.1%}")
    c3.metric("Best model", res.best_name)
    c4.metric("ROC-AUC", f"{res.results.loc[0, 'ROC-AUC']:.3f}")

    a, b = st.columns(2)
    with a:
        fig = px.pie(df, names=ml.TARGET, hole=0.55, color=ml.TARGET, color_discrete_map=COLORS,
                     title="Allergy status split")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        tc = ml.trigger_counts(df)
        fig = px.bar(tc, x="Cases", y="Trigger", orientation="h", title="Primary triggers",
                     color_discrete_sequence=[GREEN])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    seasonal = df.groupby("Season")[ml.TARGET].apply(lambda s: (s == ml.POSITIVE).mean()).reset_index(name="Rate")
    fig = px.bar(seasonal, x="Season", y="Rate", title="Allergy rate by season",
                 color_discrete_sequence=[GREEN])
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def page_explorer(df: pd.DataFrame) -> None:
    st.header("Data Explorer")
    with st.sidebar.expander("Filters", expanded=False):
        seasons = st.multiselect("Season", sorted(df["Season"].unique()), default=sorted(df["Season"].unique()))
        cities = st.multiselect("City type", sorted(df["City_Type"].unique()), default=sorted(df["City_Type"].unique()))
        lo, hi = int(df["Age"].min()), int(df["Age"].max())
        ages = st.slider("Age range", lo, hi, (lo, hi))
    f = df[df["Season"].isin(seasons) & df["City_Type"].isin(cities) & df["Age"].between(*ages)]
    st.caption(f"{len(f):,} of {len(df):,} records match the filters")

    t1, t2, t3, t4 = st.tabs(["Table", "Distributions", "Environment", "Correlation"])
    with t1:
        st.dataframe(f, use_container_width=True, height=400)
        st.download_button("Download filtered CSV", f.to_csv(index=False), "filtered_allergy_data.csv", "text/csv")
        st.dataframe(f.describe().T.round(2), use_container_width=True)
    with t2:
        col = st.selectbox("Feature", ["Age", "Pollen_Level", "Dust_Level", "AQI", "Humidity_Pct", "Temperature_C"])
        fig = px.histogram(f, x=col, color=ml.TARGET, barmode="overlay", opacity=0.65,
                           color_discrete_map=COLORS, marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        x = st.selectbox("X axis", ["Pollen_Level", "Dust_Level", "AQI", "Humidity_Pct"], index=0)
        y = st.selectbox("Y axis", ["Dust_Level", "Pollen_Level", "AQI", "Humidity_Pct"], index=0)
        fig = px.scatter(f, x=x, y=y, color=ml.TARGET, color_discrete_map=COLORS, opacity=0.6,
                         hover_data=["Age", "Season", "City_Type"])
        st.plotly_chart(fig, use_container_width=True)
    with t4:
        num = f[ml.NUMERIC].copy()
        num["Allergic"] = (f[ml.TARGET] == ml.POSITIVE).astype(int)
        fig = px.imshow(num.corr().round(2), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, height=650)
        st.plotly_chart(fig, use_container_width=True)


def page_model(res: ml.TrainResult) -> None:
    st.header("Model Performance")
    st.caption("Models are trained on 80% of the data and evaluated on a stratified 20% hold-out set; "
               "CV ROC-AUC is 5-fold on the training portion.")
    st.dataframe(
        res.results.style.format({c: "{:.3f}" for c in res.results.columns if c != "Model"})
        .highlight_max(subset=[c for c in res.results.columns if c != "Model"], color="#d8efe3"),
        use_container_width=True, hide_index=True,
    )
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for name, (fpr, tpr) in res.roc.items():
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=name)
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="Chance")
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.imshow(res.confusion, text_auto=True, x=["Not Allergic", "Allergic"], y=["Not Allergic", "Allergic"],
                        labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Greens",
                        title=f"Confusion matrix – {res.best_name}")
        st.plotly_chart(fig, use_container_width=True)

    imp = res.importance.head(12)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", error_x="Std",
                 title="What drives the prediction (permutation importance, ROC-AUC drop)",
                 color_discrete_sequence=[GREEN])
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)


def page_predict(df: pd.DataFrame, res: ml.TrainResult) -> None:
    st.header("Predict Allergy Risk")
    st.caption("Enter a patient profile. The result is a statistical estimate, not a diagnosis.")
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Profile")
            age = st.number_input("Age", 1, 100, 28)
            gender = st.selectbox("Gender", ["Female", "Male"])
            city = st.selectbox("City type", ["Urban", "Semi-Urban", "Rural"])
            season = st.selectbox("Season", ["Spring", "Summer", "Monsoon", "Autumn", "Winter"])
            fam = st.checkbox("Family history of allergy")
            asthma = st.checkbox("Asthma")
            eczema = st.checkbox("Eczema")
            smoker = st.checkbox("Smoker")
            pet = st.checkbox("Pet at home")
        with c2:
            st.subheader("Environment")
            pollen = st.slider("Pollen level", 0.0, 10.0, 4.0, 0.1)
            dust = st.slider("Dust level", 0.0, 10.0, 4.0, 0.1)
            temp = st.slider("Temperature (°C)", 0.0, 50.0, 28.0, 0.5)
            hum = st.slider("Humidity (%)", 10, 100, 50)
            aqi = st.slider("Air quality index", 20, 450, 120)
        with c3:
            st.subheader("Symptoms")
            sneeze = st.checkbox("Sneezing")
            eyes = st.checkbox("Itchy eyes")
            nasal = st.checkbox("Nasal congestion")
            rash = st.checkbox("Skin rash")
            wheeze = st.checkbox("Wheezing")
        go_btn = st.form_submit_button("Predict", type="primary")

    if go_btn:
        rec = dict(Gender=gender, City_Type=city, Season=season, Age=age, Family_History=int(fam),
                   Asthma=int(asthma), Eczema=int(eczema), Smoker=int(smoker), Pet_Exposure=int(pet),
                   Pollen_Level=pollen, Dust_Level=dust, Temperature_C=temp, Humidity_Pct=hum, AQI=aqi,
                   Sneezing=int(sneeze), Itchy_Eyes=int(eyes), Nasal_Congestion=int(nasal),
                   Skin_Rash=int(rash), Wheezing=int(wheeze))
        p, band = ml.predict_patient(res.best, rec)
        g1, g2 = st.columns([1, 1])
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100, number={"suffix": "%"},
                title={"text": f"Allergy probability – {band} risk"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": {"Low": GREEN, "Moderate": "#e9a23b", "High": RED}[band]},
                       "steps": [{"range": [0, 33], "color": "#e3f1ea"}, {"range": [33, 66], "color": "#fbefd9"},
                                 {"range": [66, 100], "color": "#f7dde1"}]}))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.subheader("Context")
            base = (df[ml.TARGET] == ml.POSITIVE).mean()
            st.write(f"Dataset base rate: **{base:.1%}**. This profile: **{p:.1%}**.")
            top = res.importance.head(3)["Feature"].tolist()
            st.write("The model relies most on: " + ", ".join(f"`{t}`" for t in top) + ".")
            st.info("Not a medical diagnosis. Persistent symptoms should be assessed by a clinician.")


def page_reports(df: pd.DataFrame, res: ml.TrainResult) -> None:
    st.header("Reports & Export")
    st.write("Download a PDF summary of the dataset, model comparison and top risk drivers.")
    st.download_button("📄 Download PDF report", pdf_report(df, res), "Allergy_Risk_Report.pdf", "application/pdf", type="primary")
    st.download_button("⬇️ Download model metrics (CSV)", res.results.to_csv(index=False), "model_metrics.csv", "text/csv")
    st.download_button("⬇️ Download feature importance (CSV)", res.importance.to_csv(index=False), "feature_importance.csv", "text/csv")


# ------------------------------------------------------------------ main ---

def main() -> None:
    if "user" not in st.session_state:
        login_screen()
        return

    st.sidebar.title("🌿 Allergy Risk Insights")
    st.sidebar.caption(f"Signed in as **{st.session_state.user}**")
    if st.sidebar.button("Sign out"):
        del st.session_state["user"]
        st.rerun()

    up = st.sidebar.file_uploader("Use your own dataset (CSV)", type="csv")
    df = load_default()
    if up is not None:
        custom = pd.read_csv(up)
        problems = ml.validate(custom)
        if problems:
            st.sidebar.error("\n".join(problems))
        else:
            df = custom
            st.sidebar.success(f"Loaded {len(df):,} rows")

    res = train(df.to_json())
    page = st.sidebar.radio("Navigate", ["Overview", "Data Explorer", "Model Performance", "Predict Risk", "Reports"])

    if page == "Overview":
        page_overview(df, res)
    elif page == "Data Explorer":
        page_explorer(df)
    elif page == "Model Performance":
        page_model(res)
    elif page == "Predict Risk":
        page_predict(df, res)
    else:
        page_reports(df, res)


main()
