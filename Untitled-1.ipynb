from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Chargement des données
# ----------------------------

DATA_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Prédiction condition de la valve", layout="wide")
st.title("Maintenance prédictive – Condition de la valve")

st.markdown(
    """
Cette application Streamlit illustre le modèle de prédiction développé dans le notebook :

- Chargement des signaux capteurs et de `profile.txt`
- Extraction de variables explicatives (features) par cycle
- Entraînement d'une **régression logistique** (modèle retenu comme le plus performant)
- Évaluation sur l'échantillon de test final
- Interface pour obtenir une prédiction pour un cycle donné
"""
)


@st.cache_data(show_spinner=True)
def load_profile() -> pd.DataFrame:
    profile = pd.read_csv(
        DATA_DIR / "profile.txt",
        sep="\t",
        header=None,
        names=["cooler", "valve", "pump_leak", "acc_pressure", "stable_flag"],
    )
    return profile


@st.cache_data(show_spinner=True)
def summarize_signal(file_path: Path, prefix: str) -> pd.DataFrame:
    """Charge un fichier .txt (cycles x points) et calcule des features simples par cycle."""
    mat = np.loadtxt(file_path)
    mat = np.asarray(mat)

    feats = {
        f"{prefix}_mean": mat.mean(axis=1),
        f"{prefix}_std": mat.std(axis=1),
        f"{prefix}_min": mat.min(axis=1),
        f"{prefix}_max": mat.max(axis=1),
        f"{prefix}_p25": np.percentile(mat, 25, axis=1),
        f"{prefix}_p50": np.percentile(mat, 50, axis=1),
        f"{prefix}_p75": np.percentile(mat, 75, axis=1),
    }
    return pd.DataFrame(feats)


@st.cache_data(show_spinner=True)
def build_features_and_target():
    profile = load_profile()
    y = (profile["valve"] == 100).astype(int)

    sensors_100hz = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
    sensors_10hz = ["FS1", "FS2"]
    sensors_1hz = ["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]
    all_sensors = sensors_100hz + sensors_10hz + sensors_1hz

    frames = []
    for name in all_sensors:
        fpath = DATA_DIR / f"{name}.txt"
        feats = summarize_signal(fpath, name)
        frames.append(feats)

    X = pd.concat(frames, axis=1)
    return X, y, profile


@st.cache_resource(show_spinner=True)
def train_model():
    X, y, profile = build_features_and_target()

    n_train = 2000
    X_train = X.iloc[:n_train].reset_index(drop=True)
    y_train = y.iloc[:n_train].reset_index(drop=True)
    X_test = X.iloc[n_train:].reset_index(drop=True)
    y_test = y.iloc[n_train:].reset_index(drop=True)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    clf.fit(X_train, y_train)

    # Évaluation rapide sur le test
    y_pred_test = clf.predict(X_test)
    y_proba_test = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_proba_test)

    return clf, X, y, profile, (X_train, y_train, X_test, y_test, acc, auc)


with st.spinner("Chargement des données et entraînement du modèle..."):
    clf, X, y, profile, eval_data = train_model()

X_train, y_train, X_test, y_test, acc_test, auc_test = eval_data

st.subheader("Performance globale sur le jeu de test")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy (test)", f"{acc_test:.3f}")
with col2:
    st.metric("AUC ROC (test)", f"{auc_test:.3f}")

st.write(
    "Le modèle est entraîné sur les **2000 premiers cycles** et évalué sur les cycles restants,"
    " conformément aux consignes du projet."
)


# ----------------------------
# Interface de prédiction
# ----------------------------

st.subheader("Obtenir une prédiction pour un cycle")

n_cycles = len(X)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### Sélection du cycle")
    cycle_display = st.number_input(
        "Numéro de cycle (1 à N)", min_value=1, max_value=int(n_cycles), value=int(n_cycles)
    )
    idx = int(cycle_display) - 1

with col_right:
    st.markdown("### Résultat de la prédiction")
    x_sel = X.iloc[[idx]]
    proba_opt = clf.predict_proba(x_sel)[0, 1]
    pred_label = int(clf.predict(x_sel)[0])

    label_str = "Valve optimale (100 %)" if pred_label == 1 else "Valve non optimale (< 100 %)"
    st.write(f"**Prédiction du modèle :** {label_str}")
    st.write(f"**Probabilité prédite d'être optimale :** {proba_opt:.3f}")

    # Si la vérité terrain est connue (elle l'est pour les données fournies)
    true_label = int(y.iloc[idx])
    true_str = "Valve optimale (100 %)" if true_label == 1 else "Valve non optimale (< 100 %)"

    if true_label == pred_label:
        st.success(f"Vérité terrain : {true_str} (prédiction correcte)")
    else:
        st.error(f"Vérité terrain : {true_str} (prédiction incorrecte)")

    st.markdown("---")
    st.markdown("**Résumé du cycle sélectionné (profil)**")
    st.write(profile.iloc[idx : idx + 1])

st.markdown(
    """
---

**Utilisation :**

1. Installer les dépendances (dans votre environnement Python) :
   - `pip install streamlit scikit-learn pandas numpy`
2. Depuis ce dossier, lancer l'application avec :
   - `streamlit run app.py`
3. Ouvrir le lien local indiqué par Streamlit dans votre navigateur.

L'application recharge automatiquement les données, réentraîne la régression logistique comme dans le notebook,
 puis permet d'inspecter la prédiction pour n'importe quel cycle de la base.
"""
)
