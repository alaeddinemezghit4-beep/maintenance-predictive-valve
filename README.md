# Maintenance prédictive – Condition de la valve

Ce projet met en place un modèle de Machine Learning pour prédire, pour chaque cycle de production d’un banc d’essai hydraulique, si la **condition de la valve** est **optimale (100 %)** ou **non optimale**.

## Contenu du projet

- `projet_maintenance_valve.ipynb` : notebook Jupyter principal contenant tout le pipeline :
  - chargement des données (`profile.txt` et fichiers capteurs PS*, FS*, TS*, VS1, CE, CP, SE),
  - extraction de features par cycle (moyenne, écart-type, min, max, percentiles),
  - séparation apprentissage / test (2000 premiers cycles pour l’apprentissage),
  - entraînement d’une **régression logistique** et de modèles alternatifs (Random Forest, Gradient Boosting),
  - évaluation sur l’échantillon de test final (matrice de confusion, accuracy, précision/rappel/F1, AUC ROC),
  - interprétation des résultats et importance des variables,
  - optimisation d’une Forêt Aléatoire par `RandomizedSearchCV`.
- `app.py` : application web **Streamlit** permettant :
  - de recharger les données et réentraîner la régression logistique,
  - d’afficher les performances globales (accuracy, AUC ROC) sur le jeu de test,
  - de sélectionner un numéro de cycle et d’obtenir la prédiction (valve optimale / non optimale) et la vérité terrain.
- Fichiers de données (`*.txt`) :
  - `profile.txt` (cible `valve` et autres états : `cooler`, `pump_leak`, `acc_pressure`, `stable_flag`),
  - fichiers capteurs (pressions PS1–PS6, débit FS1–FS2, températures TS1–TS4, vibrations VS1, etc.).

## Installation

1. Cloner le dépôt ou copier ce dossier.
2. Créer (optionnel) un environnement virtuel.
3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Lancer le notebook

Ouvrir `projet_maintenance_valve.ipynb` dans Jupyter / VS Code et exécuter les cellules dans l’ordre.

## Lancer l’application web Streamlit

Depuis le dossier du projet :

```bash
streamlit run app.py
```

Puis ouvrir le lien local (par défaut http://localhost:8501) dans un navigateur.

## Modèle final retenu

Après comparaison et optimisation, la **régression logistique avec standardisation des features** est retenue comme **modèle final** car elle offre le meilleur compromis **performance / simplicité / interprétabilité** sur le jeu de test.
