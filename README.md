# Projet de maintenance prédictive – Condition de la valve

Ce projet met en place un modèle de Machine Learning pour prédire, pour chaque cycle d’un banc d’essai hydraulique, si la **condition de la valve** est **optimale (100 %)** ou **non optimale**.

Les données sont fournies sous forme de fichiers texte (`.txt`) où **chaque ligne correspond à un cycle** et **chaque colonne à un échantillon temporel** d’un capteur (pressions, débits, températures, etc.).

## 1. Contenu du dépôt

- `Untitled-1.ipynb` : notebook Jupyter principal contenant tout le pipeline :
  - chargement de `profile.txt` et des fichiers capteurs (PS1–PS6, FS1–FS2, TS1–TS4, VS1, CE, CP, SE, EPS1),
  - extraction de variables explicatives (moyenne, écart-type, min, max, percentiles 25/50/75) par cycle,
  - séparation apprentissage / test (2000 premiers cycles pour l’apprentissage, reste pour le test),
  - entraînement d’une **régression logistique** et comparaison avec Random Forest et Gradient Boosting,
  - évaluation sur l’échantillon de test (matrice de confusion, accuracy, précision/rappel/F1, AUC ROC, courbes ROC),
  - analyse de l’importance des variables (coefficients de la régression logistique),
  - optimisation d’une Forêt Aléatoire par `RandomizedSearchCV`.

- `app.py` : application web **Streamlit** qui :
  - reconstruit les mêmes features que dans le notebook,
  - entraîne une **régression logistique** sur les 2000 premiers cycles,
  - affiche les performances globales (accuracy, AUC ROC) sur le jeu de test,
  - permet de sélectionner un numéro de cycle et de voir la prédiction, la probabilité et la vérité terrain.

- Fichiers de données (`*.txt`) :
  - `profile.txt` (cible `valve` et autres états : `cooler`, `pump_leak`, `acc_pressure`, `stable_flag`),
  - fichiers capteurs (pressions PS1–PS6, débit FS1–FS2, températures TS1–TS4, vibrations VS1, état CE/CP/SE, etc.).

## 2. Installation (localement)

1. Cloner le dépôt ou copier ce dossier.
2. (Optionnel) Créer un environnement virtuel Python.
3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## 3. Lancer le notebook

Ouvrir `Untitled-1.ipynb` dans Jupyter / VS Code et exécuter les cellules dans l’ordre. Le notebook contient :

- l’exploration de la cible (condition de la valve),
- la construction de la matrice de caractéristiques `X`,
- l’entraînement et l’évaluation des modèles,
- des commentaires détaillés pour chaque partie et une synthèse finale.

## 4. Lancer l’application Streamlit

### 4.1 Application en ligne (Streamlit Cloud)

L’application est déployée ici :

- https://alaeddinemezghit4-beep-maintenance-predictive-valve-app-7b9zbg.streamlit.app/

### 4.2 Exécution locale (optionnelle)

Depuis le dossier du projet :

```bash
streamlit run app.py
```

Puis ouvrir le lien indiqué par Streamlit (par défaut http://localhost:8501) dans un navigateur.

## 5. Modèle final retenu

Après comparaison de plusieurs algorithmes et optimisation d’une Forêt Aléatoire, la **régression logistique avec standardisation des features** est retenue comme **modèle final** car elle offre le meilleur compromis :

- bonnes performances sur le jeu de test (accuracy et AUC ROC élevées),
- simplicité de mise en œuvre,
- forte interprétabilité via l’analyse des coefficients.

L’application Streamlit met en œuvre ce modèle et permet de tester facilement la prédiction de la condition de la valve pour n’importe quel cycle.
