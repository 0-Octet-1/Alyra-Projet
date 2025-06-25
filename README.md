# 🏗️ Projet PMR - Accessibilité en Milieu Urbain

## 📋 Présentation du projet

Ce projet a été développé dans le cadre de la certification **Développeur IA** (RNCP38616) - Blocs 03 & 05 - Machine Learning & Deep Learning.

L'objectif est de développer une solution d'intelligence artificielle permettant de prédire la difficulté d'accessibilité PMR (Personnes à Mobilité Réduite) de points d'intérêt urbains, classés en 3 catégories :
- **Facilement Accessible**
- **Modérément Accessible**
- **Difficilement Accessible**

## 📅 Planning du projet

- **Début**: 19 juin 2025
- **Date de certification**: 23 juillet 2025
- **Durée totale**: ~5 semaines

## 🏁 Objectifs pédagogiques

Ce projet démontre les compétences suivantes :
- Génération et prétraitement de données synthétiques
- Exploration et visualisation des données avec DuckDB et Streamlit
- Implémentation de modèles de Machine Learning (Random Forest, Logistic Regression)
- Développement de modèles de Deep Learning (ANN/CNN avec TensorFlow/Keras)
- Évaluation et comparaison des performances des modèles
- Déploiement via une API FastAPI et interface web

## 🗂️ Structure du projet

```
PMR/
├── api/                  # API FastAPI pour prédiction (endpoints /predict_tabular et /predict_image)
├── code/
│   ├── app_data_exploration.py         # Interface Streamlit pour explorer les données
│   └── data_preparation_duckdb.py      # Script de préparation des données avec DuckDB
├── data/
│   ├── processed/            # Données prétraitées (fichiers pkl)
│   └── preprocessed/         # Données prétraitées pour les modèles
├── documents/              # Documentation technique et présentation
├── frontend/              # Interface web de démonstration (en cours)
├── models/                # Modèles entraînés et sauvegardés (en cours) 
├── notebooks/
│   ├── data_preparation.py            # Script de génération et prétraitement des données
│   ├── 01_Data_Preparation.ipynb      # Notebook pédagogique de préparation des données (en cours)
│   ├── 02_ML_Model_Training_Evaluation.ipynb (en cours)
│   ├── 03_DL_Model_Training_Evaluation.ipynb (en cours)
│   └── 04_Model_Comparison_Selection.ipynb (en cours)
└── documents/            # Documentation technique et présentation
```

## 💻 Comment utiliser ce projet

### Prérequis

1. Python 3.8+ installé
2. Package manager pip
3. Environnement virtuel (recommandé)

### Installation

1. Clonez ce dépôt :
   ```
   git clone https://github.com/0-Octet-1/Alyra-Projet.git
   cd PMR
   ```

2. Créez et activez un environnement virtuel :
   ```
   python -m venv venv
   source venv/bin/activate    # Sur Windows : venv\Scripts\activate
   ```

3. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

### Exploration des données avec Streamlit

Pour lancer l'interface d'exploration des données :
```
streamlit run code/app_data_exploration.py
```

L'interface Streamlit permet d'explorer les données avec :
- Description complète de la base de données
- Statistiques descriptives 
- Matrice de corrélation interactive
- Visualisations graphiques

### Génération et prétraitement des données

Pour générer le dataset synthétique et le prétraiter :
```
python notebooks/data_preparation.py
```
Cette commande crée un dataset de 5000 exemples avec variables descriptives, infrastructurelles et contextuelles, ainsi que les labels d'accessibilité PMR.

### Préparation des données avec DuckDB

Pour préparer les données avec DuckDB :
```
python code/data_preparation_duckdb.py
```

### Modèles disponibles (en cours de développement)

1. **Machine Learning** :
   - Random Forest Classifier
   - Logistic Regression

2. **Deep Learning** :
   - ANN (réseau de neurones tabulaire)
   - CNN (MobileNetV2 pour traitement d'images si disponibles)

### Déploiement de l'API (en cours de développement)

Pour lancer l'API de prédiction :
```
uvicorn api.api_accessibilite:app --reload
```

L'API sera accessible à l'adresse http://localhost:8000

## 📊 État d'avancement du projet

### Composées développées :

- [x] Génération du dataset synthétique PMR avec données réalistes
- [x] Prétraitement des données (imputation, encodage, standardisation)
- [x] Interface d'exploration des données avec Streamlit
- [x] Intégration avec DuckDB pour la gestion des données
- [x] Analyse exploratoire des données (EDA)

### En cours de développement :

- [ ] Modèles de Machine Learning (Random Forest, Logistic Regression)
- [ ] Modèles de Deep Learning (ANN, CNN avec MobileNetV2)
- [ ] API FastAPI pour les prédictions
- [ ] Interface web de démonstration
- [ ] Documentation technique complète

## 📊 Métriques et performances attendues

Le projet vise un **F1-score macro ≥ 75%**, avec une attention particulière sur la détection correcte de la classe "Difficilement Accessible" (critique pour les utilisateurs PMR).

## 👬 Auteur

**Créé par**: [0-Octet-1](https://github.com/0-Octet-1)

_Projet développé dans le cadre de la certification Développeur IA chez Alyra, juin-juillet 2025_

## 📝 Auteur

Ce projet a été réalisé dans le cadre de la certification Développeur IA d'Alyra.

---
*Note : Ce projet est à but pédagogique dans le cadre d'une certification professionnelle.*
