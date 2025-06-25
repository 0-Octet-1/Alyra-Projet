# 🏗️ Projet PMR - Accessibilité en Milieu Urbain

## 📋 Présentation du projet

Ce projet a été développé dans le cadre de la certification **Développeur IA** (RNCP38616) - Blocs 03 & 05 - Machine Learning & Deep Learning.

L'objectif est de développer une solution d'intelligence artificielle permettant de prédire la difficulté d'accessibilité PMR (Personnes à Mobilité Réduite) de points d'intérêt urbains, classés en 3 catégories :
- **Facilement Accessible**
- **Modérément Accessible**
- **Difficilement Accessible**

## 🎯 Objectifs pédagogiques

Ce projet démontre les compétences suivantes :
- Génération et prétraitement de données synthétiques
- Implémentation de modèles de Machine Learning (Random Forest, Logistic Regression)
- Développement de modèles de Deep Learning (ANN/CNN avec TensorFlow/Keras)
- Évaluation et comparaison des performances des modèles
- Déploiement via une API FastAPI et interface web

## 🗂️ Structure du projet

```
PMR/
├── api/                  # API FastAPI pour prédiction (endpoints /predict_tabular et /predict_image)
├── data/
│   └── preprocessed/     # Données prétraitées (train, validation, test)
├── fronted/              # Interface web de démonstration
├── models/               # Modèles entraînés et sauvegardés
├── notebooks/
│   ├── data_preparation.py            # Script de génération et prétraitement des données
│   ├── 01_Data_Preparation.ipynb      # Notebook pédagogique de préparation des données
│   ├── 02_ML_Model_Training_Evaluation.ipynb
│   ├── 03_DL_Model_Training_Evaluation.ipynb
│   └── 04_Model_Comparison_Selection.ipynb
└── documents/            # Documentation technique et présentation
```

## 💻 Comment utiliser ce projet

### Prérequis

1. Python 3.8+ installé
2. Package manager pip

### Installation

1. Clonez ce dépôt :
   ```
   git clone <lien-du-repo>
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

### Génération et prétraitement des données

Pour générer le dataset synthétique et le prétraiter :
```
python notebooks/data_preparation.py
```
Cette commande crée un dataset de 4000 exemples avec variables descriptives, infrastructurelles et contextuelles, ainsi que les labels d'accessibilité PMR.

### Exécution des notebooks

Pour utiliser les notebooks pédagogiques (nécessite Jupyter) :
```
jupyter notebook notebooks/
```

### Modèles disponibles

1. **Machine Learning** :
   - Random Forest Classifier
   - Logistic Regression

2. **Deep Learning** :
   - ANN (réseau de neurones tabulaire)
   - CNN (MobileNetV2 pour traitement d'images si disponibles)

### Déploiement de l'API

Pour lancer l'API de prédiction :
```
uvicorn api.api_accessibilite:app --reload
```

L'API sera accessible à l'adresse http://localhost:8000

## 📊 Métriques et performances

L'objectif principal est d'obtenir un **F1-score macro ≥ 75%** avec une attention particulière sur la performance de la classe "Difficilement Accessible".

## 📝 Auteur

Ce projet a été réalisé dans le cadre de la certification Développeur IA d'Alyra.

---
*Note : Ce projet est à but pédagogique dans le cadre d'une certification professionnelle.*
