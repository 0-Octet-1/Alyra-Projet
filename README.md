# 🏗️ PMR-AI : Prédiction d'Accessibilité pour Personnes à Mobilité Réduite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![F1-Score](https://img.shields.io/badge/F1--Score-94.4%25-brightgreen)](https://github.com/0-Octet-1/Alyra-Projet)

## 🌟 Aperçu du Projet

**PMR-AI** est une solution d'IA qui prédit l'accessibilité des lieux publics pour les personnes à mobilité réduite (PMR). Ce projet a été développé dans le cadre de la certification **Développeur IA** (RNCP38616) - Blocs 03 & 05.

### 🎯 Objectifs
- Prédire le niveau d'accessibilité des lieux publics (3 classes)
- Fournir une API REST pour les prédictions en temps réel
- Offrir une interface utilisateur intuitive
- Atteindre une précision optimale (F1-score > 75%)

### 🏆 Résultats
- **F1-Score** : 94.4% (Objectif : 75%)
- **Accuracy** : 97.5%
- **Temps de réponse** : < 1 seconde

## 🚀 Fonctionnalités

### 🧠 Modèles Implémentés
- **Machine Learning**
  - Random Forest (79.1% F1-score)
  - Régression Logistique (74.4% F1-score)
- **Deep Learning**
  - MLP Simple (92.6% F1-score)
  - MLP Profond (94.4% F1-score) 🏆

### 🌐 Déploiement
- **API REST** avec FastAPI
- Interface Web interactive
- Prédictions par lot (batch) et en temps réel
- Documentation Swagger intégrée

### 📊 Données
- 4000 points d'intérêt urbains
- 11 caractéristiques techniques
- 3 classes d'accessibilité
- Données synthétiques réalistes

## 🛠 Installation

### Prérequis
- Python 3.8+
- pip
- Git

### Étapes d'Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/0-Octet-1/Alyra-Projet.git
   cd PMR
   ```

2. **Créer un environnement virtuel**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Utilisation

### 1. Lancer l'API
```bash
# Depuis le dossier racine
cd api
uvicorn main:app --reload
```
L'API sera disponible à l'adresse : http://localhost:8000

### 2. Accéder à l'interface web
```bash
# Depuis le dossier racine
cd frontend
python serve.py
```
Ouvrez votre navigateur à l'adresse : http://localhost:3000

### 3. Utilisation de l'API

**Prédiction unique**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict?model_type=dl' \
  -H 'Content-Type: application/json' \
  -d '{
    "largeur_trottoir": 2.5,
    "hauteur_bordure": 0.15,
    "pente_acces": 5.0,
    "distance_transport": 150.0,
    "eclairage_qualite": 4,
    "surface_qualite": 4,
    "signalisation_presence": 1,
    "obstacles_nombre": 2,
    "rampe_presence": 1,
    "places_pmr_nombre": 3,
    "type_poi": "restaurant"
  }'
```

**Réponse attendue**
```json
{
  "prediction": 2,
  "prediction_label": "Accessible",
  "confidence": 0.94,
  "probabilities": {
    "Non accessible": 0.02,
    "Partiellement accessible": 0.04,
    "Accessible": 0.94
  },
  "model_used": "MLP Profond (Deep Learning)",
  "timestamp": "2025-07-08T20:00:00"
}
```

## 📁 Structure du Projet

```
PMR/
├── api/                    # API FastAPI
│   ├── main.py            # Points d'entrée de l'API
│   ├── models.py          # Chargement des modèles
│   ├── schemas.py         # Modèles Pydantic
│   └── test_api.py        # Tests automatisés
├── data/                  # Données
│   ├── raw/               # Données brutes
│   ├── split/             # Données divisées
│   └── preprocessed/      # Données prétraitées
├── frontend/              # Interface utilisateur
│   ├── index.html         # Application web
│   └── serve.py           # Serveur local
├── models/                # Modèles sauvegardés
│   ├── mlp_*             # Modèles MLP
│   └── rf_*              # Modèles Random Forest
├── notebooks/             # Notebooks et scripts
│   ├── data_generation.py
│   ├── data_splitting.py
│   ├── data_preprocessing.py
│   ├── ml_training_complete.py
│   └── dl_training_simple.py
└── docs/                  # Documentation
    ├── documentation_technique.md
    └── presentation_certification.md
```

## 📊 Résultats Détail

### Performances des Modèles
| Modèle | F1-Score | Accuracy | Temps d'inférence |
|--------|----------|----------|-------------------|
| MLP Profond | 94.4% | 97.5% | < 1s |
| MLP Simple | 92.6% | 96.5% | < 1s |
| Random Forest | 79.1% | 92.5% | < 0.1s |
| Régression Logistique | 74.4% | 92.5% | < 0.1s |

### Matrice de Confusion (MLP Profond)
```
                Prédit
Réel      0    1    2
    0    26    2    0   (93% précision)
    1     3  489    5   (98% précision)  
    2     0    5   70   (93% précision)
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## 📧 Contact

0-Octet-1 - [@votretwitter](https://twitter.com/votretwitter)

Lien du projet : [https://github.com/0-Octet-1/Alyra-Projet](https://github.com/0-Octet-1/Alyra-Projet)

## 🙏 Remerciements

- Alyra pour la formation
- Les formateurs pour leur accompagnement
- La communauté open source

---
*Projet développé dans le cadre de la certification Développeur IA - Juillet 2025*
