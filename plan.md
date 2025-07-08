# Plan d'action accéléré pour le projet de certification PMR

## CONTEXTE PROJET
- **Certification prévue :** 23 juillet 2025
- **Temps disponible :** 10 jours, 2-3h par jour
- **Priorité :** Livrables essentiels, démonstration fonctionnelle, documentation synthétique
- **Stratégie :** Production rapide du code complet avec commentaires pédagogiques détaillés

## ÉTAT ACTUEL DU PROJET (Mise à jour 7 juillet 2025)

### TERMINÉ ET FONCTIONNEL
- **Pipeline de données complet :** 3 scripts professionnels créés et testés
  - `data_generation.py` : Génère dataset brut 4000 échantillons PMR (11 features + target)
  - `data_splitting.py` : Split stratifié 70/15/15% train/val/test
  - `data_preprocessing.py` : Imputation, standardisation, encodage
- **Outils de gestion :**
  - `clean_all_data.py` : Script de nettoyage complet pour repartir à zéro
  - `data_pipeline_dashboard.html` : Interface web interactive pour piloter le pipeline
- **Structure de données optimisée :** Sauvegarde double format (CSV + PKL), métadonnées JSON, séparation claire des dossiers
- **Workflow MLOps professionnel :** Traçabilité complète, reproductibilité, bonnes pratiques

### EN COURS / À REFAIRE
- **Modèles ML :** Scripts créés précédemment, à recréer après nettoyage (Random Forest, Logistic Regression)
- **Modèles DL :** À créer (ANN pour données tabulaires)
- **API FastAPI :** Dossier vide, à développer
- **Interface frontend :** Dossier vide, dashboard HTML créé comme alternative
- **Documentation finale :** À rédiger de manière synthétique
  - Cahier des charges (CDC-PMR.TXT)
  - README.md complet
  - Plan détaillé
  - Requirements.txt
  - Présentation pour jury

## 📋 TASK LIST - ÉTAT D'AVANCEMENT

- [x] **Pipeline de données professionnel** (data_generation.py, data_splitting.py, data_preprocessing.py)
- [x] **Outils de gestion** (clean_all_data.py, data_pipeline_dashboard.html)
- [ ] **Recréer modèles ML** (Random Forest, Logistic Regression)
- [ ] **Implémenter modèles Deep Learning** (ANN pour données tabulaires)
- [ ] **Comparer performances ML vs DL**
- [ ] **Développer API FastAPI** pour prédiction
- [ ] **Interface web de démonstration** (dashboard HTML créé, à adapter si besoin)
- [ ] **Documentation technique synthétique**
- [ ] **Support de présentation orale**

## 🗓️ PLANNING RESTANT (8-23 juillet) - 10 JOURS

| **Jour** | **Tâches Principales** | **Livrables** |
|----------|------------------------|---------------|
| **8-9 juillet** | Recréer et entraîner modèles ML | `ml_training_complete.py` |
| **10-11 juillet** | Créer et entraîner modèles DL | `dl_training.py` |
| **12-13 juillet** | Développer API FastAPI | `api/api_accessibilite.py` |
| **14-15 juillet** | Interface web finale | `frontend/` ou adaptation dashboard |
| **16-17 juillet** | Tests finaux, intégration | Notebooks de comparaison |
| **18-22 juillet** | Documentation, préparation | README, présentation |
| **23 juillet** | **🎯 CERTIFICATION** | **Soutenance finale** |

## 🎯 **OBJECTIFS DE CERTIFICATION**

### **Bloc 03 - Machine Learning**
- **Classification PMR** : 3 niveaux d'accessibilité (Facile/Modérée/Difficile)
- **Algorithmes** : Logistic Regression (baseline) + Random Forest (principal)
- **Évaluation** : Accuracy ≥ 85%, F1-score ≥ 0.80
- **Analyse** : Importance des features, matrices de confusion

### **Bloc 05 - Deep Learning**
- **Architecture ANN** : Réseau de neurones pour données tabulaires
- **Comparaison ML vs DL** : Performance, temps d'entraînement, interprétabilité
- **Déploiement** : API FastAPI avec endpoints de prédiction
- **Interface** : Démonstration web fonctionnelle

## 📁 **STRUCTURE DU PROJET**

```
PMR/
├── data/
│   ├── raw/           # Dataset brut généré
│   ├── split/         # Données train/val/test
│   └── preprocessed/  # Données finales pour ML/DL
├── models/            # Modèles entraînés (.pkl, .h5)
├── notebooks/         # Scripts Python du pipeline
├── api/              # API FastAPI (à créer)
├── frontend/         # Interface web (à créer)
└── docs/             # Documentation
```

## 🚀 **COMMANDES RAPIDES**

### **Pipeline complet :**
```bash
cd notebooks
python data_generation.py
python data_splitting.py
python data_preprocessing.py
python ml_training_complete.py
python dl_training.py
```

### **Nettoyage :**
```bash
python clean_all_data.py
```

### **Dashboard :**
Ouvrir `notebooks/data_pipeline_dashboard.html` dans le navigateur

---

**📅 Dernière mise à jour :** 7 juillet 2025 - Pipeline de données terminé ✅

---

## 🔧 **Technologies & Outils**

### **Data Science**
- **Python** : pandas, numpy, matplotlib, seaborn
- **ML** : scikit-learn (preprocessing, models, metrics)
- **DL** : TensorFlow/Keras, MobileNetV2

### **Déploiement**
- **API** : FastAPI, uvicorn
- **Frontend** : HTML5, CSS3, JavaScript (vanilla)
- **Formats** : JSON, .h5/.keras (modèles)

### **Développement**
- **IDE** : Jupyter Notebooks
- **Versioning** : Git (recommandé)
- **Documentation** : Markdown, Word/PDF

---

## ⚡ **Prochaines Actions**

### **Immédiat**
1. **Structurer le dépôt** avec dossiers appropriés
2. **Commencer `01_Data_Preparation.ipynb`** 
3. **Générer dataset synthétique PMR**

### **Cette semaine**
- Finaliser préparation des données
- Débuter modélisation ML
- Valider approche avec exemples concrets

---

## 🎨 **Interface Utilisateur (Maquette)**

### **Fonctionnalités Web**
- **Recherche lieu** : Champ texte + bouton "Rechercher sur la carte"
- **Analyse accessibilité** : Bouton "Analyser l'accessibilité"
- **Résultats** : Niveau prédit + probabilités + interprétation
- **Upload image** : Section "Analyse d'Image" (optionnel)
- **Résultat visuel** : Affichage image + prédiction CNN

---

*Ce plan sera mis à jour régulièrement selon l'avancement du projet et les retours.*
