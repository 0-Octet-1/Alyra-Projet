# Plan pour le projet de certification Développeur IA
## **Amélioration de l'Accessibilité PMR en Milieu Urbain**

---

## 📋 **Informations Générales**

- **Projet** : Développer une solution IA pour prédire la difficulté d'accessibilité PMR de points d'intérêt urbains
- **Certification** : RNCP38616 - Développeur IA (Blocs 03 & 05)
- **Période** : 19 juin 2025 → 23 juillet 2025 (~5 semaines)
- **Objectif** : Comparaison ML vs Deep Learning pour classification d'accessibilité
- **Langue** : Français

---

## 🎯 **Objectifs Spécifiques**

### **Bloc 03 - Machine Learning**
- ✅ **Classification ML** : Prédire 3 niveaux d'accessibilité ("Facilement", "Modérément", "Difficilement Accessible")
- ✅ **Algorithmes** : Régression Logistique (baseline) + Random Forest (modèle principal)
- ✅ **Évaluation** : Accuracy, Précision, Rappel, F1-score, Matrice de confusion
- ✅ **Facteurs clés** : Identifier les caractéristiques influençant l'accessibilité

### **Bloc 05 - Deep Learning**
- ✅ **Classification DL** : Même prédiction avec approche Deep Learning
- ✅ **Données non structurées** : Traitement d'images (entrées de lieux)
- ✅ **Architectures** : ANN (données tabulaires) + CNN (Transfer Learning avec MobileNetV2)
- ✅ **Déploiement** : API FastAPI avec endpoints ML/DL

---

## 📊 **Structure des Données**

### **Dataset Synthétique**
- **Variables** : largeur_trottoir_cm, pente_acces_degres, type_lieu, presence_rampe, etc.
- **Classes cibles** : 3 niveaux d'accessibilité
- **Images** : Photos d'entrées de lieux (optionnel pour DL)
- **Prétraitement** : StandardScaler, OneHotEncoder, division 70/15/15%

---

## 📅 **Planning Détaillé (5 semaines)**

### **Semaine 1 (19-25 juin) : Génération & Préparation des Données**
- **Livrable** : `01_Data_Preparation.ipynb`
- **Objectifs** :
  - Générer dataset synthétique PMR avec variables pertinentes
  - Analyse exploratoire des données (EDA)
  - Nettoyage, imputation des valeurs manquantes
  - Encodage variables catégorielles (OneHotEncoder)
  - Normalisation (StandardScaler)
  - Division train/validation/test (70/15/15%)
- **Technologies** : pandas, numpy, matplotlib, seaborn, sklearn

### **Semaine 2 (26 juin - 2 juillet) : Modélisation Machine Learning**
- **Livrable** : `02_ML_Model_Training_Evaluation.ipynb`
- **Objectifs** :
  - Implémentation Régression Logistique (baseline)
  - Développement Random Forest (modèle principal)
  - Optimisation hyperparamètres
  - Évaluation complète (métriques, matrice confusion)
  - Analyse importance des features
- **Technologies** : sklearn (LogisticRegression, RandomForestClassifier)

### **Semaine 3 (3-9 juillet) : Deep Learning & Comparaison**
- **Livrable** : `03_DL_Model_Training_Evaluation.ipynb`
- **Objectifs** :
  - Architecture ANN pour données tabulaires
  - CNN avec Transfer Learning (MobileNetV2) pour images
  - Callbacks (EarlyStopping, ReduceLROnPlateau)
  - Évaluation et visualisations (courbes d'apprentissage)
  - Comparaison ML vs DL
- **Technologies** : TensorFlow/Keras, tf.keras.applications.MobileNetV2

### **Semaine 4 (10-16 juillet) : Déploiement & Interface**
- **Livrables** : `04_Model_Comparison_Selection.ipynb`, `api_accessibilite.py`, `frontend_demo/`
- **Objectifs** :
  - Sélection modèle final (ML vs DL)
  - API FastAPI avec endpoints :
    - `/predict_tabular` (données structurées)
    - `/predict_image` (analyse d'images)
  - Interface web de démonstration
  - Documentation technique
- **Technologies** : FastAPI, HTML/CSS/JavaScript

### **Semaine 5 (17-23 juillet) : Tests & Soutenance**
- **Livrables** : Dossier Technique, Présentation
- **Objectifs** :
  - Tests finaux et optimisations
  - Rédaction dossier technique complet
  - Préparation présentation orale
  - **Certification : 23 juillet 2025**

---

## 🏗️ **Architecture Technique**

### **Modèles ML**
```python
# Baseline
LogisticRegression(C=1.0, random_state=42)

# Modèle principal
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
```

### **Modèles DL**
```python
# ANN pour données tabulaires
model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 classes
])

# CNN avec Transfer Learning
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False
```

### **API FastAPI**
```python
@app.post("/predict_tabular")
async def predict_tabular(data: dict):
    # Prétraitement + prédiction ML/DL
    return {"niveau_difficulte": "...", "probabilites": {...}}

@app.post("/predict_image")
async def predict_image(file: UploadFile):
    # Traitement image + prédiction CNN
    return {"resultat_visuel": "..."}
```

---

## 📋 **Livrables Finaux**

### **Code & Notebooks**
1. **`01_Data_Preparation.ipynb`** - Génération dataset + prétraitement
2. **`02_ML_Model_Training_Evaluation.ipynb`** - Random Forest + évaluation
3. **`03_DL_Model_Training_Evaluation.ipynb`** - ANN/CNN + évaluation  
4. **`04_Model_Comparison_Selection.ipynb`** - Comparaison finale
5. **`api_accessibilite.py`** - API FastAPI complète
6. **`frontend_demo/`** - Interface web (HTML/CSS/JS)

### **Documentation**
7. **Dossier Technique** - Document structuré (Word/PDF)
8. **Présentation** - Support visuel pour soutenance orale

---

## 🎯 **Critères de Réussite**

### **Performance Technique**
- **F1-score macro ≥ 75%** sur ensemble de test
- **Focus classe "Difficilement Accessible"** (la plus critique)
- **API fonctionnelle** avec démonstration fluide
- **Interface web interactive**

### **Compétences Démontrées**
- ✅ Justification de chaque choix technique
- ✅ Maîtrise prétraitement données structurées/non structurées
- ✅ Comparaison rigoureuse ML vs DL
- ✅ Déploiement MLOps (API + interface)
- ✅ Considérations éthiques (biais, confidentialité)

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
