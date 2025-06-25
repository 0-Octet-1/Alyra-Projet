"""
🏗️ PROJET PMR - PRÉPARATION DES DONNÉES
📚 Certification Développeur IA - Alyra
🎯 Bloc 03 & 05 - Machine Learning & Deep Learning

═══════════════════════════════════════════════════════════════════════════════════

📖 OBJECTIF PÉDAGOGIQUE :
Ce script génère un dataset synthétique pour prédire l'accessibilité PMR de points d'intérêt urbains.
Chaque étape est expliquée pour comprendre les choix techniques lors de la soutenance.

🎯 CLASSES CIBLES :
- "Facilement Accessible" 
- "Modérément Accessible"
- "Difficilement Accessible"

═══════════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 DÉBUT - Préparation des Données PMR")
print("="*60)

# ═══════════════════════════════════════════════════════════════════════════════════
# 📊 ÉTAPE 1 : GÉNÉRATION DU DATASET SYNTHÉTIQUE
# ═══════════════════════════════════════════════════════════════════════════════════

print("\n📊 ÉTAPE 1 : Génération du Dataset Synthétique")
print("-" * 50)

def generer_dataset_pmr(n_samples=4000):
    """
    🎯 FONCTION PÉDAGOGIQUE : Génération du dataset synthétique PMR
    
    POURQUOI SYNTHÉTIQUE ?
    ✅ Contrôle total des variables et de leur distribution
    ✅ Pas de problèmes de confidentialité (données RGPD)
    ✅ Diversité garantie des cas d'usage
    ✅ Labels cohérents avec la logique métier PMR
    
    PARAMÈTRES :
    - n_samples : Nombre d'exemples à générer (recommandé : 3000-6000)
    """
    
    print(f"   🔄 Génération de {n_samples} exemples...")
    
    # Initialisation du générateur aléatoire pour la reproductibilité
    np.random.seed(42)
    
    # ───────────────────────────────────────────────────────────────────────────────
    # 📏 VARIABLES DESCRIPTIVES (Caractéristiques physiques)
    # ───────────────────────────────────────────────────────────────────────────────
    
    # Largeur du trottoir (cm) - Distribution normale avec queue à droite
    largeur_trottoir_cm = np.random.gamma(3, 40, n_samples)  # Moyenne ~120cm, min ~50cm
    largeur_trottoir_cm = np.clip(largeur_trottoir_cm, 50, 300)
    
    # Pente d'accès (degrés) - Majoritairement faible, quelques cas difficiles
    pente_acces_degres = np.random.exponential(2.5, n_samples)  # Moyenne ~2.5°
    pente_acces_degres = np.clip(pente_acces_degres, 0, 25)
    
    # Distance au transport (mètres) - Distribution uniforme
    distance_transport_m = np.random.randint(0, 1001, n_samples)
    
    # Nombre d'obstacles fixes - Distribution de Poisson
    nombre_obstacles_fixes = np.random.poisson(1.2, n_samples)
    nombre_obstacles_fixes = np.clip(nombre_obstacles_fixes, 0, 8)
    
    # ───────────────────────────────────────────────────────────────────────────────
    # 🏗️ VARIABLES INFRASTRUCTURELLES (Équipements et état)
    # ───────────────────────────────────────────────────────────────────────────────
    
    # Présence d'ascenseur (booléenne)
    presence_ascenseur = np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    
    # État de l'ascenseur (conditionné par sa présence)
    etat_ascenseur = []
    for has_elevator in presence_ascenseur:
        if has_elevator:
            etat_ascenseur.append(np.random.choice([
                "fonctionnel", "en_panne_occasionnelle", "hors_service"
            ], p=[0.7, 0.2, 0.1]))
        else:
            etat_ascenseur.append("non_applicable")
    
    # Type de revêtement du sol
    type_revetement_sol = np.random.choice([
        "lisse", "pavé", "gravier", "terre"
    ], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    # Âge de l'infrastructure (années)
    age_infrastructure_an = np.random.gamma(2, 15, n_samples)  # Moyenne ~30 ans
    age_infrastructure_an = np.clip(age_infrastructure_an, 0, 100).astype(int)
    
    # ───────────────────────────────────────────────────────────────────────────────
    # 🌍 VARIABLES CONTEXTUELLES (Environnement et usage)
    # ───────────────────────────────────────────────────────────────────────────────
    
    # Type de lieu
    type_lieu = np.random.choice([
        "commerce", "musée", "gare", "restaurant", 
        "hôpital", "hôtel", "mairie", "école"
    ], n_samples, p=[0.25, 0.1, 0.15, 0.2, 0.1, 0.05, 0.05, 0.1])
    
    # Type de transport en proximité
    type_transport_proximite = np.random.choice([
        "bus", "métro", "tram", "rer", "aucun"
    ], n_samples, p=[0.3, 0.2, 0.15, 0.1, 0.25])
    
    # Zone géographique
    zone_geographique = np.random.choice([
        "centre_historique", "résidentiel", "affaires", "périphérie"
    ], n_samples, p=[0.2, 0.4, 0.25, 0.15])
    
    # Affluence en heure de pointe
    affluence_heure_pointe = np.random.choice([
        "faible", "moyenne", "forte"
    ], n_samples, p=[0.3, 0.5, 0.2])
    
    # ───────────────────────────────────────────────────────────────────────────────
    # 🎯 GÉNÉRATION DES LABELS (Logique métier PMR)
    # ───────────────────────────────────────────────────────────────────────────────
    
    print("   🧠 Application de la logique métier pour les labels...")
    
    labels = []
    for i in range(n_samples):
        score_accessibilite = 0
        
        # 📏 Facteurs descriptifs (40% du score)
        if largeur_trottoir_cm[i] >= 140:
            score_accessibilite += 15
        elif largeur_trottoir_cm[i] >= 90:
            score_accessibilite += 10
        else:
            score_accessibilite += 5
            
        if pente_acces_degres[i] <= 2:
            score_accessibilite += 15
        elif pente_acces_degres[i] <= 5:
            score_accessibilite += 10
        else:
            score_accessibilite += 5
            
        if distance_transport_m[i] <= 200:
            score_accessibilite += 5
        elif distance_transport_m[i] <= 500:
            score_accessibilite += 3
            
        score_accessibilite -= min(nombre_obstacles_fixes[i] * 3, 15)
        
        # 🏗️ Facteurs infrastructurels (35% du score)
        if presence_ascenseur[i] and etat_ascenseur[i] == "fonctionnel":
            score_accessibilite += 20
        elif presence_ascenseur[i] and etat_ascenseur[i] == "en_panne_occasionnelle":
            score_accessibilite += 10
        elif presence_ascenseur[i] and etat_ascenseur[i] == "hors_service":
            score_accessibilite += 5
            
        if type_revetement_sol[i] == "lisse":
            score_accessibilite += 10
        elif type_revetement_sol[i] == "pavé":
            score_accessibilite += 7
        elif type_revetement_sol[i] == "gravier":
            score_accessibilite += 3
            
        if age_infrastructure_an[i] <= 10:
            score_accessibilite += 5
        elif age_infrastructure_an[i] <= 30:
            score_accessibilite += 3
            
        # 🌍 Facteurs contextuels (25% du score)
        if type_lieu[i] in ["hôpital", "mairie"]:
            score_accessibilite += 10  # Obligation légale renforcée
        elif type_lieu[i] in ["gare", "école"]:
            score_accessibilite += 7
        elif type_lieu[i] in ["musée", "hôtel"]:
            score_accessibilite += 5
            
        if zone_geographique[i] == "centre_historique":
            score_accessibilite -= 5  # Contraintes patrimoine
        elif zone_geographique[i] == "affaires":
            score_accessibilite += 3  # Standards modernes
            
        # 🎯 Attribution du label final
        if score_accessibilite >= 70:
            labels.append("Facilement Accessible")
        elif score_accessibilite >= 45:
            labels.append("Modérément Accessible")
        else:
            labels.append("Difficilement Accessible")
    
    # ───────────────────────────────────────────────────────────────────────────────
    # 📋 ASSEMBLAGE DU DATAFRAME FINAL
    # ───────────────────────────────────────────────────────────────────────────────
    
    dataset = pd.DataFrame({
        # Variables descriptives
        'largeur_trottoir_cm': largeur_trottoir_cm,
        'pente_acces_degres': pente_acces_degres,
        'distance_transport_m': distance_transport_m,
        'nombre_obstacles_fixes': nombre_obstacles_fixes,
        
        # Variables infrastructurelles
        'presence_ascenseur': presence_ascenseur,
        'etat_ascenseur': etat_ascenseur,
        'type_revetement_sol': type_revetement_sol,
        'age_infrastructure_an': age_infrastructure_an,
        
        # Variables contextuelles
        'type_lieu': type_lieu,
        'type_transport_proximite': type_transport_proximite,
        'zone_geographique': zone_geographique,
        'affluence_heure_pointe': affluence_heure_pointe,
        
        # Variable cible
        'difficulte_accessibilite': labels
    })
    
    print(f"   ✅ Dataset généré : {len(dataset)} exemples")
    return dataset

# Génération du dataset principal
df_pmr = generer_dataset_pmr(4000)

# ═══════════════════════════════════════════════════════════════════════════════════
# 📊 ÉTAPE 2 : ANALYSE EXPLORATOIRE RAPIDE (EDA)
# ═══════════════════════════════════════════════════════════════════════════════════

print("\n📊 ÉTAPE 2 : Analyse Exploratoire des Données (EDA)")
print("-" * 50)

# Informations générales sur le dataset
print("\n📋 INFORMATIONS GÉNÉRALES :")
print(f"   • Nombre d'exemples : {len(df_pmr)}")
print(f"   • Nombre de features : {len(df_pmr.columns)-1}")
print(f"   • Mémoire utilisée : {df_pmr.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Distribution des classes cibles
print("\n🎯 DISTRIBUTION DES CLASSES CIBLES :")
class_counts = df_pmr['difficulte_accessibilite'].value_counts()
class_props = df_pmr['difficulte_accessibilite'].value_counts(normalize=True)

for classe, count in class_counts.items():
    prop = class_props[classe]
    print(f"   • {classe:<25} : {count:>4} ({prop:.1%})")

# Vérification du déséquilibre des classes
print(f"\n⚖️  DÉSÉQUILIBRE DES CLASSES :")
ratio_min_max = class_counts.min() / class_counts.max()
print(f"   • Ratio min/max : {ratio_min_max:.2f}")
if ratio_min_max < 0.7:
    print("   • ⚠️  Déséquilibre détecté - À surveiller lors de l'évaluation")
else:
    print("   • ✅ Classes relativement équilibrées")

# Aperçu des premières lignes
print("\n👀 APERÇU DES DONNÉES :")
print(df_pmr.head(3).to_string())

# Vérification des valeurs manquantes
print("\n🔍 VÉRIFICATION DES VALEURS MANQUANTES :")
missing_counts = df_pmr.isnull().sum()
if missing_counts.sum() == 0:
    print("   • ✅ Aucune valeur manquante détectée")
else:
    print("   • ⚠️  Valeurs manquantes trouvées :")
    for col, count in missing_counts[missing_counts > 0].items():
        print(f"     - {col}: {count}")

print("\n" + "="*60)
print("✅ DATASET PMR GÉNÉRÉ AVEC SUCCÈS !")
print("📊 Prêt pour le preprocessing et la modélisation")
print("="*60)

# ═══════════════════════════════════════════════════════════════════════════════════
# 🧹 ÉTAPE 3 : PREPROCESSING DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════════

print("\n🧹 ÉTAPE 3 : Preprocessing et Préparation pour Modélisation")
print("-" * 50)

def preprocess_dataset(df, save_path=None, save_visualizations=True):
    """
    🎯 FONCTION PÉDAGOGIQUE : Preprocessing du dataset PMR
    
    ✅ Une étape critique pour garantir des modèles performants
    ✅ Traitement adapté selon type de variable
    ✅ Standardisation pour variables numériques
    ✅ Encodage one-hot pour variables catégorielles
    
    PARAMÈTRES :
    - df : DataFrame à prétraiter
    - save_path : Chemin pour sauvegarder les données (optionnel)
    """
    print("\n   📋 Séparation features / target...")
    
    # Séparation features / target
    X = df.drop('difficulte_accessibilite', axis=1)
    y = df['difficulte_accessibilite']
    
    print("   🔍 Identification des types de variables...")
    
    # Identification des types de variables
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    print(f"     • Variables catégorielles ({len(cat_cols)}) : {', '.join(cat_cols)}")
    print(f"     • Variables numériques ({len(num_cols)}) : {', '.join(num_cols)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 🛠️ CRÉATION DU PIPELINE DE PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n   🛠️ Construction du pipeline de preprocessing...")
    
    # Preprocessing pour variables numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Gestion valeurs manquantes
        ('scaler', StandardScaler())                    # Standardisation
    ])
    
    # Preprocessing pour variables catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Gestion valeurs manquantes
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # Encodage one-hot
    ])
    
    # Assemblage du preprocessor complet
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'  # Variables non spécifiées sont ignorées
    )
    
    print("   🔄 Application du preprocessing...")
    
    # Application du preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Récupération des noms des colonnes après one-hot encoding
    onehot_cols = []
    for cat_col in cat_cols:
        if cat_col == 'presence_ascenseur':
            onehot_cols.append(f"{cat_col}_True")
        else:
            encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            categories = encoder.categories_[cat_cols.index(cat_col)]
            # La première catégorie est droppée avec drop='first'
            onehot_cols.extend([f"{cat_col}_{cat}" for cat in categories[1:]])
    
    processed_cols = num_cols + onehot_cols
    
    # Création du DataFrame prétraité
    X_processed_df = pd.DataFrame(
        X_processed, 
        columns=processed_cols,
        index=X.index
    )
    
    print(f"   ✅ Preprocessing terminé : {X_processed.shape[1]} features après encodage")
    print(f"      (vs {X.shape[1]} features avant)")
    
    # Sauvegarde des données
    if save_path:
        import pickle
        import os
        
        # Création du dossier s'il n'existe pas
        os.makedirs(save_path, exist_ok=True)
        print(f"   📁 Dossier créé/vérifié : {save_path}")
        
        with open(f"{save_path}/X_train.pkl", "wb") as f:
            pickle.dump(X_processed_df, f)
        with open(f"{save_path}/y_train.pkl", "wb") as f:
            pickle.dump(y, f)
        
        print(f"   💾 Données prétraitées sauvegardées dans {save_path}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 📈 VISUALISATION DES DONNÉES PRÉTRAITÉES
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n   📊 Création de visualisations...")
    
    # Visualisation des distributions standardisées (si demandé)
    if save_visualizations:
        try:
            num_features = list(X_processed_df.columns)
            
            if len(num_features) > 0:  # S'il y a des features numériques à visualiser
                plt.figure(figsize=(15, 10))
                for i, col in enumerate(num_features[:min(len(num_features), 15)]):
                    plt.subplot(3, 5, i+1)
                    sns.histplot(X_processed_df[col], kde=True)
                    plt.title(f'{col}\nMoy: {X_processed_df[col].mean():.2f}, Std: {X_processed_df[col].std():.2f}')
                    plt.tight_layout()
                    
                if save_path:
                    plt.savefig(f"{save_path}/distributions_standardisees.png")
                plt.show()
                
                # Matrice de corrélation (si plus de 2 features)
                if len(num_features) > 2:
                    plt.figure(figsize=(12, 10))
                    corr_matrix = X_processed_df.corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                                cmap='coolwarm', square=True, linewidths=0.5)
                    plt.title('Matrice de corrélation des features prétraitées')
                    plt.tight_layout()
                    
                    if save_path:
                        plt.savefig(f"{save_path}/correlation_matrix.png")
                    plt.show()
        except Exception as e:
            print(f"⚠️ Avertissement : Erreur lors de la génération des visualisations : {e}")
            print("⚠️ Continuation du script sans les visualisations...")
    
    return X_processed_df, y, preprocessor

# Création du dossier data/preprocessed si nécessaire
import os
os.makedirs("e:/alyra/Projet/PMR/data/preprocessed", exist_ok=True)

# Application du preprocessing
X_processed, y, preprocessor = preprocess_dataset(df_pmr, save_path="e:/alyra/Projet/PMR/data/preprocessed", save_visualizations=False)

# ═══════════════════════════════════════════════════════════════════════════════════
# 🔄 ÉTAPE 4 : SPLIT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════════

print("\n🔄 ÉTAPE 4 : Split Train/Validation/Test")
print("-" * 50)

def split_dataset(X, y, val_size=0.15, test_size=0.15):
    """
    🎯 FONCTION PÉDAGOGIQUE : Split stratifié du dataset PMR
    
    ✅ Split stratifié pour conserver la distribution des classes
    ✅ Division en 3 sets : Train (70%), Validation (15%), Test (15%)
    
    PARAMÈTRES :
    - X : Features prétraitées
    - y : Target (classes d'accessibilité)
    - val_size : Taille relative du set de validation
    - test_size : Taille relative du set de test
    """
    print("\n   🔪 Split des données en Train/Validation/Test...")
    
    # Premier split pour séparer train et test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split pour séparer train et validation
    # Recalcul du ratio pour validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"   ✅ Split terminé:")
    print(f"     • Train: {X_train.shape[0]} exemples ({X_train.shape[0]/len(X):.1%})")
    print(f"     • Validation: {X_val.shape[0]} exemples ({X_val.shape[0]/len(X):.1%})")
    print(f"     • Test: {X_test.shape[0]} exemples ({X_test.shape[0]/len(X):.1%})")
    
    # Vérification de la stratification
    print("\n   🔍 Vérification de la stratification des classes:")
    for name, y_set in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
        class_props = pd.Series(y_set).value_counts(normalize=True)
        print(f"     • {name}: {', '.join([f'{c}: {p:.1%}' for c, p in class_props.items()])}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Split des données
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_processed, y)

# ═══════════════════════════════════════════════════════════════════════════════════
# 💾 ÉTAPE 5 : SAUVEGARDE DES DONNÉES PRÉTRAITÉES
# ═══════════════════════════════════════════════════════════════════════════════════

print("\n💾 ÉTAPE 5 : Sauvegarde des données prétraitées")
print("-" * 50)

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, folder="e:/alyra/Projet/PMR/data/preprocessed"):
    """
    🎯 FONCTION PÉDAGOGIQUE : Sauvegarde des données pour modélisation
    
    ✅ Format pickle pour assurer la préservation des types
    ✅ Sauvegarde du preprocessor pour application future
    
    PARAMÈTRES :
    - X_train, X_val, X_test : Features divisées
    - y_train, y_val, y_test : Target divisées
    - preprocessor : Pipeline de preprocessing
    - folder : Dossier de destination
    """
    print("\n   📁 Création du dossier de sauvegarde...")
    
    # Création du dossier s'il n'existe pas
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"     • Dossier créé : {folder}")
    
    print("   💾 Sauvegarde des données...")
    
    # Sauvegarde des datasets
    import pickle
    
    with open(f"{folder}/X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    
    with open(f"{folder}/X_val.pkl", "wb") as f:
        pickle.dump(X_val, f)
    
    with open(f"{folder}/X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
    
    with open(f"{folder}/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    
    with open(f"{folder}/y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)
    
    with open(f"{folder}/y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)
    
    with open(f"{folder}/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    
    # Sauvegarde également en CSV pour inspection visuelle
    pd.DataFrame(X_train).to_csv(f"{folder}/X_train.csv", index=False)
    pd.DataFrame(X_val).to_csv(f"{folder}/X_val.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{folder}/X_test.csv", index=False)
    pd.Series(y_train).to_csv(f"{folder}/y_train.csv", index=False)
    pd.Series(y_val).to_csv(f"{folder}/y_val.csv", index=False)
    pd.Series(y_test).to_csv(f"{folder}/y_test.csv", index=False)
    
    print(f"   ✅ Données sauvegardées avec succès dans {folder}")
    print(f"     • 13 fichiers générés (6 .pkl, 6 .csv, 1 preprocessor.pkl)")

# Sauvegarde des données pour la suite du projet
save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)

print("\n" + "="*60)
print("✅ PRÉPARATION DES DONNÉES TERMINÉE !")
print("🎯 Vous pouvez maintenant passer à l'étape de modélisation ML")
print("="*60)
