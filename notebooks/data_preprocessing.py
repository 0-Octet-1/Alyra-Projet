# ===================================================================
# 🔧 PREPROCESSING DES DONNÉES PMR
# ===================================================================

"""
🎯 ÉTAPE 3 : PREPROCESSING DES DONNÉES SÉPARÉES

📋 OBJECTIF :
- Charger les datasets train/val/test séparés
- Traitement des valeurs manquantes
- Standardisation des features numériques
- Encodage des variables catégorielles
- Sauvegarde des données prêtes pour ML
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_split_datasets():
    """
    📂 CHARGEMENT DES DATASETS SÉPARÉS
    """
    
    split_dir = Path("../data/split")
    
    if not split_dir.exists():
        print("❌ Dossier split non trouvé !")
        print("🔧 Exécutez d'abord : python data_splitting.py")
        return None, None, None, None, None, None
    
    print(f"📂 Chargement des datasets séparés...")
    
    try:
        X_train = pd.read_pickle(split_dir / "X_train.pkl")
        X_val = pd.read_pickle(split_dir / "X_val.pkl")
        X_test = pd.read_pickle(split_dir / "X_test.pkl")
        y_train = pd.read_pickle(split_dir / "y_train.pkl")
        y_val = pd.read_pickle(split_dir / "y_val.pkl")
        y_test = pd.read_pickle(split_dir / "y_test.pkl")
        
        print(f"✅ Datasets chargés avec succès !")
        print(f"   🔹 Train : X{X_train.shape}, y{y_train.shape}")
        print(f"   🔹 Val   : X{X_val.shape}, y{y_val.shape}")
        print(f"   🔹 Test  : X{X_test.shape}, y{y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None, None, None, None, None, None

def analyze_missing_values(X_train, X_val, X_test):
    """
    🔍 ANALYSE DES VALEURS MANQUANTES
    """
    
    print(f"\n🔍 ANALYSE DES VALEURS MANQUANTES")
    print("=" * 40)
    
    datasets = {'Train': X_train, 'Val': X_val, 'Test': X_test}
    
    for name, X in datasets.items():
        missing = X.isnull().sum()
        missing_pct = (missing / len(X)) * 100
        
        print(f"\n📊 {name} :")
        if missing.sum() == 0:
            print("   ✅ Aucune valeur manquante")
        else:
            for col, count in missing[missing > 0].items():
                print(f"   ⚠️  {col:<20} : {count:3d} ({missing_pct[col]:4.1f}%)")
    
    return missing

def preprocess_features(X_train, X_val, X_test):
    """
    🔧 PREPROCESSING DES FEATURES
    
    📋 ÉTAPES :
    1. Imputation des valeurs manquantes
    2. Identification des variables catégorielles
    3. Standardisation des variables numériques
    4. Encodage des variables catégorielles
    """
    
    print(f"\n🔧 PREPROCESSING DES FEATURES")
    print("=" * 40)
    
    # === 1. IMPUTATION DES VALEURS MANQUANTES ===
    print("🔄 Étape 1 : Imputation des valeurs manquantes...")
    
    # Imputation par la médiane pour les variables numériques
    imputer = SimpleImputer(strategy='median')
    
    # Fit sur train seulement
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Transform sur val et test
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("   ✅ Imputation terminée")
    
    # === 2. IDENTIFICATION DES VARIABLES ===
    print("🔄 Étape 2 : Identification des types de variables...")
    
    # Variables catégorielles (valeurs discrètes limitées)
    categorical_cols = []
    numerical_cols = []
    
    for col in X_train_imputed.columns:
        unique_values = X_train_imputed[col].nunique()
        if unique_values <= 5:  # Seuil pour catégorielle
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    print(f"   📊 Variables numériques ({len(numerical_cols)}) : {numerical_cols}")
    print(f"   📊 Variables catégorielles ({len(categorical_cols)}) : {categorical_cols}")
    
    # === 3. STANDARDISATION DES VARIABLES NUMÉRIQUES ===
    print("🔄 Étape 3 : Standardisation des variables numériques...")
    
    scaler = StandardScaler()
    
    # Copie des dataframes pour éviter les modifications
    X_train_processed = X_train_imputed.copy()
    X_val_processed = X_val_imputed.copy()
    X_test_processed = X_test_imputed.copy()
    
    if numerical_cols:
        # Fit sur train seulement
        X_train_processed[numerical_cols] = scaler.fit_transform(X_train_imputed[numerical_cols])
        X_val_processed[numerical_cols] = scaler.transform(X_val_imputed[numerical_cols])
        X_test_processed[numerical_cols] = scaler.transform(X_test_imputed[numerical_cols])
        print("   ✅ Standardisation terminée")
    else:
        print("   ⚠️  Aucune variable numérique à standardiser")
    
    # === 4. ENCODAGE DES VARIABLES CATÉGORIELLES ===
    print("🔄 Étape 4 : Encodage des variables catégorielles...")
    
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit sur toutes les valeurs possibles (train + val + test)
        all_values = pd.concat([
            X_train_processed[col], 
            X_val_processed[col], 
            X_test_processed[col]
        ]).dropna()
        
        le.fit(all_values)
        
        # Transform chaque dataset
        X_train_processed[col] = le.transform(X_train_processed[col])
        X_val_processed[col] = le.transform(X_val_processed[col])
        X_test_processed[col] = le.transform(X_test_processed[col])
        
        label_encoders[col] = le
        print(f"   ✅ {col} : {len(le.classes_)} classes")
    
    # === SAUVEGARDE DES PREPROCESSORS ===
    preprocessors = {
        'imputer': imputer,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }
    
    return X_train_processed, X_val_processed, X_test_processed, preprocessors

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, preprocessors):
    """
    💾 SAUVEGARDE DES DONNÉES PREPROCESSÉES
    """
    
    # Création du dossier preprocessed
    preprocessed_dir = Path("../data/preprocessed")
    preprocessed_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Sauvegarde des données preprocessées...")
    
    # Sauvegarde des datasets
    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    for split_name, (X, y) in datasets.items():
        # Features
        X.to_csv(preprocessed_dir / f"X_{split_name}.csv", index=False)
        X.to_pickle(preprocessed_dir / f"X_{split_name}.pkl")
        
        # Target
        y.to_csv(preprocessed_dir / f"y_{split_name}.csv", index=False)
        y.to_pickle(preprocessed_dir / f"y_{split_name}.pkl")
        
        print(f"   ✅ {split_name.upper():5s} : X{X.shape}, y{y.shape}")
    
    # Sauvegarde des preprocessors
    preprocessor_path = preprocessed_dir / "preprocessor.pkl"
    joblib.dump(preprocessors, preprocessor_path)
    print(f"   ✅ Preprocessors : {preprocessor_path}")
    
    # Métadonnées
    preprocessing_metadata = {
        'imputation_strategy': 'median',
        'scaling_method': 'StandardScaler',
        'encoding_method': 'LabelEncoder',
        'numerical_features': preprocessors['numerical_cols'],
        'categorical_features': preprocessors['categorical_cols'],
        'preprocessing_date': pd.Timestamp.now().isoformat(),
        'train_shape': X_train.shape,
        'val_shape': X_val.shape,
        'test_shape': X_test.shape
    }
    
    metadata_path = preprocessed_dir / "preprocessing_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(preprocessing_metadata, f, indent=2)
    
    print(f"   ✅ Métadonnées : {metadata_path}")

def verify_preprocessing(X_train, X_val, X_test):
    """
    🔍 VÉRIFICATION DU PREPROCESSING
    """
    
    print(f"\n🔍 VÉRIFICATION DU PREPROCESSING")
    print("=" * 40)
    
    # Vérification des valeurs manquantes
    missing_train = X_train.isnull().sum().sum()
    missing_val = X_val.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    
    print(f"   ✅ Valeurs manquantes : Train={missing_train}, Val={missing_val}, Test={missing_test}")
    
    # Vérification de la standardisation (moyennes proches de 0)
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        train_means = X_train[numerical_cols].mean()
        print(f"   📊 Moyennes train (doivent être ~0) : {train_means.abs().max():.4f}")
    
    print(f"   ✅ Preprocessing validé !")

def main():
    """
    🚀 FONCTION PRINCIPALE
    """
    
    print("🚀 PREPROCESSING DES DONNÉES PMR")
    print("=" * 50)
    
    # 1. Chargement des datasets séparés
    X_train, X_val, X_test, y_train, y_val, y_test = load_split_datasets()
    if X_train is None:
        return
    
    # 2. Analyse des valeurs manquantes
    analyze_missing_values(X_train, X_val, X_test)
    
    # 3. Preprocessing des features
    X_train_processed, X_val_processed, X_test_processed, preprocessors = preprocess_features(
        X_train, X_val, X_test
    )
    
    # 4. Vérification du preprocessing
    verify_preprocessing(X_train_processed, X_val_processed, X_test_processed)
    
    # 5. Sauvegarde
    save_preprocessed_data(
        X_train_processed, X_val_processed, X_test_processed,
        y_train, y_val, y_test, preprocessors
    )
    
    print(f"\n🎉 PREPROCESSING TERMINÉ AVEC SUCCÈS !")
    print(f"📁 Données prêtes dans /data/preprocessed/")
    print(f"🤖 Prêt pour l'entraînement des modèles ML/DL !")

if __name__ == "__main__":
    main()
