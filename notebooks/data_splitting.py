# ===================================================================
# 🔄 SÉPARATION DES DONNÉES PMR
# ===================================================================

"""
🎯 ÉTAPE 2 : SPLIT TRAIN/VALIDATION/TEST

📋 OBJECTIF :
- Charger le dataset brut unique
- Séparer en train/validation/test (70/15/15%)
- Split stratifié pour équilibrer les classes
- Sauvegarde séparée pour chaque ensemble
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_raw_dataset():
    """
    📂 CHARGEMENT DU DATASET BRUT
    """
    
    raw_path = Path("../data/raw/pmr_dataset_raw.pkl")
    
    if not raw_path.exists():
        print("❌ Dataset brut non trouvé !")
        print("🔧 Exécutez d'abord : python data_generation.py")
        return None
    
    print(f"📂 Chargement du dataset brut...")
    df = pd.read_pickle(raw_path)
    
    print(f"✅ Dataset chargé : {df.shape}")
    print(f"📊 Colonnes : {list(df.columns)}")
    print(f"🎯 Target : {df['accessibilite_pmr'].value_counts().sort_index().to_dict()}")
    
    return df

def split_dataset(df, test_size=0.15, val_size=0.15):
    """
    ✂️ SÉPARATION STRATIFIÉE DES DONNÉES
    
    📋 STRATÉGIE :
    - 70% Train (entraînement des modèles)
    - 15% Validation (optimisation hyperparamètres)
    - 15% Test (évaluation finale)
    - Split stratifié pour maintenir la distribution des classes
    """
    
    print(f"\n✂️ Séparation des données...")
    print(f"📊 Répartition cible : Train 70%, Val 15%, Test 15%")
    
    # Séparation features/target
    X = df.drop('accessibilite_pmr', axis=1)
    y = df['accessibilite_pmr']
    
    print(f"📋 Features : {X.shape}")
    print(f"🎯 Target : {y.shape}")
    
    # Premier split : Train+Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    # Deuxième split : Train vs Val
    # val_size ajusté pour obtenir 15% du total
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
    
    # Vérification des dimensions
    total_samples = len(df)
    print(f"\n📊 RÉSULTATS DU SPLIT :")
    print(f"   🔹 Train : {X_train.shape[0]:4d} échantillons ({X_train.shape[0]/total_samples*100:5.1f}%)")
    print(f"   🔹 Val   : {X_val.shape[0]:4d} échantillons ({X_val.shape[0]/total_samples*100:5.1f}%)")
    print(f"   🔹 Test  : {X_test.shape[0]:4d} échantillons ({X_test.shape[0]/total_samples*100:5.1f}%)")
    
    # Vérification de la stratification
    print(f"\n🎯 VÉRIFICATION STRATIFICATION :")
    for dataset_name, y_set in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = y_set.value_counts(normalize=True).sort_index()
        print(f"   {dataset_name:5s} : ", end="")
        for classe, prop in dist.items():
            print(f"Cl{classe}={prop:.1%} ", end="")
        print()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_split_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    💾 SAUVEGARDE DES DATASETS SÉPARÉS
    """
    
    # Création du dossier split
    split_dir = Path("../data/split")
    split_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Sauvegarde des datasets séparés...")
    
    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    saved_files = []
    
    for split_name, (X, y) in datasets.items():
        # Sauvegarde features
        X_path_csv = split_dir / f"X_{split_name}.csv"
        X_path_pkl = split_dir / f"X_{split_name}.pkl"
        X.to_csv(X_path_csv, index=False)
        X.to_pickle(X_path_pkl)
        
        # Sauvegarde target
        y_path_csv = split_dir / f"y_{split_name}.csv"
        y_path_pkl = split_dir / f"y_{split_name}.pkl"
        y.to_csv(y_path_csv, index=False)
        y.to_pickle(y_path_pkl)
        
        print(f"   ✅ {split_name.upper():5s} : X{X.shape}, y{y.shape}")
        saved_files.extend([X_path_csv, X_path_pkl, y_path_csv, y_path_pkl])
    
    # Métadonnées du split
    split_metadata = {
        'split_strategy': 'stratified',
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'train_ratio': len(X_train) / (len(X_train) + len(X_val) + len(X_test)),
        'val_ratio': len(X_val) / (len(X_train) + len(X_val) + len(X_test)),
        'test_ratio': len(X_test) / (len(X_train) + len(X_val) + len(X_test)),
        'random_state': RANDOM_STATE,
        'split_date': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = split_dir / "split_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    print(f"   ✅ Métadonnées : {metadata_path}")
    
    return saved_files

def verify_split_integrity(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    🔍 VÉRIFICATION DE L'INTÉGRITÉ DU SPLIT
    """
    
    print(f"\n🔍 VÉRIFICATION DE L'INTÉGRITÉ...")
    
    # Vérification des dimensions
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Nombre de features différent !"
    assert len(X_train) == len(y_train), "Tailles X_train et y_train différentes !"
    assert len(X_val) == len(y_val), "Tailles X_val et y_val différentes !"
    assert len(X_test) == len(y_test), "Tailles X_test et y_test différentes !"
    
    # Vérification des valeurs manquantes
    total_missing = (
        X_train.isnull().sum().sum() + 
        X_val.isnull().sum().sum() + 
        X_test.isnull().sum().sum()
    )
    
    print(f"   ✅ Dimensions cohérentes")
    print(f"   ✅ Correspondance X/y")
    print(f"   ⚠️  Valeurs manquantes : {total_missing}")
    print(f"   ✅ Split prêt pour preprocessing !")

def main():
    """
    🚀 FONCTION PRINCIPALE
    """
    
    print("🚀 SÉPARATION DU DATASET PMR")
    print("=" * 50)
    
    # 1. Chargement du dataset brut
    df = load_raw_dataset()
    if df is None:
        return
    
    # 2. Séparation stratifiée
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    
    # 3. Vérification de l'intégrité
    verify_split_integrity(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # 4. Sauvegarde
    saved_files = save_split_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print(f"\n🎉 SÉPARATION TERMINÉE AVEC SUCCÈS !")
    print(f"📁 {len(saved_files)} fichiers créés dans /data/split/")
    print(f"🔄 Prêt pour l'étape de preprocessing !")

if __name__ == "__main__":
    main()
