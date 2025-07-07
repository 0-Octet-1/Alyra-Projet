# ===================================================================
# 📊 GÉNÉRATION DES DONNÉES BRUTES PMR
# ===================================================================

"""
🎯 ÉTAPE 1 : GÉNÉRATION DU DATASET BRUT COMPLET

📋 OBJECTIF :
- Générer un fichier unique de données brutes PMR
- Simulation réaliste d'accessibilité urbaine
- Sauvegarde en format CSV et PKL pour traçabilité
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def generate_raw_pmr_dataset(n_samples=4000):
    """
    🏗️ GÉNÉRATION DU DATASET BRUT PMR
    
    📋 VARIABLES GÉNÉRÉES :
    - largeur_passage : Largeur des passages (cm)
    - hauteur_marche : Hauteur des marches (cm)
    - presence_rampe : Présence d'une rampe (0/1)
    - pente_rampe : Pente de la rampe (%)
    - surface_sol : Type de surface (0=lisse, 1=rugueuse, 2=irrégulière)
    - eclairage : Qualité éclairage (0=faible, 1=moyen, 2=bon)
    - signalisation : Présence signalisation PMR (0/1)
    - places_parking : Nombre places parking PMR
    - distance_transport : Distance transport public (m)
    - assistance_disponible : Assistance disponible (0/1)
    - horaires_adaptes : Horaires adaptés PMR (0/1)
    
    🎯 TARGET :
    - accessibilite_pmr : 0=Difficile, 1=Modérée, 2=Facile
    """
    
    print(f"🏗️ Génération de {n_samples} échantillons PMR...")
    
    data = {}
    
    # === VARIABLES PHYSIQUES ===
    data['largeur_passage'] = np.random.normal(120, 30, n_samples)  # cm
    data['largeur_passage'] = np.clip(data['largeur_passage'], 60, 200)
    
    data['hauteur_marche'] = np.random.exponential(8, n_samples)  # cm
    data['hauteur_marche'] = np.clip(data['hauteur_marche'], 0, 25)
    
    data['presence_rampe'] = np.random.binomial(1, 0.6, n_samples)
    
    # Pente rampe (seulement si rampe présente)
    data['pente_rampe'] = np.where(
        data['presence_rampe'] == 1,
        np.random.normal(6, 2, n_samples),
        0
    )
    data['pente_rampe'] = np.clip(data['pente_rampe'], 0, 15)
    
    # === VARIABLES ENVIRONNEMENTALES ===
    data['surface_sol'] = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
    data['eclairage'] = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.5, 0.3])
    data['signalisation'] = np.random.binomial(1, 0.7, n_samples)
    
    # === VARIABLES DE SERVICE ===
    data['places_parking'] = np.random.poisson(2, n_samples)
    data['places_parking'] = np.clip(data['places_parking'], 0, 10)
    
    data['distance_transport'] = np.random.exponential(150, n_samples)  # mètres
    data['distance_transport'] = np.clip(data['distance_transport'], 10, 500)
    
    data['assistance_disponible'] = np.random.binomial(1, 0.4, n_samples)
    data['horaires_adaptes'] = np.random.binomial(1, 0.8, n_samples)
    
    # === GÉNÉRATION DU TARGET (LOGIQUE RÉALISTE) ===
    # Score d'accessibilité basé sur les variables
    score_accessibilite = (
        (data['largeur_passage'] - 60) / 140 * 0.2 +  # Largeur
        (25 - data['hauteur_marche']) / 25 * 0.2 +     # Marches
        data['presence_rampe'] * 0.15 +                # Rampe
        (15 - data['pente_rampe']) / 15 * 0.1 +        # Pente
        data['surface_sol'] / 2 * 0.1 +                # Surface
        data['eclairage'] / 2 * 0.05 +                 # Éclairage
        data['signalisation'] * 0.05 +                 # Signalisation
        np.clip(data['places_parking'] / 5, 0, 1) * 0.05 +  # Parking
        (500 - data['distance_transport']) / 490 * 0.05 +   # Transport
        data['assistance_disponible'] * 0.03 +         # Assistance
        data['horaires_adaptes'] * 0.02                # Horaires
    )
    
    # Conversion en classes avec seuils
    data['accessibilite_pmr'] = np.where(
        score_accessibilite < 0.4, 0,  # Difficile
        np.where(score_accessibilite < 0.7, 1, 2)  # Modérée, Facile
    )
    
    # === AJOUT DE VALEURS MANQUANTES RÉALISTES ===
    missing_rate = 0.014  # 1.4% comme dans l'analyse précédente
    
    for col in ['largeur_passage', 'pente_rampe', 'distance_transport']:
        n_missing = int(n_samples * missing_rate)
        missing_idx = np.random.choice(n_samples, n_missing, replace=False)
        data[col][missing_idx] = np.nan
    
    # === CRÉATION DU DATAFRAME ===
    df = pd.DataFrame(data)
    
    print("✅ Dataset généré avec succès !")
    print(f"📊 Shape : {df.shape}")
    print(f"🎯 Distribution des classes :")
    for classe, count in df['accessibilite_pmr'].value_counts().sort_index().items():
        labels = {0: 'Difficile', 1: 'Modérée', 2: 'Facile'}
        pct = count / len(df) * 100
        print(f"   {classe} ({labels[classe]:<8}) : {count:4d} ({pct:5.1f}%)")
    
    print(f"❓ Valeurs manquantes : {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.2f}%)")
    
    return df

def save_raw_dataset(df):
    """
    💾 SAUVEGARDE DU DATASET BRUT
    """
    
    # Création du dossier raw
    raw_dir = Path("../data/raw")
    raw_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Sauvegarde du dataset brut...")
    
    # Sauvegarde en CSV (lisible)
    csv_path = raw_dir / "pmr_dataset_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"   ✅ CSV : {csv_path}")
    
    # Sauvegarde en PKL (efficace)
    pkl_path = raw_dir / "pmr_dataset_raw.pkl"
    df.to_pickle(pkl_path)
    print(f"   ✅ PKL : {pkl_path}")
    
    # Métadonnées
    metadata = {
        'n_samples': int(len(df)),
        'n_features': int(len(df.columns) - 1),  # -1 pour le target
        'target_column': 'accessibilite_pmr',
        'missing_values': int(df.isnull().sum().sum()),
        'generation_date': pd.Timestamp.now().isoformat(),
        'random_state': int(RANDOM_STATE)
    }
    
    metadata_path = raw_dir / "dataset_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Métadonnées : {metadata_path}")
    
    return csv_path, pkl_path

def main():
    """
    🚀 FONCTION PRINCIPALE
    """
    
    print("🚀 GÉNÉRATION DU DATASET BRUT PMR")
    print("=" * 50)
    
    # 1. Génération des données
    df_raw = generate_raw_pmr_dataset(n_samples=4000)
    
    # 2. Sauvegarde
    csv_path, pkl_path = save_raw_dataset(df_raw)
    
    print(f"\n🎉 DATASET BRUT CRÉÉ AVEC SUCCÈS !")
    print(f"📁 Fichiers disponibles :")
    print(f"   📄 CSV : {csv_path}")
    print(f"   📦 PKL : {pkl_path}")
    print(f"🔄 Prêt pour l'étape de split et preprocessing !")

if __name__ == "__main__":
    main()
