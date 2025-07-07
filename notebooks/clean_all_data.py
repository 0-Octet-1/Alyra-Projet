# ===================================================================
# 🧹 NETTOYAGE COMPLET DES DONNÉES PMR
# ===================================================================

"""
🎯 SCRIPT DE NETTOYAGE COMPLET

📋 OBJECTIF :
- Supprimer tous les fichiers de données générés
- Nettoyer les dossiers data/raw, data/split, data/preprocessed
- Supprimer les modèles entraînés
- Remettre le projet à l'état initial pour recommencer

⚠️  ATTENTION : Ce script supprime DÉFINITIVEMENT tous les fichiers générés !
"""

import os
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def clean_directory(dir_path, description):
    """
    🗑️ NETTOYAGE D'UN DOSSIER
    """
    
    if dir_path.exists():
        print(f"🧹 Nettoyage : {description}")
        
        # Lister les fichiers avant suppression
        files = list(dir_path.glob("*"))
        if files:
            print(f"   📁 {len(files)} fichiers trouvés :")
            for file in files:
                print(f"      - {file.name}")
            
            # Suppression
            shutil.rmtree(dir_path)
            print(f"   ✅ Dossier supprimé : {dir_path}")
        else:
            print(f"   ✅ Dossier déjà vide : {dir_path}")
    else:
        print(f"   ✅ Dossier inexistant : {dir_path}")

def clean_files_in_directory(dir_path, description, patterns=None):
    """
    🗑️ NETTOYAGE DE FICHIERS SPÉCIFIQUES DANS UN DOSSIER
    """
    
    if not dir_path.exists():
        print(f"   ✅ Dossier inexistant : {dir_path}")
        return
    
    print(f"🧹 Nettoyage : {description}")
    
    if patterns is None:
        patterns = ["*"]
    
    files_deleted = 0
    for pattern in patterns:
        files = list(dir_path.glob(pattern))
        for file in files:
            if file.is_file():
                print(f"   🗑️  Suppression : {file.name}")
                file.unlink()
                files_deleted += 1
    
    if files_deleted == 0:
        print(f"   ✅ Aucun fichier à supprimer")
    else:
        print(f"   ✅ {files_deleted} fichiers supprimés")

def main():
    """
    🚀 FONCTION PRINCIPALE DE NETTOYAGE
    """
    
    print("🧹 NETTOYAGE COMPLET DU PROJET PMR")
    print("=" * 50)
    print("⚠️  ATTENTION : Suppression définitive des données générées !")
    
    # Confirmation de l'utilisateur
    response = input("\n❓ Êtes-vous sûr de vouloir tout supprimer ? (oui/non) : ").lower().strip()
    
    if response not in ['oui', 'o', 'yes', 'y']:
        print("❌ Nettoyage annulé par l'utilisateur")
        return
    
    print("\n🚀 DÉBUT DU NETTOYAGE...")
    print("=" * 30)
    
    # === 1. NETTOYAGE DES DONNÉES BRUTES ===
    raw_dir = Path("../data/raw")
    clean_directory(raw_dir, "Données brutes (/data/raw/)")
    
    # === 2. NETTOYAGE DES DONNÉES SÉPARÉES ===
    split_dir = Path("../data/split")
    clean_directory(split_dir, "Données séparées (/data/split/)")
    
    # === 3. NETTOYAGE DES DONNÉES PREPROCESSÉES ===
    preprocessed_dir = Path("../data/preprocessed")
    clean_directory(preprocessed_dir, "Données preprocessées (/data/preprocessed/)")
    
    # === 4. NETTOYAGE DES ANCIENNES DONNÉES PROCESSED ===
    processed_dir = Path("../data/processed")
    if processed_dir.exists():
        clean_files_in_directory(
            processed_dir, 
            "Anciennes données processed (/data/processed/)",
            ["*.pkl", "*.csv", "*.png"]
        )
    
    # === 5. NETTOYAGE DES MODÈLES ===
    models_dir = Path("../models")
    if models_dir.exists():
        clean_files_in_directory(
            models_dir,
            "Modèles entraînés (/models/)",
            ["*.pkl", "*.h5", "*.json", "*.txt"]
        )
    
    # === 6. NETTOYAGE DES SCRIPTS TEMPORAIRES ===
    notebooks_dir = Path(".")
    clean_files_in_directory(
        notebooks_dir,
        "Scripts temporaires ML (/notebooks/)",
        ["ml_training_part*.py", "ml_training_complete.py"]
    )
    
    # === 7. NETTOYAGE DES LOGS ET CACHES ===
    cache_patterns = [
        "__pycache__",
        "*.pyc", 
        ".ipynb_checkpoints",
        "*.log"
    ]
    
    print(f"🧹 Nettoyage : Caches et fichiers temporaires")
    for pattern in cache_patterns:
        files = list(Path("..").rglob(pattern))
        for file in files:
            if file.is_file():
                print(f"   🗑️  Suppression : {file}")
                file.unlink()
            elif file.is_dir():
                print(f"   🗑️  Suppression dossier : {file}")
                shutil.rmtree(file)
    
    print("\n" + "=" * 50)
    print("🎉 NETTOYAGE TERMINÉ AVEC SUCCÈS !")
    print("=" * 50)
    
    print("\n📋 ÉTAT APRÈS NETTOYAGE :")
    print("   ✅ Données brutes supprimées")
    print("   ✅ Données séparées supprimées") 
    print("   ✅ Données preprocessées supprimées")
    print("   ✅ Modèles ML/DL supprimés")
    print("   ✅ Scripts temporaires supprimés")
    print("   ✅ Caches nettoyés")
    
    print("\n🚀 PRÊT POUR UN NOUVEAU DÉPART !")
    print("📝 Pour recommencer :")
    print("   1. python data_generation.py")
    print("   2. python data_splitting.py") 
    print("   3. python data_preprocessing.py")
    print("   4. python ml_training_complete.py")

if __name__ == "__main__":
    main()
