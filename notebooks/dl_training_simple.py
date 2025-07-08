#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simplifié d'entraînement Deep Learning avec MLPClassifier
Projet PMR - Version robuste et stable

Auteur: 0-Octet-1
Date: 8 juillet 2025
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)

# Configuration des chemins
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'preprocessed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

def print_section(title):
    """Affiche une section avec formatage"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def load_data():
    """Charge les données préprocessées"""
    print_section("CHARGEMENT DES DONNÉES")
    
    try:
        # Charger les features
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'X_train.pkl'), 'rb') as f:
            X_train = pickle.load(f)
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'X_val.pkl'), 'rb') as f:
            X_val = pickle.load(f)
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'X_test.pkl'), 'rb') as f:
            X_test = pickle.load(f)
            
        # Charger les targets
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'y_val.pkl'), 'rb') as f:
            y_val = pickle.load(f)
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        
        # Convertir en arrays numpy si nécessaire
        if hasattr(X_train, 'values'):
            X_train = X_train.values
            X_val = X_val.values
            X_test = X_test.values
        
        # Combiner train et validation
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])
        
        print(f"✅ Données chargées:")
        print(f"   - Train: {X_train_full.shape[0]} échantillons, {X_train_full.shape[1]} features")
        print(f"   - Test: {X_test.shape[0]} échantillons")
        print(f"   - Classes: {np.unique(y_train_full)}")
        
        return X_train_full, X_test, y_train_full, y_test
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None, None, None

def train_simple_mlp(X_train, y_train, X_test, y_test):
    """Entraîne un MLP simple et robuste"""
    print_section("ENTRAÎNEMENT MLP SIMPLE")
    
    try:
        # Modèle MLP simple
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42,
            verbose=False
        )
        
        print("🔄 Entraînement en cours...")
        mlp.fit(X_train, y_train)
        
        print(f"✅ Entraînement terminé !")
        print(f"   Itérations: {mlp.n_iter_}")
        
        # Prédictions
        y_pred = mlp.predict(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        print(f"\n📊 RÉSULTATS MLP:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1-score macro: {f1_macro:.4f}")
        print(f"   - F1-score weighted: {f1_weighted:.4f}")
        print(f"   - Precision macro: {precision_macro:.4f}")
        print(f"   - Recall macro: {recall_macro:.4f}")
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Matrice de confusion:")
        print(cm)
        
        return {
            'model': mlp,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm.tolist(),
            'n_iterations': mlp.n_iter_
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return None

def train_deep_mlp(X_train, y_train, X_test, y_test):
    """Entraîne un MLP plus profond"""
    print_section("ENTRAÎNEMENT MLP PROFOND")
    
    try:
        # Modèle MLP profond
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42,
            verbose=False
        )
        
        print("🔄 Entraînement en cours...")
        mlp.fit(X_train, y_train)
        
        print(f"✅ Entraînement terminé !")
        print(f"   Itérations: {mlp.n_iter_}")
        
        # Prédictions
        y_pred = mlp.predict(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        
        print(f"\n📊 RÉSULTATS MLP PROFOND:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1-score macro: {f1_macro:.4f}")
        print(f"   - F1-score weighted: {f1_weighted:.4f}")
        print(f"   - Precision macro: {precision_macro:.4f}")
        print(f"   - Recall macro: {recall_macro:.4f}")
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Matrice de confusion:")
        print(cm)
        
        return {
            'model': mlp,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm.tolist(),
            'n_iterations': mlp.n_iter_
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return None

def compare_with_ml(mlp_results):
    """Compare avec les modèles ML existants"""
    print_section("COMPARAISON ML vs MLP")
    
    try:
        # Charger les résultats ML
        ml_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('ml_results_') and f.endswith('.json')]
        
        if ml_files:
            latest_ml_file = sorted(ml_files)[-1]
            ml_path = os.path.join(MODELS_DIR, latest_ml_file)
            
            with open(ml_path, 'r') as f:
                ml_results = json.load(f)
            
            print("📊 COMPARAISON DES PERFORMANCES:")
            print(f"{'Modèle':<20} {'Accuracy':<10} {'F1-macro':<10}")
            print("-" * 40)
            
            # Modèles ML
            if 'random_forest' in ml_results:
                rf = ml_results['random_forest']
                print(f"{'Random Forest':<20} {rf['accuracy']:<10.4f} {rf['f1_macro']:<10.4f}")
            
            if 'logistic_regression' in ml_results:
                lr = ml_results['logistic_regression']
                print(f"{'Logistic Regression':<20} {lr['accuracy']:<10.4f} {lr['f1_macro']:<10.4f}")
            
            # Modèles MLP
            for name, results in mlp_results.items():
                if results and 'f1_macro' in results:
                    print(f"{name:<20} {results['accuracy']:<10.4f} {results['f1_macro']:<10.4f}")
            
            # Meilleur modèle
            all_models = {}
            if 'random_forest' in ml_results:
                all_models['Random Forest'] = ml_results['random_forest']['f1_macro']
            if 'logistic_regression' in ml_results:
                all_models['Logistic Regression'] = ml_results['logistic_regression']['f1_macro']
            
            for name, results in mlp_results.items():
                if results and 'f1_macro' in results:
                    all_models[name] = results['f1_macro']
            
            if all_models:
                best_model = max(all_models, key=all_models.get)
                best_f1 = all_models[best_model]
                
                print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
                print(f"📊 F1-score: {best_f1:.4f}")
                
                # Objectif
                target_f1 = 0.75
                status = "✅" if best_f1 >= target_f1 else "❌"
                print(f"\n🎯 Objectif F1 ≥ 75%: {status} ({best_f1:.1%})")
                
                return best_model, best_f1
        
        return None, None
        
    except Exception as e:
        print(f"❌ Erreur lors de la comparaison: {e}")
        return None, None

def save_results(mlp_results):
    """Sauvegarde les résultats"""
    print_section("SAUVEGARDE")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les modèles
        for name, results in mlp_results.items():
            if results and 'model' in results:
                model_filename = f'mlp_{name.lower().replace(" ", "_")}_{timestamp}.pkl'
                model_path = os.path.join(MODELS_DIR, model_filename)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(results['model'], f)
                
                print(f"✅ {name} sauvegardé: {model_filename}")
        
        # Sauvegarder les résultats
        results_to_save = {}
        for name, results in mlp_results.items():
            if results:
                results_copy = results.copy()
                if 'model' in results_copy:
                    del results_copy['model']
                results_to_save[name] = results_copy
        
        results_data = {
            'timestamp': timestamp,
            'models': results_to_save,
            'framework': 'scikit-learn MLPClassifier'
        }
        
        results_filename = f'mlp_results_{timestamp}.json'
        results_path = os.path.join(MODELS_DIR, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"✅ Résultats sauvegardés: {results_filename}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")

def main():
    """Fonction principale"""
    print_section("DEEP LEARNING SIMPLIFIÉ - PROJET PMR")
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Charger les données
    X_train, X_test, y_train, y_test = load_data()
    
    if X_train is None:
        print("❌ Impossible de charger les données. Arrêt du script.")
        return
    
    # 2. Entraîner les modèles MLP
    mlp_results = {}
    
    # MLP Simple
    simple_results = train_simple_mlp(X_train, y_train, X_test, y_test)
    if simple_results:
        mlp_results['MLP Simple'] = simple_results
    
    # MLP Profond
    deep_results = train_deep_mlp(X_train, y_train, X_test, y_test)
    if deep_results:
        mlp_results['MLP Profond'] = deep_results
    
    # 3. Comparer les résultats
    if mlp_results:
        best_mlp = max(mlp_results, key=lambda x: mlp_results[x]['f1_macro'])
        print(f"\n🏆 Meilleur MLP: {best_mlp}")
        print(f"📊 F1-score: {mlp_results[best_mlp]['f1_macro']:.4f}")
        
        # 4. Comparer avec ML
        best_global, best_f1 = compare_with_ml(mlp_results)
        
        # 5. Sauvegarder
        save_results(mlp_results)
        
        print_section("TERMINÉ")
        print(f"✅ Entraînement Deep Learning terminé avec succès !")
        if best_global:
            print(f"🌟 Meilleur modèle global: {best_global} (F1: {best_f1:.4f})")
    else:
        print("❌ Aucun modèle MLP n'a pu être entraîné.")

if __name__ == "__main__":
    main()
