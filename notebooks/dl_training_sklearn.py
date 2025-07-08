#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement Deep Learning avec scikit-learn MLPClassifier
Projet PMR - Certification Développeur IA

Alternative plus stable utilisant MLPClassifier pour les réseaux de neurones
Compatible avec l'environnement existant

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

# Imports Deep Learning avec scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Imports pour évaluation
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des chemins
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'preprocessed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Créer le dossier models
os.makedirs(MODELS_DIR, exist_ok=True)

def print_section(title):
    """Affiche une section avec formatage"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def load_data():
    """Charge les données préprocessées"""
    print_section("CHARGEMENT DES DONNÉES POUR DEEP LEARNING")
    
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
    
    # Combiner train et validation pour l'entraînement final
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])
    
    print(f"✅ Données chargées et préparées:")
    print(f"   - Train complet: {X_train_full.shape[0]} échantillons, {X_train_full.shape[1]} features")
    print(f"   - Test: {X_test.shape[0]} échantillons")
    print(f"   - Classes uniques: {np.unique(y_train_full)}")
    
    return X_train, X_val, X_test, X_train_full, y_train, y_val, y_test, y_train_full

def create_mlp_models():
    """Définit les architectures MLP à tester"""
    models = {
        'Simple MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42
        ),
        'Deep MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42
        ),
        'Wide MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42
        ),
        'Optimized MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='tanh',
            solver='lbfgs',
            alpha=0.01,
            max_iter=1000,
            random_state=42
        )
    }
    
    return models

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Entraîne et évalue un modèle MLP"""
    print_section(f"ENTRAÎNEMENT {model_name.upper()}")
    
    print(f"🔄 Entraînement en cours...")
    print(f"   Architecture: {model.hidden_layer_sizes}")
    print(f"   Solver: {model.solver}")
    print(f"   Activation: {model.activation}")
    
    # Entraînement
    model.fit(X_train, y_train)
    
    print(f"✅ {model_name} entraîné avec succès !")
    print(f"   Nombre d'itérations: {model.n_iter_}")
    if hasattr(model, 'loss_'):
        print(f"   Loss finale: {model.loss_:.6f}")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    
    print(f"\n📊 Métriques de performance:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - F1-score macro: {f1_macro:.4f}")
    print(f"   - F1-score weighted: {f1_weighted:.4f}")
    print(f"   - Precision macro: {precision_macro:.4f}")
    print(f"   - Recall macro: {recall_macro:.4f}")
    
    # Rapport de classification
    print(f"\n📋 Rapport de classification:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Matrice de confusion:")
    print(cm)
    
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'confusion_matrix_mlp_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # Courbe de perte si disponible
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_)
        plt.title(f'{model_name} - Courbe de perte')
        plt.xlabel('Itération')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f'loss_curve_mlp_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm.tolist(),
        'n_iterations': model.n_iter_,
        'final_loss': model.loss_ if hasattr(model, 'loss_') else None
    }

def optimize_best_mlp(X_train, y_train, X_test, y_test):
    """Optimise le meilleur modèle MLP avec GridSearch"""
    print_section("OPTIMISATION HYPERPARAMÈTRES MLP")
    
    # Paramètres à optimiser
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (128, 64), (100, 50, 25)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    
    # Modèle de base
    mlp = MLPClassifier(
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42
    )
    
    print("🔍 Recherche des meilleurs hyperparamètres...")
    print("   Cela peut prendre quelques minutes...")
    
    # GridSearch avec validation croisée
    grid_search = GridSearchCV(
        mlp, 
        param_grid, 
        cv=3, 
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Optimisation terminée !")
    print(f"🏆 Meilleurs paramètres: {grid_search.best_params_}")
    print(f"📊 Meilleur score CV: {grid_search.best_score_:.4f}")
    
    # Évaluer le modèle optimisé
    best_model = grid_search.best_estimator_
    optimized_results = train_and_evaluate_model(
        best_model, X_train, y_train, X_test, y_test, "Optimized MLP (GridSearch)"
    )
    
    return optimized_results

def compare_with_ml_models(mlp_results):
    """Compare les résultats MLP avec les modèles ML existants"""
    print_section("COMPARAISON ML vs MLP (Deep Learning)")
    
    # Charger les résultats ML existants
    ml_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('ml_results_') and f.endswith('.json')]
    
    if ml_files:
        latest_ml_file = sorted(ml_files)[-1]
        ml_path = os.path.join(MODELS_DIR, latest_ml_file)
        
        with open(ml_path, 'r') as f:
            ml_results = json.load(f)
        
        print("📊 COMPARAISON DES PERFORMANCES:")
        print(f"{'Modèle':<25} {'Accuracy':<10} {'F1-macro':<10} {'F1-weighted':<12}")
        print("-" * 57)
        
        # Modèles ML
        if 'random_forest' in ml_results:
            rf = ml_results['random_forest']
            print(f"{'Random Forest (ML)':<25} {rf['accuracy']:<10.4f} {rf['f1_macro']:<10.4f} {rf['f1_weighted']:<12.4f}")
        
        if 'logistic_regression' in ml_results:
            lr = ml_results['logistic_regression']
            print(f"{'Logistic Regression (ML)':<25} {lr['accuracy']:<10.4f} {lr['f1_macro']:<10.4f} {lr['f1_weighted']:<12.4f}")
        
        # Modèles MLP
        for model_name, results in mlp_results.items():
            if model_name != 'model':  # Exclure l'objet modèle
                display_name = f"{model_name} (MLP)"
                print(f"{display_name:<25} {results['accuracy']:<10.4f} {results['f1_macro']:<10.4f} {results['f1_weighted']:<12.4f}")
        
        # Identifier le meilleur modèle global
        all_models = {}
        if 'random_forest' in ml_results:
            all_models['Random Forest (ML)'] = ml_results['random_forest']['f1_macro']
        if 'logistic_regression' in ml_results:
            all_models['Logistic Regression (ML)'] = ml_results['logistic_regression']['f1_macro']
        
        for model_name, results in mlp_results.items():
            if model_name != 'model':
                all_models[f"{model_name} (MLP)"] = results['f1_macro']
        
        best_model = max(all_models, key=all_models.get)
        best_f1 = all_models[best_model]
        
        print(f"\n🏆 MEILLEUR MODÈLE GLOBAL: {best_model}")
        print(f"📊 F1-score macro: {best_f1:.4f}")
        
        # Vérifier l'objectif
        target_f1 = 0.75
        print(f"\n🎯 Objectif F1-score ≥ {target_f1:.0%}:")
        for model_name, f1_score in all_models.items():
            status = "✅" if f1_score >= target_f1 else "❌"
            print(f"   {status} {model_name}: {f1_score:.1%}")
        
        return best_model, best_f1
    else:
        print("❌ Aucun résultat ML trouvé pour comparaison")
        return None, None

def save_mlp_results(mlp_results):
    """Sauvegarde les modèles et résultats MLP"""
    print_section("SAUVEGARDE DES MODÈLES MLP")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Préparer les résultats pour sauvegarde
    results_to_save = {}
    models_to_save = {}
    
    for model_name, results in mlp_results.items():
        if isinstance(results, dict) and 'model' in results:
            # Sauvegarder le modèle
            model = results['model']
            model_filename = f'mlp_{model_name.lower().replace(" ", "_")}_{timestamp}.pkl'
            model_path = os.path.join(MODELS_DIR, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            models_to_save[model_name] = model_filename
            print(f"✅ {model_name} sauvegardé: {model_filename}")
            
            # Préparer les résultats (sans l'objet modèle)
            results_copy = results.copy()
            del results_copy['model']
            results_to_save[model_name] = results_copy
    
    # Sauvegarder les résultats
    results_with_timestamp = {
        'timestamp': timestamp,
        'models': results_to_save,
        'model_files': models_to_save,
        'training_info': {
            'framework': 'scikit-learn MLPClassifier',
            'max_iterations': 500,
            'early_stopping': True,
            'validation_fraction': 0.2
        }
    }
    
    results_filename = f'mlp_results_{timestamp}.json'
    results_path = os.path.join(MODELS_DIR, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results_with_timestamp, f, indent=2, default=str)
    
    print(f"✅ Résultats MLP sauvegardés: {results_filename}")

def main():
    """Fonction principale"""
    print_section("ENTRAÎNEMENT MLP (DEEP LEARNING) - PROJET PMR")
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Charger les données
    X_train, X_val, X_test, X_train_full, y_train, y_val, y_test, y_train_full = load_data()
    
    # 2. Créer les modèles MLP
    mlp_models = create_mlp_models()
    
    # 3. Entraîner et évaluer chaque modèle
    mlp_results = {}
    
    for model_name, model in mlp_models.items():
        results = train_and_evaluate_model(
            model, X_train_full, y_train_full, X_test, y_test, model_name
        )
        mlp_results[model_name] = results
    
    # 4. Optimiser le meilleur modèle
    print("\n" + "="*60)
    print("  OPTIMISATION DU MEILLEUR MODÈLE")
    print("="*60)
    
    # Trouver le meilleur modèle actuel
    best_current = max(mlp_results, key=lambda x: mlp_results[x]['f1_macro'])
    print(f"🏆 Meilleur modèle actuel: {best_current}")
    print(f"📊 F1-score: {mlp_results[best_current]['f1_macro']:.4f}")
    
    # Optimiser avec GridSearch
    try:
        optimized_results = optimize_best_mlp(X_train_full, y_train_full, X_test, y_test)
        mlp_results['Optimized MLP (GridSearch)'] = optimized_results
    except Exception as e:
        print(f"⚠️ Erreur lors de l'optimisation: {e}")
        print("Continuons avec les modèles existants...")
    
    # 5. Comparer les modèles MLP entre eux
    print_section("COMPARAISON DES MODÈLES MLP")
    best_mlp_model = max(mlp_results, key=lambda x: mlp_results[x]['f1_macro'])
    best_mlp_f1 = mlp_results[best_mlp_model]['f1_macro']
    
    print(f"🏆 Meilleur modèle MLP: {best_mlp_model}")
    print(f"📊 F1-score macro: {best_mlp_f1:.4f}")
    
    # 6. Comparer avec les modèles ML
    best_global_model, best_global_f1 = compare_with_ml_models(mlp_results)
    
    # 7. Sauvegarder tout
    save_mlp_results(mlp_results)
    
    print_section("ENTRAÎNEMENT MLP TERMINÉ")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Tous les modèles MLP ont été entraînés et sauvegardés")
    print(f"🏆 Meilleur modèle MLP: {best_mlp_model} (F1: {best_mlp_f1:.4f})")
    if best_global_model:
        print(f"🌟 Meilleur modèle global: {best_global_model} (F1: {best_global_f1:.4f})")
    
    # Vérifier l'objectif
    target_f1 = 0.75
    if best_mlp_f1 >= target_f1:
        print(f"🎯 ✅ OBJECTIF ATTEINT ! F1-score MLP ≥ {target_f1:.0%}")
    else:
        print(f"🎯 ❌ Objectif non atteint. F1-score MLP: {best_mlp_f1:.1%} < {target_f1:.0%}")

if __name__ == "__main__":
    main()
