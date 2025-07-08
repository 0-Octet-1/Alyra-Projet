#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script complet d'entraînement des modèles de Machine Learning
Projet PMR - Certification Développeur IA

Ce script :
1. Vérifie la présence des données préprocessées
2. Exécute le pipeline de données si nécessaire
3. Entraîne les modèles ML (Random Forest, Logistic Regression)
4. Évalue les performances
5. Sauvegarde les modèles et résultats

Auteur: 0-Octet-1
Date: 8 juillet 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports pour ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des chemins
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_SPLIT_DIR = os.path.join(PROJECT_ROOT, 'data', 'split')
DATA_PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'preprocessed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

# Créer les dossiers si nécessaire
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_SPLIT_DIR, exist_ok=True)
os.makedirs(DATA_PREPROCESSED_DIR, exist_ok=True)

def print_section(title):
    """Affiche une section avec formatage"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def execute_pipeline_if_needed():
    """
    Vérifie si les données préprocessées existent, sinon exécute le pipeline
    """
    print_section("VÉRIFICATION DU PIPELINE DE DONNÉES")
    
    # Vérifier si les données préprocessées existent
    preprocessed_files = [
        'X_train_preprocessed.pkl',
        'X_val_preprocessed.pkl', 
        'X_test_preprocessed.pkl',
        'y_train.pkl',
        'y_val.pkl',
        'y_test.pkl'
    ]
    
    all_files_exist = all(
        os.path.exists(os.path.join(DATA_PREPROCESSED_DIR, file)) 
        for file in preprocessed_files
    )
    
    if all_files_exist:
        print("✅ Données préprocessées trouvées")
        return True
    
    print("❌ Données préprocessées manquantes")
    print("🔄 Exécution du pipeline de données...")
    
    # Exécuter les scripts du pipeline
    scripts_to_run = [
        'data_generation.py',
        'data_splitting.py', 
        'data_preprocessing.py'
    ]
    
    for script in scripts_to_run:
        script_path = os.path.join(NOTEBOOKS_DIR, script)
        if os.path.exists(script_path):
            print(f"🔄 Exécution de {script}...")
            try:
                exec(open(script_path).read())
                print(f"✅ {script} terminé")
            except Exception as e:
                print(f"❌ Erreur dans {script}: {e}")
                return False
        else:
            print(f"❌ Script {script} non trouvé")
            return False
    
    print("✅ Pipeline de données terminé")
    return True

def load_preprocessed_data():
    """
    Charge les données préprocessées
    """
    print_section("CHARGEMENT DES DONNÉES PRÉPROCESSÉES")
    
    try:
        # Charger les features
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'X_train_preprocessed.pkl'), 'rb') as f:
            X_train = pickle.load(f)
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'X_val_preprocessed.pkl'), 'rb') as f:
            X_val = pickle.load(f)
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'X_test_preprocessed.pkl'), 'rb') as f:
            X_test = pickle.load(f)
            
        # Charger les targets
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'y_val.pkl'), 'rb') as f:
            y_val = pickle.load(f)
        with open(os.path.join(DATA_PREPROCESSED_DIR, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
            
        print(f"✅ Données chargées:")
        print(f"   - Train: {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
        print(f"   - Validation: {X_val.shape[0]} échantillons")
        print(f"   - Test: {X_test.shape[0]} échantillons")
        
        # Distribution des classes
        print(f"\n📊 Distribution des classes (Train):")
        class_counts = pd.Series(y_train).value_counts().sort_index()
        for class_name, count in class_counts.items():
            percentage = (count / len(y_train)) * 100
            print(f"   - Classe {class_name}: {count} ({percentage:.1f}%)")
            
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        return None, None, None, None, None, None

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Entraîne un modèle Random Forest avec optimisation des hyperparamètres
    """
    print_section("ENTRAÎNEMENT RANDOM FOREST")
    
    # Définir la grille de paramètres
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Modèle de base
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid Search avec validation croisée
    print("🔍 Recherche des meilleurs hyperparamètres...")
    grid_search = GridSearchCV(
        rf_base, 
        param_grid, 
        cv=5, 
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Meilleur modèle
    best_rf = grid_search.best_estimator_
    
    print(f"✅ Meilleurs paramètres: {grid_search.best_params_}")
    print(f"✅ Meilleur score CV: {grid_search.best_score_:.4f}")
    
    # Évaluation sur validation
    y_val_pred = best_rf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"📊 Performance sur validation:")
    print(f"   - Accuracy: {val_accuracy:.4f}")
    print(f"   - F1-score macro: {val_f1:.4f}")
    
    return best_rf, grid_search.best_params_, val_f1

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Entraîne un modèle Logistic Regression avec optimisation des hyperparamètres
    """
    print_section("ENTRAÎNEMENT LOGISTIC REGRESSION")
    
    # Définir la grille de paramètres
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }
    
    # Modèle de base
    lr_base = LogisticRegression(random_state=42, n_jobs=-1)
    
    # Grid Search avec validation croisée
    print("🔍 Recherche des meilleurs hyperparamètres...")
    grid_search = GridSearchCV(
        lr_base, 
        param_grid, 
        cv=5, 
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Meilleur modèle
    best_lr = grid_search.best_estimator_
    
    print(f"✅ Meilleurs paramètres: {grid_search.best_params_}")
    print(f"✅ Meilleur score CV: {grid_search.best_score_:.4f}")
    
    # Évaluation sur validation
    y_val_pred = best_lr.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"📊 Performance sur validation:")
    print(f"   - Accuracy: {val_accuracy:.4f}")
    print(f"   - F1-score macro: {val_f1:.4f}")
    
    return best_lr, grid_search.best_params_, val_f1

def evaluate_model(model, X_test, y_test, model_name):
    """
    Évalue un modèle sur le jeu de test
    """
    print_section(f"ÉVALUATION {model_name.upper()}")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    
    print(f"📊 Métriques de performance:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - F1-score macro: {f1_macro:.4f}")
    print(f"   - F1-score weighted: {f1_weighted:.4f}")
    print(f"   - Precision macro: {precision_macro:.4f}")
    print(f"   - Recall macro: {recall_macro:.4f}")
    
    # Rapport de classification détaillé
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
    plt.savefig(os.path.join(MODELS_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # Importance des features (si disponible)
    if hasattr(model, 'feature_importances_'):
        print(f"\n🎯 Top 10 features importantes:")
        feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
        
        # Visualisation importance des features
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Importance des Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def save_models_and_results(models_dict, results_dict):
    """
    Sauvegarde les modèles et résultats
    """
    print_section("SAUVEGARDE DES MODÈLES ET RÉSULTATS")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder les modèles
    for model_name, model_data in models_dict.items():
        model_filename = f'{model_name.lower().replace(" ", "_")}_{timestamp}.pkl'
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Modèle {model_name} sauvegardé: {model_filename}")
    
    # Sauvegarder les résultats
    results_filename = f'ml_training_results_{timestamp}.json'
    results_path = os.path.join(MODELS_DIR, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"✅ Résultats sauvegardés: {results_filename}")
    
    # Créer un résumé des performances
    summary_filename = f'ml_performance_summary_{timestamp}.txt'
    summary_path = os.path.join(MODELS_DIR, summary_filename)
    
    with open(summary_path, 'w') as f:
        f.write("RÉSUMÉ DES PERFORMANCES - MODÈLES ML\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date d'entraînement: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model_name, results in results_dict.items():
            if 'test_results' in results:
                test_results = results['test_results']
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  - Accuracy: {test_results['accuracy']:.4f}\n")
                f.write(f"  - F1-score macro: {test_results['f1_macro']:.4f}\n")
                f.write(f"  - F1-score weighted: {test_results['f1_weighted']:.4f}\n")
                f.write(f"  - Precision macro: {test_results['precision_macro']:.4f}\n")
                f.write(f"  - Recall macro: {test_results['recall_macro']:.4f}\n\n")
    
    print(f"✅ Résumé des performances sauvegardé: {summary_filename}")

def main():
    """
    Fonction principale
    """
    print_section("SCRIPT D'ENTRAÎNEMENT ML - PROJET PMR")
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Vérifier et exécuter le pipeline si nécessaire
    if not execute_pipeline_if_needed():
        print("❌ Échec du pipeline de données")
        return
    
    # 2. Charger les données préprocessées
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    if X_train is None:
        print("❌ Impossible de charger les données")
        return
    
    # 3. Entraîner Random Forest
    rf_model, rf_params, rf_val_f1 = train_random_forest(X_train, y_train, X_val, y_val)
    
    # 4. Entraîner Logistic Regression
    lr_model, lr_params, lr_val_f1 = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # 5. Évaluer sur le jeu de test
    rf_test_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    lr_test_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # 6. Comparer les modèles
    print_section("COMPARAISON DES MODÈLES")
    print(f"Random Forest - F1 macro: {rf_test_results['f1_macro']:.4f}")
    print(f"Logistic Regression - F1 macro: {lr_test_results['f1_macro']:.4f}")
    
    if rf_test_results['f1_macro'] > lr_test_results['f1_macro']:
        print("🏆 Meilleur modèle: Random Forest")
        best_model_name = "Random Forest"
    else:
        print("🏆 Meilleur modèle: Logistic Regression")
        best_model_name = "Logistic Regression"
    
    # Vérifier l'objectif F1-score ≥ 75%
    target_f1 = 0.75
    rf_meets_target = rf_test_results['f1_macro'] >= target_f1
    lr_meets_target = lr_test_results['f1_macro'] >= target_f1
    
    print(f"\n🎯 Objectif F1-score ≥ {target_f1:.0%}:")
    print(f"   - Random Forest: {'✅' if rf_meets_target else '❌'} ({rf_test_results['f1_macro']:.1%})")
    print(f"   - Logistic Regression: {'✅' if lr_meets_target else '❌'} ({lr_test_results['f1_macro']:.1%})")
    
    # 7. Sauvegarder tout
    models_dict = {
        'Random Forest': {
            'model': rf_model,
            'best_params': rf_params,
            'validation_f1': rf_val_f1
        },
        'Logistic Regression': {
            'model': lr_model,
            'best_params': lr_params,
            'validation_f1': lr_val_f1
        }
    }
    
    results_dict = {
        'Random Forest': {
            'best_params': rf_params,
            'validation_f1': rf_val_f1,
            'test_results': rf_test_results
        },
        'Logistic Regression': {
            'best_params': lr_params,
            'validation_f1': lr_val_f1,
            'test_results': lr_test_results
        },
        'comparison': {
            'best_model': best_model_name,
            'target_f1_achieved': rf_meets_target or lr_meets_target
        }
    }
    
    save_models_and_results(models_dict, results_dict)
    
    print_section("ENTRAÎNEMENT TERMINÉ")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Tous les modèles ML ont été entraînés et sauvegardés")
    print(f"🏆 Meilleur modèle: {best_model_name}")
    print(f"📁 Résultats disponibles dans: {MODELS_DIR}")

if __name__ == "__main__":
    main()
