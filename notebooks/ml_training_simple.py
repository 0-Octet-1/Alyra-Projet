#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement ML simplifié - Projet PMR
Entraîne Random Forest et Logistic Regression sur données préprocessées

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

# Imports ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def load_data():
    """Charge les données préprocessées"""
    print_section("CHARGEMENT DES DONNÉES")
    
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
        
    print(f"✅ Données chargées:")
    print(f"   - Train: {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
    print(f"   - Validation: {X_val.shape[0]} échantillons")
    print(f"   - Test: {X_test.shape[0]} échantillons")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_random_forest(X_train, y_train, X_val, y_val):
    """Entraîne Random Forest avec paramètres optimisés"""
    print_section("ENTRAÎNEMENT RANDOM FOREST")
    
    # Paramètres optimisés
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print("🔄 Entraînement en cours...")
    rf.fit(X_train, y_train)
    
    # Évaluation sur validation
    y_val_pred = rf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"✅ Random Forest entraîné !")
    print(f"📊 Performance validation:")
    print(f"   - Accuracy: {val_accuracy:.4f}")
    print(f"   - F1-score macro: {val_f1:.4f}")
    
    return rf

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Entraîne Logistic Regression avec paramètres optimisés"""
    print_section("ENTRAÎNEMENT LOGISTIC REGRESSION")
    
    # Paramètres optimisés
    lr = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    print("🔄 Entraînement en cours...")
    lr.fit(X_train, y_train)
    
    # Évaluation sur validation
    y_val_pred = lr.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"✅ Logistic Regression entraîné !")
    print(f"📊 Performance validation:")
    print(f"   - Accuracy: {val_accuracy:.4f}")
    print(f"   - F1-score macro: {val_f1:.4f}")
    
    return lr

def evaluate_model(model, X_test, y_test, model_name):
    """Évalue un modèle sur le jeu de test"""
    print_section(f"ÉVALUATION {model_name.upper()}")
    
    # Prédictions
    y_pred = model.predict(X_test)
    
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
    
    # Rapport de classification
    print(f"\n📋 Rapport de classification:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Matrice de confusion:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm.tolist()
    }

def save_models(rf_model, lr_model, rf_results, lr_results):
    """Sauvegarde les modèles et résultats"""
    print_section("SAUVEGARDE")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder Random Forest
    rf_path = os.path.join(MODELS_DIR, f'random_forest_{timestamp}.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✅ Random Forest sauvegardé: {os.path.basename(rf_path)}")
    
    # Sauvegarder Logistic Regression
    lr_path = os.path.join(MODELS_DIR, f'logistic_regression_{timestamp}.pkl')
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"✅ Logistic Regression sauvegardé: {os.path.basename(lr_path)}")
    
    # Sauvegarder les résultats
    results = {
        'timestamp': timestamp,
        'random_forest': rf_results,
        'logistic_regression': lr_results
    }
    
    results_path = os.path.join(MODELS_DIR, f'ml_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Résultats sauvegardés: {os.path.basename(results_path)}")

def main():
    """Fonction principale"""
    print_section("ENTRAÎNEMENT ML - PROJET PMR")
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # 2. Entraîner Random Forest
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    
    # 3. Entraîner Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # 4. Évaluer sur le jeu de test
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # 5. Comparer les modèles
    print_section("COMPARAISON DES MODÈLES")
    print(f"Random Forest - F1 macro: {rf_results['f1_macro']:.4f}")
    print(f"Logistic Regression - F1 macro: {lr_results['f1_macro']:.4f}")
    
    if rf_results['f1_macro'] > lr_results['f1_macro']:
        print("🏆 Meilleur modèle: Random Forest")
    else:
        print("🏆 Meilleur modèle: Logistic Regression")
    
    # Vérifier l'objectif F1-score ≥ 75%
    target_f1 = 0.75
    rf_meets_target = rf_results['f1_macro'] >= target_f1
    lr_meets_target = lr_results['f1_macro'] >= target_f1
    
    print(f"\n🎯 Objectif F1-score ≥ {target_f1:.0%}:")
    print(f"   - Random Forest: {'✅' if rf_meets_target else '❌'} ({rf_results['f1_macro']:.1%})")
    print(f"   - Logistic Regression: {'✅' if lr_meets_target else '❌'} ({lr_results['f1_macro']:.1%})")
    
    # 6. Sauvegarder
    save_models(rf_model, lr_model, rf_results, lr_results)
    
    print_section("ENTRAÎNEMENT TERMINÉ")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Modèles ML entraînés et sauvegardés avec succès !")

if __name__ == "__main__":
    main()
