#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script complet d'entraînement des modèles Deep Learning
Projet PMR - Certification Développeur IA

Ce script :
1. Charge les données préprocessées
2. Entraîne plusieurs architectures ANN pour données tabulaires
3. Optimise les hyperparamètres
4. Évalue les performances
5. Compare avec les modèles ML existants
6. Sauvegarde les meilleurs modèles

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

# Imports Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Imports pour évaluation
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration GPU (si disponible)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("🚀 GPU détecté et configuré")
else:
    print("💻 Utilisation du CPU")

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
    
    # Encoder les labels pour le Deep Learning (one-hot encoding)
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"✅ Données chargées et préparées:")
    print(f"   - Train: {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
    print(f"   - Validation: {X_val.shape[0]} échantillons")
    print(f"   - Test: {X_test.shape[0]} échantillons")
    print(f"   - Nombre de classes: {num_classes}")
    print(f"   - Shape y_train: {y_train_cat.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_cat, y_val_cat, y_test_cat, num_classes

def create_simple_ann(input_dim, num_classes):
    """Crée un ANN simple"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_deep_ann(input_dim, num_classes):
    """Crée un ANN plus profond"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_wide_ann(input_dim, num_classes):
    """Crée un ANN large"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train_cat, X_val, y_val_cat, model_name, epochs=100):
    """Entraîne un modèle avec callbacks"""
    print_section(f"ENTRAÎNEMENT {model_name.upper()}")
    
    # Compiler le modèle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"📋 Architecture du modèle {model_name}:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, f'{model_name.lower().replace(" ", "_")}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"🔄 Entraînement en cours...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"✅ {model_name} entraîné avec succès !")
    
    return model, history

def evaluate_dl_model(model, X_test, y_test, y_test_cat, model_name):
    """Évalue un modèle Deep Learning"""
    print_section(f"ÉVALUATION {model_name.upper()}")
    
    # Prédictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
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
    
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'confusion_matrix_dl_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm.tolist()
    }

def plot_training_history(history, model_name):
    """Visualise l'historique d'entraînement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'training_history_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def compare_with_ml_models(dl_results):
    """Compare les résultats DL avec les modèles ML existants"""
    print_section("COMPARAISON ML vs DL")
    
    # Charger les résultats ML existants
    ml_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('ml_results_') and f.endswith('.json')]
    
    if ml_files:
        latest_ml_file = sorted(ml_files)[-1]
        ml_path = os.path.join(MODELS_DIR, latest_ml_file)
        
        with open(ml_path, 'r') as f:
            ml_results = json.load(f)
        
        print("📊 COMPARAISON DES PERFORMANCES:")
        print(f"{'Modèle':<20} {'Accuracy':<10} {'F1-macro':<10} {'F1-weighted':<12}")
        print("-" * 52)
        
        # Modèles ML
        if 'random_forest' in ml_results:
            rf = ml_results['random_forest']
            print(f"{'Random Forest':<20} {rf['accuracy']:<10.4f} {rf['f1_macro']:<10.4f} {rf['f1_weighted']:<12.4f}")
        
        if 'logistic_regression' in ml_results:
            lr = ml_results['logistic_regression']
            print(f"{'Logistic Regression':<20} {lr['accuracy']:<10.4f} {lr['f1_macro']:<10.4f} {lr['f1_weighted']:<12.4f}")
        
        # Modèles DL
        for model_name, results in dl_results.items():
            print(f"{model_name:<20} {results['accuracy']:<10.4f} {results['f1_macro']:<10.4f} {results['f1_weighted']:<12.4f}")
        
        # Identifier le meilleur modèle global
        all_models = {}
        if 'random_forest' in ml_results:
            all_models['Random Forest (ML)'] = ml_results['random_forest']['f1_macro']
        if 'logistic_regression' in ml_results:
            all_models['Logistic Regression (ML)'] = ml_results['logistic_regression']['f1_macro']
        
        for model_name, results in dl_results.items():
            all_models[f"{model_name} (DL)"] = results['f1_macro']
        
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

def save_dl_results(models_dict, results_dict, histories_dict):
    """Sauvegarde les modèles et résultats DL"""
    print_section("SAUVEGARDE DES MODÈLES DEEP LEARNING")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder les modèles
    for model_name, model in models_dict.items():
        model_filename = f'dl_{model_name.lower().replace(" ", "_")}_{timestamp}.h5'
        model_path = os.path.join(MODELS_DIR, model_filename)
        model.save(model_path)
        print(f"✅ {model_name} sauvegardé: {model_filename}")
    
    # Sauvegarder les résultats
    results_with_timestamp = {
        'timestamp': timestamp,
        'models': results_dict,
        'training_info': {
            'framework': 'TensorFlow/Keras',
            'epochs_max': 100,
            'batch_size': 32,
            'optimizer': 'Adam',
            'learning_rate': 0.001
        }
    }
    
    results_filename = f'dl_results_{timestamp}.json'
    results_path = os.path.join(MODELS_DIR, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results_with_timestamp, f, indent=2, default=str)
    
    print(f"✅ Résultats DL sauvegardés: {results_filename}")
    
    # Sauvegarder les historiques d'entraînement
    for model_name, history in histories_dict.items():
        plot_training_history(history, model_name)
    
    print("✅ Graphiques d'entraînement sauvegardés")

def main():
    """Fonction principale"""
    print_section("ENTRAÎNEMENT DEEP LEARNING - PROJET PMR")
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # 1. Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_cat, y_val_cat, y_test_cat, num_classes = load_data()
    input_dim = X_train.shape[1]
    
    # 2. Créer et entraîner les modèles
    models_dict = {}
    results_dict = {}
    histories_dict = {}
    
    # Modèle ANN Simple
    print_section("CRÉATION DES MODÈLES")
    simple_ann = create_simple_ann(input_dim, num_classes)
    simple_ann, simple_history = train_model(simple_ann, X_train, y_train_cat, X_val, y_val_cat, "Simple ANN", epochs=100)
    simple_results = evaluate_dl_model(simple_ann, X_test, y_test, y_test_cat, "Simple ANN")
    
    models_dict["Simple ANN"] = simple_ann
    results_dict["Simple ANN"] = simple_results
    histories_dict["Simple ANN"] = simple_history
    
    # Modèle ANN Profond
    deep_ann = create_deep_ann(input_dim, num_classes)
    deep_ann, deep_history = train_model(deep_ann, X_train, y_train_cat, X_val, y_val_cat, "Deep ANN", epochs=100)
    deep_results = evaluate_dl_model(deep_ann, X_test, y_test, y_test_cat, "Deep ANN")
    
    models_dict["Deep ANN"] = deep_ann
    results_dict["Deep ANN"] = deep_results
    histories_dict["Deep ANN"] = deep_history
    
    # Modèle ANN Large
    wide_ann = create_wide_ann(input_dim, num_classes)
    wide_ann, wide_history = train_model(wide_ann, X_train, y_train_cat, X_val, y_val_cat, "Wide ANN", epochs=100)
    wide_results = evaluate_dl_model(wide_ann, X_test, y_test, y_test_cat, "Wide ANN")
    
    models_dict["Wide ANN"] = wide_ann
    results_dict["Wide ANN"] = wide_results
    histories_dict["Wide ANN"] = wide_history
    
    # 3. Comparer les modèles DL entre eux
    print_section("COMPARAISON DES MODÈLES DEEP LEARNING")
    best_dl_model = max(results_dict, key=lambda x: results_dict[x]['f1_macro'])
    best_dl_f1 = results_dict[best_dl_model]['f1_macro']
    
    print(f"🏆 Meilleur modèle DL: {best_dl_model}")
    print(f"📊 F1-score macro: {best_dl_f1:.4f}")
    
    # 4. Comparer avec les modèles ML
    best_global_model, best_global_f1 = compare_with_ml_models(results_dict)
    
    # 5. Sauvegarder tout
    save_dl_results(models_dict, results_dict, histories_dict)
    
    print_section("ENTRAÎNEMENT DEEP LEARNING TERMINÉ")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Tous les modèles Deep Learning ont été entraînés et sauvegardés")
    print(f"🏆 Meilleur modèle DL: {best_dl_model} (F1: {best_dl_f1:.4f})")
    if best_global_model:
        print(f"🌟 Meilleur modèle global: {best_global_model} (F1: {best_global_f1:.4f})")

if __name__ == "__main__":
    main()
