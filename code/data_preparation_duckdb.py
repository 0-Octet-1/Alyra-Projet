#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de préparation des données utilisant DuckDB pour le projet PMR
Ce script permet de charger, explorer et préparer les données d'accessibilité PMR
en utilisant DuckDB comme moteur de base de données analytique en mémoire.
"""

import os
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# Création des chemins de dossiers nécessaires
os.makedirs('data/processed', exist_ok=True)

# Connexion à DuckDB (en mémoire)
con = duckdb.connect(database=':memory:')

def generate_synthetic_data(n_samples=3000):
    """
    Génère un jeu de données synthétique pour le projet d'accessibilité PMR
    
    Args:
        n_samples: Nombre d'échantillons à générer
        
    Returns:
        DataFrame pandas contenant les données synthétiques
    """
    print("Génération des données synthétiques...")
    
    # Caractéristiques numériques
    age_infrastructure_an = np.random.normal(25, 12, n_samples).clip(0, 100)
    largeur_trottoir_cm = np.random.gamma(9, 15, n_samples).clip(30, 300)
    pente_acces_degres = np.random.exponential(2, n_samples).clip(0, 20)
    hauteur_marche_cm = np.random.gamma(2, 3, n_samples).clip(0, 30)
    places_pmr_nb = np.random.poisson(2, n_samples).clip(0, 10)
    distance_transport_m = np.random.gamma(10, 30, n_samples).clip(10, 1000)
    
    # Caractéristiques catégorielles
    type_lieu = np.random.choice(
        ['commerce', 'service_public', 'loisir', 'sante', 'education', 'restaurant'],
        n_samples,
        p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
    )
    
    rampe_acces = np.random.choice(
        ['aucune', 'temporaire', 'permanente', 'integree'], 
        n_samples,
        p=[0.3, 0.15, 0.4, 0.15]
    )
    
    ascenseur = np.random.choice(
        ['non', 'etroit', 'adapte', 'moderne'], 
        n_samples,
        p=[0.4, 0.2, 0.3, 0.1]
    )
    
    signalisation_pmr = np.random.choice(
        ['absente', 'minimale', 'claire', 'complete'], 
        n_samples,
        p=[0.3, 0.3, 0.3, 0.1]
    )
    
    zone_urbaine = np.random.choice(
        ['centre_historique', 'quartier_moderne', 'zone_commerciale', 'residentiel'], 
        n_samples,
        p=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Création du DataFrame
    df = pd.DataFrame({
        'age_infrastructure_an': age_infrastructure_an,
        'largeur_trottoir_cm': largeur_trottoir_cm,
        'pente_acces_degres': pente_acces_degres,
        'hauteur_marche_cm': hauteur_marche_cm,
        'places_pmr_nb': places_pmr_nb,
        'distance_transport_m': distance_transport_m,
        'type_lieu': type_lieu,
        'rampe_acces': rampe_acces,
        'ascenseur': ascenseur,
        'signalisation_pmr': signalisation_pmr,
        'zone_urbaine': zone_urbaine
    })
    
    # Génération de la variable cible (avec un léger déséquilibre en faveur des cas difficiles)
    # Calcul du score d'accessibilité basé sur les caractéristiques
    accessibilite_score = (
        df['largeur_trottoir_cm'] / 100 -
        df['pente_acces_degres'] * 0.5 -
        df['hauteur_marche_cm'] * 0.3 +
        df['places_pmr_nb'] * 1.5 -
        df['age_infrastructure_an'] * 0.05 -
        df['distance_transport_m'] * 0.01 +
        (df['rampe_acces'].map({
            'aucune': -3, 
            'temporaire': 0, 
            'permanente': 2, 
            'integree': 3
        })) +
        (df['ascenseur'].map({
            'non': -2, 
            'etroit': 0, 
            'adapte': 2, 
            'moderne': 3
        })) +
        (df['signalisation_pmr'].map({
            'absente': -2, 
            'minimale': 0, 
            'claire': 1, 
            'complete': 2
        }))
    )
    
    # Normalisation du score
    accessibilite_score = (accessibilite_score - accessibilite_score.min()) / (accessibilite_score.max() - accessibilite_score.min()) * 10
    
    # Création de la variable cible catégorielle
    conditions = [
        (accessibilite_score < 3),
        (accessibilite_score >= 3) & (accessibilite_score < 6),
        (accessibilite_score >= 6)
    ]
    choices = ['Difficilement_Accessible', 'Moderement_Accessible', 'Facilement_Accessible']
    df['accessibilite'] = np.select(conditions, choices, default='Moderement_Accessible')
    
    # Ajout de quelques valeurs manquantes pour simuler des données réelles
    for col in ['age_infrastructure_an', 'hauteur_marche_cm', 'places_pmr_nb']:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    print(f"Données synthétiques générées: {df.shape[0]} exemples avec {df.shape[1]} caractéristiques")
    return df

def load_data_to_duckdb(df):
    """
    Charge un DataFrame dans DuckDB
    
    Args:
        df: DataFrame pandas à charger
        
    Returns:
        Connexion DuckDB
    """
    print("Chargement des données dans DuckDB...")
    con.execute("CREATE TABLE IF NOT EXISTS accessibilite_pmr AS SELECT * FROM df")
    print("Données chargées avec succès dans la table 'accessibilite_pmr'")
    return con

def explore_data_with_duckdb(con):
    """
    Explore les données avec des requêtes DuckDB
    
    Args:
        con: Connexion DuckDB
    """
    print("\n--- EXPLORATION DES DONNÉES AVEC DUCKDB ---")
    
    # Afficher les premières lignes
    print("\n• Aperçu des données:")
    result = con.execute("SELECT * FROM accessibilite_pmr LIMIT 5").fetchdf()
    print(result)
    
    # Statistiques descriptives des variables numériques
    print("\n• Statistiques descriptives des variables numériques:")
    num_stats = con.execute("""
        SELECT 
            MIN(age_infrastructure_an) as min_age,
            MAX(age_infrastructure_an) as max_age,
            AVG(age_infrastructure_an) as avg_age,
            MIN(largeur_trottoir_cm) as min_largeur,
            MAX(largeur_trottoir_cm) as max_largeur,
            AVG(largeur_trottoir_cm) as avg_largeur,
            MIN(pente_acces_degres) as min_pente,
            MAX(pente_acces_degres) as max_pente,
            AVG(pente_acces_degres) as avg_pente
        FROM accessibilite_pmr
    """).fetchdf()
    print(num_stats)
    
    # Distribution de la variable cible
    print("\n• Distribution de la variable cible:")
    target_dist = con.execute("""
        SELECT 
            accessibilite, 
            COUNT(*) as nombre, 
            (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM accessibilite_pmr)) as pourcentage
        FROM accessibilite_pmr
        GROUP BY accessibilite
        ORDER BY nombre DESC
    """).fetchdf()
    print(target_dist)
    
    # Relations entre variables catégorielles et cible
    print("\n• Relation entre le type de lieu et l'accessibilité:")
    type_access = con.execute("""
        SELECT 
            type_lieu, 
            accessibilite, 
            COUNT(*) as nombre
        FROM accessibilite_pmr
        GROUP BY type_lieu, accessibilite
        ORDER BY type_lieu, accessibilite
    """).fetchdf()
    print(type_access)
    
    return {
        "apercu": result,
        "stats_num": num_stats,
        "dist_cible": target_dist,
        "type_access": type_access
    }

def preprocess_data(con):
    """
    Prétraitement des données avec DuckDB et Pipeline scikit-learn
    
    Args:
        con: Connexion DuckDB
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    print("\n--- PRÉTRAITEMENT DES DONNÉES ---")
    
    # Récupération des données depuis DuckDB
    df = con.execute("SELECT * FROM accessibilite_pmr").fetchdf()
    
    # Séparation en caractéristiques et variable cible
    X = df.drop('accessibilite', axis=1)
    y = df['accessibilite']
    
    # Identification des colonnes par type
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Caractéristiques numériques: {numeric_features}")
    print(f"Caractéristiques catégorielles: {categorical_features}")
    
    # Création du pipeline de prétraitement
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Division en ensembles d'entraînement et de test (stratified par classe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Taille de l'ensemble d'entraînement: {X_train.shape[0]} exemples")
    print(f"Taille de l'ensemble de test: {X_test.shape[0]} exemples")
    
    # Appliquer le prétraitement à l'ensemble d'entraînement seulement
    # (pour éviter la fuite de données)
    X_train_processed = pd.DataFrame(
        preprocessor.fit_transform(X_train),
    )
    
    # Conversion du pipeline et des données en format pickle pour plus tard
    import pickle
    os.makedirs('data/processed', exist_ok=True)
    
    # Sauvegarder les données brutes et prétraitées
    X_train.to_pickle('data/processed/X_train_raw.pkl')
    X_test.to_pickle('data/processed/X_test_raw.pkl')
    y_train.to_pickle('data/processed/y_train.pkl')
    y_test.to_pickle('data/processed/y_test.pkl')
    
    # Sauvegarder le pipeline de prétraitement
    with open('data/processed/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("Prétraitement terminé et données sauvegardées dans le dossier 'data/processed'")
    
    return X_train, X_test, y_train, y_test, preprocessor

def visualize_data(df):
    """
    Crée des visualisations pour explorer les données
    
    Args:
        df: DataFrame pandas contenant les données
    """
    print("\n--- VISUALISATION DES DONNÉES ---")
    
    # Configurer la figure pour un meilleur affichage
    plt.figure(figsize=(16, 14))
    
    # 1. Distribution de la variable cible
    plt.subplot(2, 2, 1)
    sns.countplot(x='accessibilite', data=df)
    plt.title('Distribution des classes d\'accessibilité')
    plt.xticks(rotation=45)
    
    # 2. Distribution de la largeur des trottoirs par classe d'accessibilité
    plt.subplot(2, 2, 2)
    sns.boxplot(x='accessibilite', y='largeur_trottoir_cm', data=df)
    plt.title('Largeur des trottoirs par classe d\'accessibilité')
    plt.xticks(rotation=45)
    
    # 3. Distribution de la pente d'accès par classe d'accessibilité
    plt.subplot(2, 2, 3)
    sns.boxplot(x='accessibilite', y='pente_acces_degres', data=df)
    plt.title('Pente d\'accès par classe d\'accessibilité')
    plt.xticks(rotation=45)
    
    # 4. Type de lieu par classe d'accessibilité
    plt.subplot(2, 2, 4)
    type_access = pd.crosstab(df['type_lieu'], df['accessibilite'])
    type_access.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Type de lieu par classe d\'accessibilité')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/processed/data_visualization.png')
    print("Visualisations sauvegardées dans 'data/processed/data_visualization.png'")
    
    return plt.gcf()  # Retourne la figure pour affichage interactif si nécessaire

def main():
    """Fonction principale exécutant le workflow de préparation des données"""
    print("=== PRÉPARATION DES DONNÉES AVEC DUCKDB ===")
    
    # Générer les données synthétiques
    df = generate_synthetic_data(n_samples=5000)
    
    # Charger les données dans DuckDB
    con = load_data_to_duckdb(df)
    
    # Explorer les données avec DuckDB
    explore_results = explore_data_with_duckdb(con)
    
    # Visualiser les données
    visualize_data(df)
    
    # Prétraiter les données
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(con)
    
    print("\n=== PRÉPARATION DES DONNÉES TERMINÉE ===")
    print("Les données sont prêtes pour la modélisation ML.")

if __name__ == "__main__":
    main()
