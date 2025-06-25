#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interface Streamlit pour l'exploration des données d'accessibilité PMR
Ce script permet d'explorer et visualiser interactivement les données
générées par le script data_preparation_duckdb.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import os
import sys
import pickle
from sklearn.pipeline import Pipeline

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Exploration des données d'accessibilité PMR",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger les données (soit les générer, soit les charger depuis les fichiers)
@st.cache_data
def load_data(regenerate=False):
    """
    Charge ou génère les données d'accessibilité PMR
    
    Args:
        regenerate: Si True, régénère les données même si les fichiers existent
        
    Returns:
        DataFrame pandas contenant les données
    """
    # Import déclaré en début de fonction pour éviter les erreurs
    import os
    import sys
    import pickle
    
    # On ajoute le dossier parent (racine du projet) au path Python
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Chemin des données prétraitées
    processed_data_path = os.path.join('..', 'data', 'processed')
    
    # Si les données n'existent pas ou si on demande une régénération
    if regenerate or not os.path.exists(os.path.join(processed_data_path, 'X_train_raw.pkl')):
        st.info("Génération de nouvelles données... (cela peut prendre quelques instants)")
        
        # Import de la fonction de génération depuis notre module
        import sys
        import os
        # On ajoute le dossier parent (racine du projet) au path Python
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Import direct depuis le module sans utiliser le préfixe 'code'
        from data_preparation_duckdb import generate_synthetic_data, load_data_to_duckdb
        
        # Génération des données
        df = generate_synthetic_data(n_samples=5000)
        
        # Création du dossier si nécessaire
        os.makedirs(processed_data_path, exist_ok=True)
        
        # Sauvegarde des données brutes pour référence future
        df.to_pickle(os.path.join(processed_data_path, 'raw_data.pkl'))
        
        return df
    else:
        # Chargement des données brutes depuis le fichier pickle
        try:
            df = pd.read_pickle(os.path.join(processed_data_path, 'raw_data.pkl'))
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            
            # En cas d'erreur, on régénère les données
            st.warning("Régénération des données...")
            import sys
            import os
            # On ajoute le dossier parent (racine du projet) au path Python
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # Import direct depuis le module sans utiliser le préfixe 'code'
            from data_preparation_duckdb import generate_synthetic_data
            df = generate_synthetic_data(n_samples=5000)
            return df

# Fonction pour créer et retourner une connexion DuckDB
@st.cache_resource
def get_duckdb_connection(df):
    """
    Crée une connexion DuckDB et charge les données
    
    Args:
        df: DataFrame pandas à charger
        
    Returns:
        Connexion DuckDB
    """
    con = duckdb.connect(database=':memory:')
    con.execute("CREATE TABLE IF NOT EXISTS accessibilite_pmr AS SELECT * FROM df")
    return con

# Fonction pour afficher une description complète de la base de données
def show_database_description(df):
    """
    Affiche une description complète de la base de données avec:
    - Structure générale
    - Dictionnaire des variables
    - Types et informations manquantes
    - Distribution des classes cibles
    """
    st.header("📊 Description complète de la base de données")
    
    # Information générale sur le dataset
    st.subheader("Structure générale du dataset")
    st.markdown(f"""
    - **Nombre total d'observations**: {df.shape[0]}
    - **Nombre de variables**: {df.shape[1]}
    - **Variables numériques**: {len(df.select_dtypes(include=['float64', 'int64']).columns)}
    - **Variables catégorielles**: {len(df.select_dtypes(include=['object', 'category']).columns)}
    - **Mémoire utilisée**: {df.memory_usage().sum() / 1024**2:.2f} MB
    """)
    
    # Afficher les colonnes disponibles pour diagnostic
    st.subheader("Colonnes disponibles dans le dataset")
    st.write(df.columns.tolist())
    
    # Vérifier si la colonne de classe existe sous un autre nom
    target_column = None
    possible_target_names = ['classe_accessibilite', 'classe', 'accessibilite', 'target', 'label', 'y']
    for column in possible_target_names:
        if column in df.columns:
            target_column = column
            break
    
    # Si on trouve une colonne cible, on affiche sa distribution
    if target_column:
        st.subheader(f"Distribution des classes d'accessibilité PMR (colonne: {target_column})")
        fig, ax = plt.subplots(figsize=(10, 6))
        class_counts = df[target_column].value_counts()
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        class_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_xlabel("Classe d'accessibilité")
        ax.set_ylabel("Nombre d'observations")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
        # Distribution en pourcentage si on a les classes attendues
        if len(class_counts) <= 3:
            st.markdown("**Distribution en pourcentage**:")
            for class_name, count in class_counts.items():
                st.markdown(f"- {class_name}: {count / len(df) * 100:.1f}%")
    else:
        st.warning("Colonne de classe d'accessibilité non trouvée dans le dataset. "
                  "Veuillez vérifier le nom de la colonne ou la génération des données.")
        
        # Affiche un aperçu des données pour un diagnostic plus approfondi
        st.subheader("Aperçu des 10 premières lignes (pour diagnostic)")
        st.dataframe(df.head(10))
    
    
    # Dictionnaire des variables
    st.subheader("Dictionnaire des variables")
    variable_dict = {
        "nom_poi": "Nom du point d'intérêt urbain",
        "type_poi": "Type du point d'intérêt (commerce, service public, loisir...)",
        "largeur_trottoir_cm": "Largeur du trottoir d'accès en centimètres",
        "presence_rampe": "Présence d'une rampe d'accès (Oui/Non)",
        "hauteur_marche_cm": "Hauteur de la marche d'entrée en centimètres",
        "nb_marches": "Nombre de marches à l'entrée",
        "presence_ascenseur": "Présence d'un ascenseur (Oui/Non)",
        "largeur_porte_cm": "Largeur de la porte d'entrée en centimètres",
        "type_porte": "Type de porte (manuelle, automatique, battante...)",
        "pente_acces_degres": "Pente d'accès en degrés",
        "distance_place_pmr_m": "Distance de la place de stationnement PMR la plus proche en mètres",
        "presence_toilettes_pmr": "Présence de toilettes adaptées aux PMR (Oui/Non)",
        "largeur_couloir_cm": "Largeur du couloir principal en centimètres",
        "presence_boucle_magnetique": "Présence d'une boucle magnétique pour malentendants (Oui/Non)",
        "presence_signaletique_adaptee": "Présence d'une signalétique adaptée (Oui/Non)",
        "contraste_visuel_suffisant": "Contraste visuel suffisant pour malvoyants (Oui/Non)",
        "zone_manoeuvre_fauteuil": "Zone de manœuvre suffisante pour fauteuil roulant (Oui/Non)",
        "personnels_formes": "Personnel formé à l'accueil des PMR (Oui/Non)",
        "age_infrastructure_an": "Âge de l'infrastructure en années",
        "classe_accessibilite": "Classe d'accessibilité PMR (Facilement/Modérément/Difficilement accessible)"
    }
    
    # Créer un DataFrame pour afficher le dictionnaire
    dict_df = pd.DataFrame(list(variable_dict.items()), columns=["Variable", "Description"])
    st.dataframe(dict_df, hide_index=True, width=800)
    
    # Analyse des valeurs manquantes
    st.subheader("Analyse des valeurs manquantes")
    missing_df = pd.DataFrame({
        'Variable': df.columns,
        'Type': df.dtypes.astype(str),
        'Valeurs manquantes': df.isna().sum(),
        'Pourcentage (%)': (df.isna().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_df, hide_index=True, width=800)

# Fonction pour afficher les statistiques descriptives
def show_statistics(df, con):
    st.header("📊 Statistiques descriptives")
    
    col1, col2 = st.columns(2)
    
    # Afficher les statistiques des variables numériques
    with col1:
        st.subheader("Variables numériques")
        st.write(df.describe())
    
    # Afficher la distribution de la variable cible
    with col2:
        st.subheader("Distribution des classes d'accessibilité")
        target_dist = con.execute("""
            SELECT 
                accessibilite, 
                COUNT(*) as nombre, 
                (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM accessibilite_pmr)) as pourcentage
            FROM accessibilite_pmr
            GROUP BY accessibilite
            ORDER BY nombre DESC
        """).fetchdf()
        
        st.dataframe(target_dist)
        
        # Créer un graphique pie pour la distribution
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            target_dist['pourcentage'], 
            labels=target_dist['accessibilite'], 
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.05 if x == 'Difficilement_Accessible' else 0 for x in target_dist['accessibilite']]
        )
        ax.axis('equal')  # Aspect ratio égal pour un cercle
        st.pyplot(fig)

# Fonction pour afficher les corrélations
def show_correlations(df):
    st.header("🔄 Analyse des corrélations")
    
    # Récupérer le nom de la colonne cible s'il existe
    target_column = None
    possible_target_names = ['classe_accessibilite', 'classe', 'accessibilite', 'target', 'label', 'y']
    for column in possible_target_names:
        if column in df.columns:
            target_column = column
            break
    
    # Sélectionner uniquement les colonnes numériques
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculer la matrice de corrélation
    corr_matrix = df[num_columns].corr()
    
    # Options pour la visualisation des corrélations
    st.subheader("Options de visualisation")
    col1, col2 = st.columns(2)
    
    with col1:
        corr_threshold = st.slider(
            "Seuil minimum de corrélation",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Masquer les corrélations inférieures à cette valeur absolue"
        )
    
    with col2:
        mask_option = st.radio(
            "Masquage de la matrice",
            options=["Afficher tout", "Masquer la diagonale", "Masquer le triangle inférieur"],
            index=2,
            horizontal=True
        )
    
    # Créer un masque selon l'option choisie
    mask = None
    if mask_option == "Masquer la diagonale":
        mask = np.eye(len(corr_matrix))
    elif mask_option == "Masquer le triangle inférieur":
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Appliquer un seuil aux corrélations faibles
    if corr_threshold > 0:
        # Créer une copie pour l'affichage
        display_matrix = corr_matrix.copy()
        # Remplacer les corrélations faibles par NaN pour qu'elles soient transparentes dans le heatmap
        display_matrix[abs(display_matrix) < corr_threshold] = np.nan
    else:
        display_matrix = corr_matrix
    
    # Afficher la matrice de corrélation sous forme de heatmap amélioré
    st.subheader("Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Utiliser un meilleur colormap et plus d'options
    sns.heatmap(
        display_matrix,
        mask=mask,
        annot=True,  # Afficher les valeurs
        cmap='coolwarm',  # Colormap rouge-bleu
        fmt=".2f",  # Format à deux décimales
        ax=ax,
        center=0,  # Centrer la colormap à zéro
        square=True,  # Cellules carrées
        linewidths=0.5,  # Lignes entre les cellules
        cbar_kws={"shrink": 0.8, "label": "Coefficient de corrélation"}
    )
    
    # Rotation des étiquettes pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Analyse textuelle des corrélations les plus fortes
    st.subheader("Corrélations importantes")
    
    # Convertir la matrice en un format tabulaire pour analyse
    corr_df = corr_matrix.unstack().reset_index()
    corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
    
    # Filtrer pour enlever les auto-corrélations et les duplications
    corr_df = corr_df[
        (corr_df['Feature 1'] != corr_df['Feature 2']) &  # Enlever auto-corrélations
        (abs(corr_df['Correlation']) >= corr_threshold)  # Appliquer le seuil
    ]
    
    # Trier par corrélation absolue
    corr_df['Abs_Correlation'] = abs(corr_df['Correlation'])
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    corr_df = corr_df.drop('Abs_Correlation', axis=1)
    
    # Enlever les duplications (la corrélation entre A et B est la même qu'entre B et A)
    corr_df['Pair'] = corr_df.apply(lambda x: '_'.join(sorted([x['Feature 1'], x['Feature 2']])), axis=1)
    corr_df = corr_df.drop_duplicates('Pair').drop('Pair', axis=1)
    
    # Afficher les corrélations les plus fortes
    st.dataframe(corr_df.head(15), hide_index=True, width=800)
    
    # Afficher l'interprétation des corrélations
    st.subheader("Interprétation des corrélations")
    st.write("""
    - **Corrélation positive forte** (proche de 1) : Lorsqu'une variable augmente, l'autre augmente également de façon proportionnelle
    - **Corrélation négative forte** (proche de -1) : Lorsqu'une variable augmente, l'autre diminue de façon proportionnelle
    - **Corrélation proche de 0** : Pas de relation linéaire entre les variables
    """)
    
    # Si on a une colonne cible, montrer les corrélations avec cette colonne spécifiquement
    if target_column and target_column in df.columns and df[target_column].dtype.name in ['category', 'object']:
        st.subheader(f"Relation entre les caractéristiques numériques et la classe d'accessibilité")
        
        # One-hot encode la colonne cible pour calculer les corrélations
        target_dummies = pd.get_dummies(df[target_column], prefix='class')
        
        # Fusionner avec les variables numériques
        extended_df = pd.concat([df[num_columns], target_dummies], axis=1)
        
        # Calculer les nouvelles corrélations
        target_corr = extended_df.corr().loc[num_columns, target_dummies.columns]
        
        # Afficher les corrélations triées en barplot horizontal
        fig, axes = plt.subplots(1, len(target_dummies.columns), figsize=(15, 8))
        
        # Si une seule classe, axes n'est pas un itérable
        if len(target_dummies.columns) == 1:
            axes = [axes]
            
        for i, class_col in enumerate(target_dummies.columns):
            # Trier les corrélations
            class_corrs = target_corr[class_col].sort_values(ascending=False)
            
            # Plot horizontal bar chart
            class_corrs.plot(kind='barh', ax=axes[i], color='skyblue')
            axes[i].set_title(f"Corrélations avec {class_col}")
            axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for j, v in enumerate(class_corrs):
                axes[i].text(v + (0.01 if v >= 0 else -0.05), j, f"{v:.2f}", va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("""
        Ce graphique montre les caractéristiques les plus prédictives pour chaque classe d'accessibilité.
        Les valeurs positives indiquent qu'une valeur élevée de la caractéristique est associée à cette classe,
        tandis que les valeurs négatives indiquent une relation inverse.
        """)
        

# Fonction pour afficher les visualisations
def show_visualizations(df):
    st.header("📈 Visualisations")
    
    # Sélection du type de visualisation
    viz_type = st.selectbox(
        "Choisir un type de visualisation",
        ["Distribution des variables numériques", "Relations avec la classe d'accessibilité", "Croisements catégoriels"]
    )
    
    if viz_type == "Distribution des variables numériques":
        # Sélectionner la variable à visualiser
        num_columns = df.select_dtypes(include=['int64', 'float64']).columns
        selected_var = st.selectbox("Sélectionner une variable", num_columns)
        
        # Création de la figure
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogramme
        sns.histplot(df[selected_var], kde=True, ax=ax[0])
        ax[0].set_title(f'Distribution de {selected_var}')
        
        # Boxplot par classe d'accessibilité
        sns.boxplot(x='accessibilite', y=selected_var, data=df, ax=ax[1])
        ax[1].set_title(f'{selected_var} par classe d\'accessibilité')
        ax[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    elif viz_type == "Relations avec la classe d'accessibilité":
        # Créer des paires de visualisations pour les variables numériques par rapport à la classe
        num_columns = df.select_dtypes(include=['int64', 'float64']).columns
        selected_vars = st.multiselect(
            "Sélectionner les variables à comparer (max 3 recommandé)",
            num_columns,
            default=list(num_columns[:2])
        )
        
        if selected_vars:
            fig, ax = plt.subplots(1, len(selected_vars), figsize=(5*len(selected_vars), 5))
            
            # Si une seule variable est sélectionnée, ax n'est pas un tableau
            if len(selected_vars) == 1:
                ax = [ax]
            
            for i, var in enumerate(selected_vars):
                sns.boxplot(x='accessibilite', y=var, data=df, ax=ax[i])
                ax[i].set_title(f'{var} par classe')
                ax[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
        
    else:  # Croisements catégoriels
        # Sélectionner deux variables catégorielles à croiser
        cat_columns = df.select_dtypes(include=['object']).columns.tolist()
        cat_columns.remove('accessibilite')  # On retire la variable cible
        
        col1, col2 = st.columns(2)
        with col1:
            cat_var1 = st.selectbox("Première variable", cat_columns)
        with col2:
            cat_var2 = st.selectbox("Deuxième variable", cat_columns, index=1 if len(cat_columns) > 1 else 0)
        
        # Tableau croisé
        cross_tab = pd.crosstab(df[cat_var1], df[cat_var2])
        st.write(cross_tab)
        
        # Visualisation sous forme de heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='viridis', ax=ax)
        plt.title(f'Tableau croisé: {cat_var1} vs {cat_var2}')
        st.pyplot(fig)

# Fonction pour afficher la requête DuckDB personnalisée
def show_custom_query(con):
    st.header("🔍 Requête DuckDB personnalisée")
    
    # Zone de texte pour entrer une requête SQL personnalisée
    default_query = "SELECT type_lieu, accessibilite, COUNT(*) as nombre FROM accessibilite_pmr GROUP BY type_lieu, accessibilite ORDER BY type_lieu, accessibilite"
    query = st.text_area("Entrer une requête SQL personnalisée", default_query, height=100)
    
    if st.button("Exécuter la requête"):
        try:
            # Exécuter la requête et récupérer les résultats
            result = con.execute(query).fetchdf()
            
            # Afficher les résultats sous forme de tableau
            st.dataframe(result)
            
            # Si la requête renvoie des données numériques, proposer une visualisation
            if len(result.columns) >= 2 and result.shape[0] > 1:
                if st.checkbox("Visualiser les résultats"):
                    # Tenter de déterminer le meilleur type de graphique
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if result.shape[0] <= 20 and result.select_dtypes(include=['int64', 'float64']).shape[1] >= 1:
                        # Trouver une colonne numérique pour le graphique
                        num_cols = result.select_dtypes(include=['int64', 'float64']).columns
                        if len(num_cols) > 0:
                            # Utiliser un barplot pour visualiser
                            result.plot(kind='bar', x=result.columns[0], y=num_cols[0], ax=ax)
                            plt.title(f'Visualisation de {num_cols[0]} par {result.columns[0]}')
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Erreur d'exécution de la requête: {e}")

# Fonction principale pour l'application Streamlit
def main():
    st.title("♿ Exploration des données d'accessibilité PMR")
    st.markdown("""
    Cette application permet d'explorer et visualiser les données du projet d'accessibilité PMR,
    en utilisant DuckDB comme moteur de requêtes pour une analyse rapide et efficace.
    """)
    
    # Barre latérale avec options
    st.sidebar.title("Options")
    regenerate_data = st.sidebar.checkbox("Régénérer les données")
    
    # Charger les données
    df = load_data(regenerate=regenerate_data)
    
    # Créer une connexion DuckDB
    con = get_duckdb_connection(df)
    
    # Afficher un aperçu des données
    st.header("📄 Aperçu des données")
    st.dataframe(df.head())
    
    st.markdown(f"**{df.shape[0]} exemples** avec **{df.shape[1]} caractéristiques**")
    
    # Navigation par onglets avec Description comme premier onglet
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Description", "Statistiques", "Corrélations", "Visualisations", "Requêtes SQL"])
    
    with tab0:
        show_database_description(df)
    
    with tab1:
        show_statistics(df, con)
    
    with tab2:
        show_correlations(df)
    
    with tab3:
        show_visualizations(df)
    
    with tab4:
        show_custom_query(con)

if __name__ == "__main__":
    main()
