#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API FastAPI pour la prédiction d'accessibilité PMR
Projet PMR - Certification Développeur IA

API REST pour prédire l'accessibilité PMR de points d'intérêt urbains
Utilise le meilleur modèle Deep Learning (MLP Profond - F1: 94.4%)

Auteur: 0-Octet-1
Date: 8 juillet 2025
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Configuration des chemins
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Configuration de l'API
app = FastAPI(
    title="API PMR - Prédiction d'Accessibilité",
    description="API REST pour prédire l'accessibilité PMR de points d'intérêt urbains",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour les modèles
model_ml = None
model_dl = None
preprocessor = None
feature_names = None
model_info = {}

class PMRFeatures(BaseModel):
    """Modèle Pydantic pour les features d'entrée"""
    
    # Features principales
    largeur_trottoir: float = Field(..., ge=0.0, le=10.0, description="Largeur du trottoir en mètres")
    hauteur_bordure: float = Field(..., ge=0.0, le=0.5, description="Hauteur de la bordure en mètres")
    pente_acces: float = Field(..., ge=0.0, le=45.0, description="Pente d'accès en degrés")
    distance_transport: float = Field(..., ge=0.0, le=2000.0, description="Distance aux transports en mètres")
    eclairage_qualite: int = Field(..., ge=1, le=5, description="Qualité de l'éclairage (1-5)")
    surface_qualite: int = Field(..., ge=1, le=5, description="Qualité de la surface (1-5)")
    signalisation_presence: int = Field(..., ge=0, le=1, description="Présence de signalisation (0/1)")
    obstacles_nombre: int = Field(..., ge=0, le=20, description="Nombre d'obstacles")
    rampe_presence: int = Field(..., ge=0, le=1, description="Présence de rampe (0/1)")
    places_pmr_nombre: int = Field(..., ge=0, le=50, description="Nombre de places PMR")
    type_poi: str = Field(..., description="Type de point d'intérêt")
    
    @validator('type_poi')
    def validate_type_poi(cls, v):
        valid_types = ['restaurant', 'magasin', 'service_public', 'transport', 'loisir']
        if v.lower() not in valid_types:
            raise ValueError(f'Type POI doit être un de: {valid_types}')
        return v.lower()

class PredictionResponse(BaseModel):
    """Modèle de réponse pour les prédictions"""
    
    prediction: int = Field(..., description="Classe prédite (0: Non accessible, 1: Partiellement accessible, 2: Accessible)")
    prediction_label: str = Field(..., description="Label de la prédiction")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probabilités pour chaque classe")
    model_used: str = Field(..., description="Modèle utilisé pour la prédiction")
    timestamp: str = Field(..., description="Timestamp de la prédiction")

class BatchPredictionRequest(BaseModel):
    """Modèle pour les prédictions en lot"""
    
    features_list: List[PMRFeatures] = Field(..., description="Liste des features à prédire")
    model_type: Optional[str] = Field("dl", description="Type de modèle (ml/dl)")

def load_models():
    """Charge les modèles et le preprocesseur"""
    global model_ml, model_dl, preprocessor, feature_names, model_info
    
    try:
        # Charger le meilleur modèle DL (MLP Profond)
        dl_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('mlp_mlp_profond_') and f.endswith('.pkl')]
        if dl_files:
            latest_dl_file = sorted(dl_files)[-1]
            dl_path = os.path.join(MODELS_DIR, latest_dl_file)
            
            with open(dl_path, 'rb') as f:
                model_dl = pickle.load(f)
            
            print(f"✅ Modèle DL chargé: {latest_dl_file}")
        
        # Charger le meilleur modèle ML (Random Forest)
        ml_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('random_forest_') and f.endswith('.pkl')]
        if ml_files:
            latest_ml_file = sorted(ml_files)[-1]
            ml_path = os.path.join(MODELS_DIR, latest_ml_file)
            
            with open(ml_path, 'rb') as f:
                model_ml = pickle.load(f)
            
            print(f"✅ Modèle ML chargé: {latest_ml_file}")
        
        # Charger les informations des modèles
        results_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('mlp_results_') and f.endswith('.json')]
        if results_files:
            latest_results_file = sorted(results_files)[-1]
            results_path = os.path.join(MODELS_DIR, latest_results_file)
            
            with open(results_path, 'r') as f:
                model_info = json.load(f)
        
        # Définir les noms des features
        feature_names = [
            'largeur_trottoir', 'hauteur_bordure', 'pente_acces', 'distance_transport',
            'eclairage_qualite', 'surface_qualite', 'signalisation_presence',
            'obstacles_nombre', 'rampe_presence', 'places_pmr_nombre',
            'type_poi_loisir', 'type_poi_magasin', 'type_poi_restaurant',
            'type_poi_service_public', 'type_poi_transport'
        ]
        
        print("✅ Modèles et informations chargés avec succès")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles: {e}")
        raise

def preprocess_features(features: PMRFeatures) -> np.ndarray:
    """Préprocesse les features pour la prédiction"""
    
    # Créer le vecteur de features
    feature_vector = [
        features.largeur_trottoir,
        features.hauteur_bordure,
        features.pente_acces,
        features.distance_transport,
        features.eclairage_qualite,
        features.surface_qualite,
        features.signalisation_presence,
        features.obstacles_nombre,
        features.rampe_presence,
        features.places_pmr_nombre
    ]
    
    # One-hot encoding pour type_poi
    poi_types = ['loisir', 'magasin', 'restaurant', 'service_public', 'transport']
    for poi_type in poi_types:
        feature_vector.append(1 if features.type_poi == poi_type else 0)
    
    return np.array(feature_vector).reshape(1, -1)

def get_prediction_label(prediction: int) -> str:
    """Convertit la prédiction numérique en label"""
    labels = {
        0: "Non accessible",
        1: "Partiellement accessible", 
        2: "Accessible"
    }
    return labels.get(prediction, "Inconnu")

# Événements de démarrage
@app.on_event("startup")
async def startup_event():
    """Charge les modèles au démarrage de l'API"""
    print("🚀 Démarrage de l'API PMR...")
    load_models()
    print("✅ API PMR prête !")

# Routes de l'API

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil de l'API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API PMR - Prédiction d'Accessibilité</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .info { background: #e8f4fd; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
            .method { font-weight: bold; color: #007bff; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦽 API PMR - Prédiction d'Accessibilité</h1>
            
            <div class="info">
                <h3>📊 Informations sur les Modèles</h3>
                <p><strong>Meilleur Modèle:</strong> MLP Profond (Deep Learning)</p>
                <p><strong>Performance:</strong> F1-score macro: 94.4%</p>
                <p><strong>Accuracy:</strong> 97.5%</p>
            </div>
            
            <h3>🔗 Endpoints Disponibles</h3>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code> - Vérifier l'état de l'API
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/models/info</code> - Informations sur les modèles
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict</code> - Prédiction simple
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict/batch</code> - Prédictions en lot
            </div>
            
            <h3>📚 Documentation</h3>
            <p>
                <a href="/docs" target="_blank">📖 Documentation Swagger</a> | 
                <a href="/redoc" target="_blank">📋 Documentation ReDoc</a>
            </p>
            
            <div class="info">
                <h4>🎯 Classes de Prédiction</h4>
                <ul>
                    <li><strong>0:</strong> Non accessible</li>
                    <li><strong>1:</strong> Partiellement accessible</li>
                    <li><strong>2:</strong> Accessible</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "ml_model": model_ml is not None,
            "dl_model": model_dl is not None
        },
        "version": "1.0.0"
    }

@app.get("/models/info")
async def get_models_info():
    """Informations sur les modèles chargés"""
    return {
        "models_available": {
            "ml_model": "Random Forest" if model_ml else None,
            "dl_model": "MLP Profond" if model_dl else None
        },
        "performance": model_info.get("models", {}),
        "feature_names": feature_names,
        "classes": {
            0: "Non accessible",
            1: "Partiellement accessible",
            2: "Accessible"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: PMRFeatures, model_type: str = "dl"):
    """Prédiction simple pour un point d'intérêt"""
    
    try:
        # Sélectionner le modèle
        if model_type.lower() == "dl" and model_dl is not None:
            model = model_dl
            model_name = "MLP Profond (Deep Learning)"
        elif model_type.lower() == "ml" and model_ml is not None:
            model = model_ml
            model_name = "Random Forest (Machine Learning)"
        else:
            raise HTTPException(status_code=400, detail="Modèle non disponible")
        
        # Préprocesser les features
        X = preprocess_features(features)
        
        # Faire la prédiction
        prediction = model.predict(X)[0]
        
        # Obtenir les probabilités
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            
            prob_dict = {
                "Non accessible": float(probabilities[0]),
                "Partiellement accessible": float(probabilities[1]),
                "Accessible": float(probabilities[2])
            }
        else:
            confidence = 0.95  # Valeur par défaut
            prob_dict = {"prediction": float(prediction)}
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=get_prediction_label(int(prediction)),
            confidence=confidence,
            probabilities=prob_dict,
            model_used=model_name,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Prédictions en lot"""
    
    try:
        # Sélectionner le modèle
        model_type = request.model_type.lower()
        if model_type == "dl" and model_dl is not None:
            model = model_dl
            model_name = "MLP Profond (Deep Learning)"
        elif model_type == "ml" and model_ml is not None:
            model = model_ml
            model_name = "Random Forest (Machine Learning)"
        else:
            raise HTTPException(status_code=400, detail="Modèle non disponible")
        
        predictions = []
        
        for features in request.features_list:
            # Préprocesser les features
            X = preprocess_features(features)
            
            # Faire la prédiction
            prediction = model.predict(X)[0]
            
            # Obtenir les probabilités
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                confidence = float(np.max(probabilities))
                
                prob_dict = {
                    "Non accessible": float(probabilities[0]),
                    "Partiellement accessible": float(probabilities[1]),
                    "Accessible": float(probabilities[2])
                }
            else:
                confidence = 0.95
                prob_dict = {"prediction": float(prediction)}
            
            predictions.append({
                "prediction": int(prediction),
                "prediction_label": get_prediction_label(int(prediction)),
                "confidence": confidence,
                "probabilities": prob_dict
            })
        
        return {
            "predictions": predictions,
            "model_used": model_name,
            "total_predictions": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors des prédictions: {str(e)}")

@app.get("/predict/example")
async def get_prediction_example():
    """Exemple de données pour tester l'API"""
    return {
        "example_request": {
            "largeur_trottoir": 2.5,
            "hauteur_bordure": 0.15,
            "pente_acces": 5.0,
            "distance_transport": 150.0,
            "eclairage_qualite": 4,
            "surface_qualite": 4,
            "signalisation_presence": 1,
            "obstacles_nombre": 2,
            "rampe_presence": 1,
            "places_pmr_nombre": 3,
            "type_poi": "restaurant"
        },
        "curl_example": """
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "largeur_trottoir": 2.5,
       "hauteur_bordure": 0.15,
       "pente_acces": 5.0,
       "distance_transport": 150.0,
       "eclairage_qualite": 4,
       "surface_qualite": 4,
       "signalisation_presence": 1,
       "obstacles_nombre": 2,
       "rampe_presence": 1,
       "places_pmr_nombre": 3,
       "type_poi": "restaurant"
     }'
        """
    }

if __name__ == "__main__":
    print("🚀 Lancement de l'API PMR...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
