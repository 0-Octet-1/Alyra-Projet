#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour l'API PMR
Projet PMR - Certification Développeur IA

Teste les différents endpoints de l'API FastAPI

Auteur: 0-Octet-1
Date: 8 juillet 2025
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test de l'endpoint health"""
    print("🔍 Test de l'endpoint /health...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check OK: {data['status']}")
            print(f"   Modèles chargés: {data['models_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur health check: {e}")
        return False

def test_models_info():
    """Test de l'endpoint models/info"""
    print("\n🔍 Test de l'endpoint /models/info...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Informations modèles récupérées:")
            print(f"   Modèles disponibles: {data['models_available']}")
            if 'performance' in data:
                print(f"   Performance disponible: Oui")
            return True
        else:
            print(f"❌ Models info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur models info: {e}")
        return False

def test_single_prediction():
    """Test de prédiction simple"""
    print("\n🔍 Test de prédiction simple...")
    
    # Données de test
    test_data = {
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
    }
    
    try:
        # Test avec modèle DL
        response = requests.post(
            f"{API_BASE_URL}/predict?model_type=dl",
            json=test_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prédiction DL réussie:")
            print(f"   Prédiction: {data['prediction']} ({data['prediction_label']})")
            print(f"   Confiance: {data['confidence']:.3f}")
            print(f"   Modèle: {data['model_used']}")
            
            # Test avec modèle ML
            response_ml = requests.post(
                f"{API_BASE_URL}/predict?model_type=ml",
                json=test_data
            )
            
            if response_ml.status_code == 200:
                data_ml = response_ml.json()
                print("✅ Prédiction ML réussie:")
                print(f"   Prédiction: {data_ml['prediction']} ({data_ml['prediction_label']})")
                print(f"   Confiance: {data_ml['confidence']:.3f}")
                print(f"   Modèle: {data_ml['model_used']}")
            
            return True
        else:
            print(f"❌ Prédiction failed: {response.status_code}")
            print(f"   Erreur: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")
        return False

def test_batch_prediction():
    """Test de prédictions en lot"""
    print("\n🔍 Test de prédictions en lot...")
    
    # Données de test multiples
    test_data = {
        "features_list": [
            {
                "largeur_trottoir": 3.0,
                "hauteur_bordure": 0.10,
                "pente_acces": 3.0,
                "distance_transport": 100.0,
                "eclairage_qualite": 5,
                "surface_qualite": 5,
                "signalisation_presence": 1,
                "obstacles_nombre": 0,
                "rampe_presence": 1,
                "places_pmr_nombre": 5,
                "type_poi": "service_public"
            },
            {
                "largeur_trottoir": 1.5,
                "hauteur_bordure": 0.25,
                "pente_acces": 15.0,
                "distance_transport": 500.0,
                "eclairage_qualite": 2,
                "surface_qualite": 2,
                "signalisation_presence": 0,
                "obstacles_nombre": 8,
                "rampe_presence": 0,
                "places_pmr_nombre": 0,
                "type_poi": "magasin"
            }
        ],
        "model_type": "dl"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=test_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prédictions en lot réussies:")
            print(f"   Nombre de prédictions: {data['total_predictions']}")
            print(f"   Modèle: {data['model_used']}")
            
            for i, pred in enumerate(data['predictions']):
                print(f"   Prédiction {i+1}: {pred['prediction']} ({pred['prediction_label']}) - Confiance: {pred['confidence']:.3f}")
            
            return True
        else:
            print(f"❌ Prédictions en lot failed: {response.status_code}")
            print(f"   Erreur: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur prédictions en lot: {e}")
        return False

def test_example_endpoint():
    """Test de l'endpoint d'exemple"""
    print("\n🔍 Test de l'endpoint /predict/example...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/predict/example")
        if response.status_code == 200:
            data = response.json()
            print("✅ Exemple récupéré:")
            print(f"   Données d'exemple disponibles: Oui")
            print(f"   Commande curl disponible: Oui")
            return True
        else:
            print(f"❌ Example endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur example endpoint: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 TESTS DE L'API PMR")
    print("=" * 50)
    
    # Attendre que l'API soit prête
    print("⏳ Attente du démarrage de l'API...")
    time.sleep(3)
    
    tests = [
        ("Health Check", test_health),
        ("Models Info", test_models_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Example Endpoint", test_example_endpoint)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé des tests
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés ! L'API fonctionne parfaitement.")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les logs ci-dessus.")

if __name__ == "__main__":
    main()
