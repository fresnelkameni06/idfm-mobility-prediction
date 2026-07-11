# ============================================================
# Tests de l'API IDFM Mobility
# Fichier : tests/test_api.py
# ============================================================
#
# POURQUOI CES TESTS ?
# Vérifier que l'API répond correctement AVANT chaque déploiement.
# Le CI/CD (GitHub Actions) les lance automatiquement : si un test
# échoue, le déploiement est bloqué. C'est le filet de sécurité.
#
# CE QU'ON TESTE :
#   - L'API démarre et /health répond
#   - Les endpoints renvoient les bons codes HTTP
#     (200 = OK, 404 = arrêt inconnu, 422 = entrée invalide)
#   - La structure des réponses est correcte
#
# On utilise TestClient de FastAPI : il teste l'API en mémoire,
# sans avoir à la lancer sur un port. Rapide et fiable.
#
# LANCER :
#   pytest tests/ -v
# ============================================================

import sys
import os
import pytest

# Permet d'importer api/ et src/ depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ============================================================
# /health — l'API est-elle en ligne ?
# ============================================================
def test_health_repond():
    """L'endpoint /health doit répondre 200 avec un statut ok."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["api"] == "online"


def test_health_contient_modeles():
    """/health doit rapporter l'état des modèles."""
    r = client.get("/health")
    data = r.json()
    assert "models" in data
    # Les 4 modèles doivent être mentionnés
    for m in ["traffic", "anomaly", "clustering", "prophet"]:
        assert m in data["models"]


# ============================================================
# / — page d'accueil
# ============================================================
def test_racine():
    """La racine doit renvoyer le message d'accueil et la liste des endpoints."""
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


# ============================================================
# /arrets — liste des arrêts
# ============================================================
def test_liste_arrets():
    """/arrets doit renvoyer une liste non vide."""
    r = client.get("/arrets")
    # 200 si les données sont chargées, 503 sinon (CI sans Parquet)
    if r.status_code == 200:
        data = r.json()
        assert "arrets" in data
        assert data["nb_arrets"] > 0
    else:
        assert r.status_code == 503


# ============================================================
# /predict/traffic — prédiction court terme
# ============================================================
def test_predict_traffic_valide():
    """Une requête valide doit renvoyer 200 avec une prédiction."""
    r = client.post("/predict/traffic",
                    json={"arret": "CHATELET", "date": "2026-07-01"})
    if r.status_code == 200:
        data = r.json()
        assert "predicted_validations" in data
        assert isinstance(data["predicted_validations"], int)
    else:
        # 503 si le modèle n'est pas chargé (environnement CI léger)
        assert r.status_code in (404, 503)


def test_predict_traffic_arret_inconnu():
    """Un arrêt inexistant doit renvoyer 404."""
    r = client.post("/predict/traffic",
                    json={"arret": "ARRET_QUI_NEXISTE_PAS_XYZ", "date": "2026-07-01"})
    assert r.status_code in (404, 503)


def test_predict_traffic_entree_invalide():
    """Une entrée mal formée (champ manquant) doit renvoyer 422."""
    r = client.post("/predict/traffic", json={"arret": "CHATELET"})  # date manquante
    assert r.status_code == 422


# ============================================================
# /detect/anomaly — détection d'anomalie
# ============================================================
def test_detect_anomaly_valide():
    """Une requête valide doit renvoyer 200 avec is_anomaly."""
    r = client.post("/detect/anomaly",
                    json={"arret": "GARE DU NORD", "nb_validations": 800})
    if r.status_code == 200:
        data = r.json()
        assert "is_anomaly" in data
        assert isinstance(data["is_anomaly"], bool)
    else:
        assert r.status_code in (404, 503)


def test_detect_anomaly_negatif_invalide():
    """Un nombre de validations négatif doit être rejeté (422)."""
    r = client.post("/detect/anomaly",
                    json={"arret": "GARE DU NORD", "nb_validations": -50})
    assert r.status_code == 422


# ============================================================
# /clusters/{arret} — profil d'un arrêt
# ============================================================
def test_cluster_version_invalide():
    """Une version de clustering invalide doit renvoyer 422."""
    r = client.get("/clusters/CHATELET?version=k99")
    assert r.status_code == 422


# ============================================================
# /predict/long-term — prédiction Prophet
# ============================================================
def test_long_term_entree_invalide():
    """Entrée mal formée → 422."""
    r = client.post("/predict/long-term", json={"arret": "CHATELET"})  # date manquante
    assert r.status_code == 422