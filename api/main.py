# ============================================================
# ÉTAPE 6 — API FastAPI — IDFM Mobility
# Fichier : api/main.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# Il expose les modèles ML comme des services REST.
# N'importe quelle application (Streamlit, Power BI, site web...)
# peut interroger l'API via des requêtes HTTP.
#
# ARCHITECTURE :
#   Requête HTTP → main.py → predict.py → modèles .pkl → réponse JSON
#
# L'API ne fait AUCUN calcul ML elle-même — elle appelle
# les fonctions de src/predict.py. Séparation des responsabilités.
#
# LES ENDPOINTS DE PRÉDICTION :
#   GET  /health              → l'API est-elle en ligne ?
#   POST /predict/traffic     → prédire le trafic (COURT terme, Random Forest)
#   POST /predict/long-term   → prédire le trafic (LONG terme, Prophet)
#   POST /detect/anomaly      → détecter une anomalie
#   GET  /clusters/{arret}    → profil d'un arrêt (k2 ou k5)
#
# LANCER L'API :
#   uvicorn api.main:app --reload
#   Puis ouvrir http://localhost:8000/docs (Swagger)
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging

# On importe nos fonctions de prédiction (le "pont" vers les modèles)
from src.predict import (
    predict_traffic,
    predict_long_term,
    predict_anomaly,
    predict_cluster,
    get_all_arrets,
    get_arret_stats,
    get_all_clusters,
    add_validations,
    _traffic_model,
    _anomaly_model,
    _cluster_model_k5,
    _prophet_feries,
    _df_historique
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# INITIALISATION DE L'API
# ============================================================
app = FastAPI(
    title="IDFM Mobility API",
    description="API de prédiction des flux de mobilité en Île-de-France. "
                "Prédiction de trafic (court et long terme), détection "
                "d'anomalies et clustering des arrêts.",
    version="1.1.0"
)

# CORS : autoriser les appels depuis d'autres domaines
# POURQUOI : Streamlit et Power BI tournent sur d'autres adresses.
# Sans CORS, le navigateur bloquerait leurs requêtes vers l'API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # en production, restreindre aux domaines connus
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODÈLES PYDANTIC — Validation des entrées/sorties
# ============================================================
# POURQUOI PYDANTIC ?
# Pydantic valide automatiquement les données reçues.
# Si quelqu'un envoie un mauvais type, FastAPI rejette
# la requête avec une erreur 422 claire — sans qu'on
# ait à écrire de vérifications manuelles.

class TrafficRequest(BaseModel):
    """Entrée pour la prédiction de trafic (court terme)."""
    arret: str = Field(..., description="Nom de l'arrêt en majuscules",
                       example="CHATELET")
    date: str  = Field(..., description="Date à prédire (YYYY-MM-DD)",
                       example="2026-07-01")


class LongTermRequest(BaseModel):
    """Entrée pour la prédiction long terme (Prophet)."""
    arret: str = Field(..., description="Nom de l'arrêt en majuscules",
                       example="CHATELET")
    date: str  = Field(..., description="Date lointaine à prédire (YYYY-MM-DD)",
                       example="2026-09-15")
    avec_courbe: bool = Field(False, description="Inclure l'historique et la "
                              "projection complète, pour tracer un graphique")


class AnomalyRequest(BaseModel):
    """Entrée pour la détection d'anomalie."""
    arret: str = Field(..., description="Nom de l'arrêt",
                       example="GARE DU NORD")
    nb_validations: int = Field(..., description="Nombre de validations observé",
                                example=800, ge=0)


class ValidationInput(BaseModel):
    """Entrée pour l'ajout de nouvelles validations dans MySQL."""
    JOUR: str = Field(..., description="Date (YYYY-MM-DD)", example="2026-07-05")
    LIBELLE_ARRET: str = Field(..., description="Nom de l'arrêt", example="CHATELET")
    CATEGORIE_TITRE: str = Field(..., description="Catégorie du titre de transport",
                                 example="Forfait Navigo")
    NB_VALD: int = Field(..., description="Nombre de validations", example=145000, ge=0)
    CODE_STIF_TRNS: Optional[str] = Field("100", description="Code transporteur")
    CODE_STIF_RES: Optional[str] = Field("110", description="Code réseau")
    CODE_STIF_ARRET: Optional[str] = Field("0", description="Code arrêt")
    ID_ZDC: Optional[int] = Field(0, description="ID zone de correspondance")


# Catégories de titres valides (pour information dans Swagger)
CATEGORIES_TITRES = [
    "Forfait Navigo", "Imagine R", "Amethyste", "Forfaits courts",
    "Contrat Solidarité Transport", "Autres titres", "NON DEFINI"
]


# ============================================================
# ENDPOINT 1 — GET /health
# ============================================================
@app.get("/health", tags=["Monitoring"])
def health():
    """
    Vérifie que l'API est en ligne et que les modèles sont chargés.

    UTILITÉ : monitoring, healthcheck Docker, vérification avant démo.
    Retourne l'état de chaque composant.
    """
    return {
        "status": "ok",
        "api": "online",
        "models": {
            "traffic": "loaded" if _traffic_model is not None else "missing",
            "anomaly": "loaded" if _anomaly_model is not None else "missing",
            "clustering": "loaded" if _cluster_model_k5 is not None else "missing",
            "prophet": "loaded" if _prophet_feries is not None else "missing"
        },
        "data": "loaded" if _df_historique is not None else "missing"
    }


# ============================================================
# ENDPOINT 2 — POST /predict/traffic  (COURT terme, Random Forest)
# ============================================================
@app.post("/predict/traffic", tags=["Prédiction"])
def endpoint_predict_traffic(request: TrafficRequest):
    """
    Prédit le nombre de validations pour un arrêt à une date PROCHE.

    Utilise le Random Forest, qui s'appuie sur le trafic récent (lags).
    Idéal pour le lendemain ou les jours suivant les dernières données.

    ENTRÉE (JSON) :
        {"arret": "CHATELET", "date": "2026-07-01"}

    SORTIE :
        Prédiction + score de confiance + top 3 facteurs SHAP
        (explicabilité : pourquoi cette prédiction ?)
    """
    try:
        result = predict_traffic(request.arret, request.date)
        return result
    except ValueError as e:
        # Arrêt inconnu → 404
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # Modèle non disponible → 503
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur predict_traffic : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne")


# ============================================================
# ENDPOINT 3 — POST /predict/long-term  (LONG terme, Prophet)
# ============================================================
@app.post("/predict/long-term", tags=["Prédiction"])
def endpoint_predict_long_term(request: LongTermRequest):
    """
    Prédit le trafic d'un arrêt à une date LOINTAINE (des mois à l'avance).

    Utilise Prophet, qui apprend une équation du temps :
        trafic(t) = tendance + saison_hebdo + saison_annuelle + fériés
    Contrairement au Random Forest, il n'a pas besoin du trafic récent —
    il évalue directement n'importe quelle date future.

    Le modèle est entraîné à la volée sur l'historique de l'arrêt (~0.5s).

    ENTRÉE (JSON) :
        {"arret": "CHATELET", "date": "2026-09-15", "avec_courbe": false}

    SORTIE :
        Prédiction + intervalle de confiance à 80% + décomposition
        (tendance, effet du jour de semaine, de la saison, des fériés).
        Si avec_courbe=true : historique + projection complète pour un graphe.

    REFUS EXPLICITE :
        Un arrêt avec moins de 180 jours d'historique ou moins de 500
        validations/jour est refusé (422) — la saisonnalité annuelle ne
        peut pas y être apprise de façon fiable.
    """
    try:
        result = predict_long_term(request.arret, request.date,
                                   avec_courbe=request.avec_courbe)
        return result
    except ValueError as e:
        # Arrêt inconnu OU sous les seuils (historique / trafic) → 422
        # 422 plutôt que 404 : l'arrêt peut exister mais être inéligible
        # à la prédiction long terme. Le message précise la raison.
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # Prophet ou calendrier non disponible → 503
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur predict_long_term : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne")


# ============================================================
# ENDPOINT 4 — POST /detect/anomaly
# ============================================================
@app.post("/detect/anomaly", tags=["Détection"])
def endpoint_detect_anomaly(request: AnomalyRequest):
    """
    Détecte si un nombre de validations est anormal pour un arrêt.

    ENTRÉE (JSON) :
        {"arret": "GARE DU NORD", "nb_validations": 800}

    SORTIE :
        is_anomaly + score + z_score + message explicatif
        (ex: "Trafic anormalement bas — possible grève")
    """
    try:
        result = predict_anomaly(request.arret, request.nb_validations)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur predict_anomaly : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne")


# ============================================================
# ENDPOINT 5 — GET /clusters/{arret}
# ============================================================
@app.get("/clusters/{arret}", tags=["Clustering"])
def endpoint_get_cluster(
    arret: str,
    version: str = Query("k5", description="Version : 'k5' (5 profils) ou 'k2' (2 groupes)")
):
    """
    Retourne le profil (cluster) d'un arrêt et ses arrêts similaires.

    ENTRÉE :
        arret dans l'URL : /clusters/CHATELET
        version en query : /clusters/CHATELET?version=k2

    SORTIE :
        cluster + profil métier + 5 arrêts similaires + caractéristiques

    DEUX VUES :
        k5 (défaut) → vue détaillée (Méga-hubs, Résidentiel, etc.)
        k2          → vue simple (Méga-hubs vs Réseau standard)
    """
    if version not in ("k2", "k5"):
        raise HTTPException(status_code=422,
                            detail="version doit être 'k2' ou 'k5'")
    try:
        result = predict_cluster(arret, version=version)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur predict_cluster : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne")


# ============================================================
# ENDPOINT 6 — GET /arrets
# ============================================================
@app.get("/arrets", tags=["Données"])
def endpoint_get_arrets():
    """
    Retourne la liste de tous les arrêts disponibles.

    UTILITÉ : remplir les listes déroulantes de Streamlit.
    """
    try:
        arrets = get_all_arrets()
        return {"nb_arrets": len(arrets), "arrets": arrets}
    except Exception as e:
        logger.error(f"Erreur get_arrets : {e}")
        raise HTTPException(status_code=503, detail=str(e))


# ============================================================
# ENDPOINT 7 — GET /stats/{arret}
# ============================================================
@app.get("/stats/{arret}", tags=["Données"])
def endpoint_get_stats(arret: str):
    """
    Retourne les statistiques historiques d'un arrêt
    (moyenne, min, max, nombre de jours...).
    """
    try:
        return get_arret_stats(arret)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur get_stats : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne")


# ============================================================
# ENDPOINT 8 — GET /clusters (liste des clusters)
# ============================================================
@app.get("/clusters", tags=["Clustering"])
def endpoint_get_all_clusters(
    version: str = Query("k5", description="Version : 'k5' ou 'k2'")
):
    """
    Retourne la liste des clusters et leur composition
    (profil, nombre d'arrêts, exemples).

    UTILITÉ : vue d'ensemble pour le dashboard et la carte.
    """
    if version not in ("k2", "k5"):
        raise HTTPException(status_code=422, detail="version doit être 'k2' ou 'k5'")
    try:
        return get_all_clusters(version=version)
    except Exception as e:
        logger.error(f"Erreur get_all_clusters : {e}")
        raise HTTPException(status_code=503, detail=str(e))


# ============================================================
# ENDPOINT 9 — POST /add-validations
# ============================================================
@app.post("/add-validations", tags=["Données"])
def endpoint_add_validations(validation: ValidationInput):
    """
    Ajoute une nouvelle validation dans MySQL (le brut).

    Le Parquet enrichi devra ensuite être régénéré (preprocessing.py)
    pour que les modèles prennent en compte cette nouvelle donnée.

    CATÉGORIES VALIDES :
        Forfait Navigo, Imagine R, Amethyste, Forfaits courts,
        Contrat Solidarité Transport, Autres titres, NON DEFINI
    """
    try:
        result = add_validations(validation.dict())
        return result
    except Exception as e:
        logger.error(f"Erreur add_validations : {e}")
        raise HTTPException(status_code=500,
                            detail=f"Erreur lors de l'insertion : {e}")


# ============================================================
# ENDPOINT RACINE — Message d'accueil
# ============================================================
@app.get("/", tags=["Accueil"])
def root():
    """Page d'accueil de l'API avec la liste des endpoints."""
    return {
        "message": "Bienvenue sur l'API IDFM Mobility",
        "documentation": "/docs",
        "endpoints": {
            "health": "GET /health",
            "prediction_trafic_court": "POST /predict/traffic",
            "prediction_trafic_long": "POST /predict/long-term",
            "detection_anomalie": "POST /detect/anomaly",
            "cluster_arret": "GET /clusters/{arret}?version=k5",
            "liste_arrets": "GET /arrets",
            "stats_arret": "GET /stats/{arret}",
            "liste_clusters": "GET /clusters?version=k5",
            "ajout_donnees": "POST /add-validations"
        }
    }


# ============================================================
# LANCEMENT DIRECT (développement)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    # host 0.0.0.0 = accessible depuis l'extérieur (nécessaire pour Docker)
    uvicorn.run(app, host="0.0.0.0", port=8000)